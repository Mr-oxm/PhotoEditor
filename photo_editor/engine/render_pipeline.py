"""Render pipeline -- orchestrates the compositor and caches its output.

The pipeline owns the planar compositor and the layer raster cache, and is
the single place that decides whether a frame can be served from cache.

Threading
---------
A pipeline instance is **not** thread-safe. Each consumer that renders
concurrently (the interactive canvas, an export job) must own its own
instance, or serialise access. Sharing one instance across the render
thread pool and the save/export worker was a genuine data race: both
mutated ``_uint8_buf``, the tile cache and the compositor's scratch pool.
"""

from __future__ import annotations

import numpy as np

from ..blending.planar import to_interleaved
from ..core.document import Document
from ..core.memory_budget import render_cache_budget
from .layer_cache import LayerRasterCache
from .parallel_compositor import ParallelCompositor
from .planar_compositor import PlanarCompositor
from .sandwich_cache import SandwichCache
from .tile_cache import TileCache


class RenderPipeline:
    """Full pipeline: layer compositing -> adjustment layers -> output."""

    def __init__(self, cache_budget_mb: int | None = None,
                 max_workers: int | None = None) -> None:
        # Twenty 4K layers prepared at level 1 (1920x1080) need ~663 MB.
        # Undersizing this does not merely cache less -- it thrashes,
        # re-preparing every layer on every frame, which measured
        # 421 ms/frame against 35 ms with the working set resident. The
        # default therefore scales with the machine.
        if cache_budget_mb is None:
            budget = render_cache_budget()
        else:
            budget = cache_budget_mb << 20
        self._layer_cache = LayerRasterCache(budget_bytes=budget)
        self._compositor = ParallelCompositor(
            cache=self._layer_cache, max_workers=max_workers)
        # Interactive path: reuses the composited layers below the one being
        # dragged, so a drag frame is a copy plus one blend rather than a
        # full walk of the stack.
        self._sandwich = SandwichCache()
        self._interactive = PlanarCompositor(
            cache=self._layer_cache, sandwich=self._sandwich)
        self._focus_layer_id: str | None = None
        self._tile_cache = TileCache(tile_size=256)
        self._last_width = 0
        self._last_height = 0
        # Cached outputs
        self._result_planar: np.ndarray | None = None
        self._planar_valid: bool = False
        self._result_level: int = 0
        # Bumped by every invalidate(). A render captures it before starting
        # and refuses to publish its result if it changed meanwhile --
        # otherwise a render that began before an edit would mark its
        # now-stale output as the valid cache, and the *next* render would
        # return it without recompositing.
        self._epoch: int = 0
        self._result_roi: tuple[int, int, int, int] | None = None
        self._result_uint8: np.ndarray | None = None
        self._uint8_valid: bool = False
        self._uint8_buf: np.ndarray | None = None

    # ---- Composite ---------------------------------------------------------

    def execute_planar(self, document: Document, level: int = 0,
                       roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """Composite to a planar (4, H, W) float32 buffer at *level*.

        Level L renders at 1/2**L scale; level 0 is full resolution.

        *roi* is a document-space rectangle (x, y, w, h). Only that region
        is composited, which is what makes a zoomed-in view of a large
        document affordable: at 100% zoom a 4K document shows perhaps
        0.25 Mpx of its 8.3 Mpx, and rendering the rest is pure waste.
        """
        w, h = document.width, document.height
        if w != self._last_width or h != self._last_height:
            self._tile_cache.initialize(w, h)
            self._last_width, self._last_height = w, h

        ow, oh = level_size(w, h, level)
        lroi = _roi_to_level(roi, level, ow, oh) if roi is not None else None

        if (self._planar_valid and self._result_planar is not None
                and self._result_level == level
                and self._result_roi == lroi):
            return self._result_planar

        epoch = self._epoch
        origin = (0, 0) if lroi is None else (lroi[0], lroi[1])
        cw, ch = (ow, oh) if lroi is None else (lroi[2], lroi[3])

        if self._focus_layer_id is not None:
            # Interactive: single-threaded so the sandwich caches whole
            # frames rather than per-band slivers.
            self._interactive.focus_layer_id = self._focus_layer_id
            self._prime_sandwich(document, cw, ch, level, origin, lroi)
            result = self._interactive.composite(
                document.layers, cw, ch, level=level,
                origin=origin, frame_roi=lroi)
        else:
            result = self._compositor.composite(
                document.layers, cw, ch, level=level,
                origin=origin, frame_roi=lroi)
        if True:  # MUTATION
            self._result_planar = result
            self._planar_valid = True
            self._result_level = level
            self._result_roi = lroi
        return result

    def _prime_sandwich(self, document, cw, ch, level, origin, lroi) -> None:
        """Make sure the under-cache covers the current focus and view."""
        try:
            self._interactive.prime_sandwich(
                document.layers, cw, ch, self._focus_layer_id,
                level=level, origin=origin, frame_roi=lroi)
        except Exception:
            # Caching is an optimisation: never let it break a render.
            self._sandwich.clear()

    def begin_interaction(self, layer_id: str | None) -> None:
        """Enter the interactive path, editing *layer_id*.

        While set, everything below that layer is composited once and then
        reused, so a drag frame costs a copy plus one blend instead of a
        full walk of the stack.
        """
        if layer_id != self._focus_layer_id:
            self._sandwich.clear()
        self._focus_layer_id = layer_id

    def end_interaction(self) -> None:
        self._focus_layer_id = None
        self._sandwich.clear()

    @property
    def sandwich(self) -> SandwichCache:
        return self._sandwich

    def preview_level(self, document: Document, max_size: int) -> int:
        """Mip level whose longest side fits within *max_size* pixels."""
        return LayerRasterCache.choose_level(
            document.width, document.height, max_size)

    def execute(self, document: Document) -> np.ndarray:
        """Composite to an interleaved (H, W, 4) float32 buffer.

        Kept for callers that expect the historical interleaved contract
        (export, clipboard, tests). Interactive rendering should prefer
        :meth:`execute_to_uint8`, which avoids this conversion.
        """
        return to_interleaved(self.execute_planar(document))

    def execute_to_uint8(self, document: Document, level: int = 0,
                         out: np.ndarray | None = None,
                         roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """Return the composite as uint8 RGBA, cached between invalidations.

        *out*, when supplied and correctly shaped, receives the result. The
        render scheduler passes alternating buffers so the frame the UI
        thread is converting to a QPixmap is never the one being written.
        """
        if (out is None and self._uint8_valid
                and self._result_uint8 is not None
                and self._result_level == level
                and roi is None and self._result_roi is None):
            return self._result_uint8
        epoch = self._epoch
        planar = self.execute_planar(document, level=level, roi=roi)
        h, w = planar.shape[1], planar.shape[2]
        shape = (h, w, 4)
        if out is not None and out.shape == shape and out.dtype == np.uint8:
            _planar_to_uint8(planar, out)
            return out
        if self._uint8_buf is None or self._uint8_buf.shape != shape:
            self._uint8_buf = np.empty(shape, dtype=np.uint8)
        _planar_to_uint8(planar, self._uint8_buf)
        if True:  # MUTATION
            self._result_uint8 = self._uint8_buf
            self._uint8_valid = True
            self._result_level = level
        if out is not None:
            return np.array(self._uint8_buf, copy=True)
        return self._uint8_buf

    # ---- Invalidation ------------------------------------------------------

    def invalidate(self, layer_id: str | None = None) -> None:
        """Mark the composite stale.

        *layer_id* narrows the layer-raster cache eviction to that layer;
        the composite itself is always recomputed because the compositing
        order and the layers above it may depend on it.
        """
        self._epoch += 1
        self._planar_valid = False
        self._uint8_valid = False
        self._result_planar = None
        self._result_roi = None
        if layer_id is None:
            self._layer_cache.clear()
            self._sandwich.clear()
        else:
            self._layer_cache.invalidate(layer_id)
            # The sandwich keys off the signatures of the layers it covers,
            # so an edit to one of them invalidates it automatically -- no
            # need to drop it here, which is what makes a drag cheap.
        self._tile_cache.invalidate_all()

    def invalidate_region(self, x: int, y: int, width: int, height: int) -> None:
        """Mark tiles overlapping (x, y, width, height) dirty."""
        self._epoch += 1
        self._tile_cache.invalidate_region(x, y, width, height)
        self._planar_valid = False
        self._uint8_valid = False
        self._result_planar = None
        self._result_roi = None

    # ---- Introspection -----------------------------------------------------

    @property
    def layer_cache(self) -> LayerRasterCache:
        return self._layer_cache

    def cache_stats(self) -> dict:
        return self._layer_cache.stats()

    def shutdown(self) -> None:
        """Release the compositor's worker threads."""
        self._compositor.shutdown()


def _roi_to_level(roi, level: int, ow: int, oh: int):
    """Map a document-space ROI onto the level grid, rounded outward.

    Rounding outward matters: a ROI that rounds inward loses a row or
    column of pixels at the edge, which shows up as a flickering seam when
    panning.
    """
    import math
    x, y, w, h = roi
    scale = 1.0 / (1 << level)
    rx = max(0, int(math.floor(x * scale)))
    ry = max(0, int(math.floor(y * scale)))
    rx2 = min(ow, int(math.ceil((x + w) * scale)))
    ry2 = min(oh, int(math.ceil((y + h) * scale)))
    rw, rh = max(1, rx2 - rx), max(1, ry2 - ry)
    if rx == 0 and ry == 0 and rw >= ow and rh >= oh:
        return None          # Covers everything -- no point tracking a ROI.
    return (rx, ry, rw, rh)


def level_roi_to_document(lroi, level: int) -> tuple[int, int, int, int]:
    """Inverse of :func:`_roi_to_level` -- level pixels back to document space."""
    rx, ry, rw, rh = lroi
    f = 1 << level
    return (rx * f, ry * f, rw * f, rh * f)


def level_size(width: int, height: int, level: int) -> tuple[int, int]:
    """Output size of a composite rendered at mip *level*."""
    if level <= 0:
        return width, height
    scale = 1.0 / (1 << level)
    return (max(1, int(round(width * scale))),
            max(1, int(round(height * scale))))


def _planar_to_uint8(planar: np.ndarray, out: np.ndarray) -> None:
    """Convert planar float32 [0,1] to interleaved uint8 RGBA in *out*.

    Rounds rather than truncates. Casting ``value * 255`` straight to uint8
    truncates, which biases every channel of every exported and displayed
    pixel downward by up to one level -- 0.25 came out as 63 instead of 64.

    This runs on every displayed frame, so it is worth doing in one pass per
    plane rather than four over the whole interleaved buffer. Measured at
    1920x1080: 6.99 ms for merge-then-four-NumPy-passes, 0.82 ms this way.

    ``convertScaleAbs`` scales, rounds, saturates and casts in a single
    threaded pass. Its absolute value is not a concern here: the compositor
    clamps blend results to [0, 1] and Porter-Duff 'over' cannot produce a
    negative from non-negative inputs, which is asserted directly in
    tests/test_render_fidelity.py.
    """
    try:
        import cv2
    except ImportError:
        scaled = np.ascontiguousarray(planar.transpose(1, 2, 0))
        np.multiply(scaled, 255.0, out=scaled)
        np.add(scaled, 0.5, out=scaled)
        np.clip(scaled, 0.0, 255.0, out=scaled)
        np.copyto(out, scaled, casting="unsafe")
        return

    planes = _uint8_plane_buffers(planar.shape[1], planar.shape[2])
    for i in range(4):
        cv2.convertScaleAbs(planar[i], dst=planes[i], alpha=255.0)
    cv2.merge(planes, out)


_PLANE_CACHE: dict[tuple[int, int], list] = {}


def _uint8_plane_buffers(height: int, width: int) -> list:
    """Reusable single-channel scratch planes, so the hot path allocates none."""
    key = (height, width)
    planes = _PLANE_CACHE.get(key)
    if planes is None:
        planes = [np.empty((height, width), dtype=np.uint8) for _ in range(4)]
        if len(_PLANE_CACHE) > 4:
            _PLANE_CACHE.pop(next(iter(_PLANE_CACHE)))
        _PLANE_CACHE[key] = planes
    return planes
