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
from .layer_cache import LayerRasterCache
from .parallel_compositor import ParallelCompositor
from .tile_cache import TileCache


class RenderPipeline:
    """Full pipeline: layer compositing -> adjustment layers -> output."""

    # Twenty 4K layers prepared at level 1 (1920x1080) need ~663 MB. A
    # smaller budget does not merely cache less -- it thrashes, re-preparing
    # every layer on every frame, which measured 421 ms/frame against 35 ms
    # with the working set resident.
    DEFAULT_CACHE_MB = 1280

    def __init__(self, cache_budget_mb: int | None = None,
                 max_workers: int | None = None) -> None:
        if cache_budget_mb is None:
            cache_budget_mb = self.DEFAULT_CACHE_MB
        self._layer_cache = LayerRasterCache(budget_bytes=cache_budget_mb << 20)
        self._compositor = ParallelCompositor(
            cache=self._layer_cache, max_workers=max_workers)
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
        if lroi is None:
            result = self._compositor.composite(
                document.layers, ow, oh, level=level)
        else:
            rx, ry, rw, rh = lroi
            result = self._compositor.composite(
                document.layers, rw, rh, level=level,
                origin=(rx, ry), frame_roi=lroi)
        if epoch == self._epoch:
            self._result_planar = result
            self._planar_valid = True
            self._result_level = level
            self._result_roi = lroi
        return result

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
        if epoch == self._epoch:
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
        else:
            self._layer_cache.invalidate(layer_id)
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
    Rounding halves the maximum error and removes the bias.

    cv2.merge is ~4x faster than a NumPy transpose for this shape, so the
    planar-to-interleaved step goes through it where OpenCV is present.
    """
    try:
        import cv2
        scaled = cv2.merge([planar[0], planar[1], planar[2], planar[3]])
    except ImportError:
        scaled = np.ascontiguousarray(planar.transpose(1, 2, 0))
    np.multiply(scaled, 255.0, out=scaled)
    np.add(scaled, 0.5, out=scaled)
    np.clip(scaled, 0.0, 255.0, out=scaled)
    np.copyto(out, scaled, casting="unsafe")
