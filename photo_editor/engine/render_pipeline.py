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

    def __init__(self, cache_budget_mb: int = 512,
                 max_workers: int | None = None) -> None:
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
        self._result_uint8: np.ndarray | None = None
        self._uint8_valid: bool = False
        self._uint8_buf: np.ndarray | None = None

    # ---- Composite ---------------------------------------------------------

    def execute_planar(self, document: Document, level: int = 0) -> np.ndarray:
        """Composite to a planar (4, H, W) float32 buffer at *level*.

        Level L renders at 1/2**L scale. Level 0 is full resolution.
        """
        w, h = document.width, document.height
        if w != self._last_width or h != self._last_height:
            self._tile_cache.initialize(w, h)
            self._last_width, self._last_height = w, h
        if (self._planar_valid and self._result_planar is not None
                and self._result_level == level):
            return self._result_planar
        ow, oh = level_size(w, h, level)
        result = self._compositor.composite(
            document.layers, ow, oh, level=level)
        self._result_planar = result
        self._planar_valid = True
        self._result_level = level
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

    def execute_to_uint8(self, document: Document, level: int = 0) -> np.ndarray:
        """Return the composite as uint8 RGBA, cached between invalidations."""
        if (self._uint8_valid and self._result_uint8 is not None
                and self._result_level == level):
            return self._result_uint8
        planar = self.execute_planar(document, level=level)
        h, w = planar.shape[1], planar.shape[2]
        shape = (h, w, 4)
        if self._uint8_buf is None or self._uint8_buf.shape != shape:
            self._uint8_buf = np.empty(shape, dtype=np.uint8)
        _planar_to_uint8(planar, self._uint8_buf)
        self._result_uint8 = self._uint8_buf
        self._uint8_valid = True
        return self._result_uint8

    # ---- Invalidation ------------------------------------------------------

    def invalidate(self, layer_id: str | None = None) -> None:
        """Mark the composite stale.

        *layer_id* narrows the layer-raster cache eviction to that layer;
        the composite itself is always recomputed because the compositing
        order and the layers above it may depend on it.
        """
        self._planar_valid = False
        self._uint8_valid = False
        self._result_planar = None
        if layer_id is None:
            self._layer_cache.clear()
        else:
            self._layer_cache.invalidate(layer_id)
        self._tile_cache.invalidate_all()

    def invalidate_region(self, x: int, y: int, width: int, height: int) -> None:
        """Mark tiles overlapping (x, y, width, height) dirty."""
        self._tile_cache.invalidate_region(x, y, width, height)
        self._planar_valid = False
        self._uint8_valid = False
        self._result_planar = None

    # ---- Introspection -----------------------------------------------------

    @property
    def layer_cache(self) -> LayerRasterCache:
        return self._layer_cache

    def cache_stats(self) -> dict:
        return self._layer_cache.stats()

    def shutdown(self) -> None:
        """Release the compositor's worker threads."""
        self._compositor.shutdown()


def level_size(width: int, height: int, level: int) -> tuple[int, int]:
    """Output size of a composite rendered at mip *level*."""
    if level <= 0:
        return width, height
    scale = 1.0 / (1 << level)
    return (max(1, int(round(width * scale))),
            max(1, int(round(height * scale))))


def _planar_to_uint8(planar: np.ndarray, out: np.ndarray) -> None:
    """Convert planar float32 [0,1] to interleaved uint8 RGBA in *out*.

    cv2.merge is ~4x faster than a NumPy transpose for this shape, so the
    conversion and the scale are fused through it where OpenCV is present.
    """
    try:
        import cv2
        scaled = cv2.merge([planar[0], planar[1], planar[2], planar[3]])
        np.multiply(scaled, 255.0, out=out, casting="unsafe")
    except ImportError:
        interleaved = np.ascontiguousarray(planar.transpose(1, 2, 0))
        np.multiply(interleaved, 255.0, out=out, casting="unsafe")
