"""Band-parallel compositing.

The document is split into horizontal bands, each composited independently
into its slice of a shared output buffer. Bands are a natural unit here
because a band is just a normal composite with a shifted origin -- no
special-casing inside the compositor, and the result is bit-identical to
compositing the whole canvas at once (``tests/test_parallel_compositor.py``
asserts exactly that).

Why threads work despite the GIL
--------------------------------
NumPy releases the GIL inside its large ufunc loops, which is where
essentially all of a composite's time is spent. Measured on 20 layers at
1600x1000 (``bench/spike_parallel.py``):

    1 thread      93.9 ms
    8 threads     14.7 ms   (6.4x)

Efficiency peaks around 8 threads with 64-row bands. More threads
oversubscribe against OpenCV's own pool; taller bands leave cores idle.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..core.layer_stack import LayerStack
from .planar_compositor import PlanarCompositor

# Measured optimum on a 14-core M4 Pro; see the module docstring. Kept
# modest deliberately: OpenCV runs its own thread pool inside the same
# process, and oversubscribing the two costs more than it gains.
DEFAULT_BAND_HEIGHT = 64
DEFAULT_MAX_WORKERS = 8

# Below this many pixels the thread-dispatch overhead dominates.
MIN_PARALLEL_PIXELS = 512 * 512

# Bytes of prepared layer data one frame needs, as a multiple of the layer
# cache budget, above which banding becomes counter-productive: every band
# re-prepares the layers the previous band just evicted, so the thrash is
# multiplied by the band count rather than paid once.
WORKING_SET_LIMIT = 1.0


class ParallelCompositor:
    """Composites a layer stack across several threads, band by band."""

    def __init__(
        self,
        cache=None,
        max_workers: int | None = None,
        band_height: int = DEFAULT_BAND_HEIGHT,
    ) -> None:
        cpu = os.cpu_count() or 4
        self._max_workers = max_workers or min(DEFAULT_MAX_WORKERS, cpu)
        self._band_height = band_height
        self._cache = cache
        # One compositor per worker: each owns its scratch pool, so bands
        # never contend for buffers. The layer cache is shared and is
        # internally synchronised.
        self._compositors = [
            PlanarCompositor(cache=cache) for _ in range(self._max_workers)
        ]
        self._pool: ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------

    def _ensure_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="basera-composite",
            )
        return self._pool

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    # ------------------------------------------------------------------

    def composite(self, stack: LayerStack, width: int, height: int,
                  out: np.ndarray | None = None,
                  level: int = 0) -> np.ndarray:
        """Composite *stack* into a planar (4, height, width) buffer.

        *width* / *height* are output pixels; at *level* > 0 that is the
        scaled size, not the document size.
        """
        if out is None:
            out = np.zeros((4, height, width), dtype=np.float32)

        if (width * height < MIN_PARALLEL_PIXELS
                or self._max_workers <= 1
                or height <= self._band_height
                or self._would_thrash(stack, width, height)):
            self._compositors[0].composite(stack, width, height, out=out,
                                           level=level)
            return out

        bands = [
            (y, min(height, y + self._band_height))
            for y in range(0, height, self._band_height)
        ]

        # Warm the shared layer cache single-threaded first. Otherwise every
        # band races to prepare the same layers and each does the work.
        self._prewarm(stack, level)

        def render_band(index_band):
            index, (y0, y1) = index_band
            comp = self._compositors[index % self._max_workers]
            comp.composite(stack, width, y1 - y0, origin=(0, y0),
                           out=out[:, y0:y1, :], level=level)

        pool = self._ensure_pool()
        list(pool.map(render_band, enumerate(bands)))
        return out

    def _would_thrash(self, stack: LayerStack, width: int, height: int) -> bool:
        """True when this frame's prepared layers cannot all be cached.

        Banding only pays off if a layer prepared for the first band is
        still cached for the thirty-fourth. When the working set exceeds
        the cache, serial compositing prepares each layer once per frame,
        while banding prepares it once per *band* -- measured at 4K with
        twenty layers, that is 588 ms serial against 2,194 ms banded.
        """
        cache = self._cache
        if cache is None:
            return False
        budget = getattr(cache, "_budget", 0)
        if budget <= 0:
            return False
        # 4 channels x float32 per output pixel, per visible layer.
        per_layer = width * height * 4 * 4
        n_visible = sum(1 for l in stack if l.visible)
        return (per_layer * n_visible) > (budget * WORKING_SET_LIMIT)

    def _prewarm(self, stack: LayerStack, level: int = 0) -> None:
        """Populate the layer cache on one thread before fanning out."""
        if self._cache is None:
            return
        # A 1x1 composite walks the same preparation path as a full one but
        # writes almost nothing, so it fills the cache cheaply.
        self._compositors[0].composite(stack, 1, 1, origin=(0, 0), level=level)


class _SerialFallback:
    """Adapter so callers can hold a single 'compositor' object."""

    def __init__(self, cache=None) -> None:
        self._inner = PlanarCompositor(cache=cache)

    def composite(self, stack, width, height, out=None):
        return self._inner.composite(stack, width, height, out=out)

    def shutdown(self) -> None:
        pass
