"""Planar RGBA buffers and the planar 'over' compositing operator.

Layout
------
A planar buffer is a C-contiguous ``(4, H, W)`` float32 array holding
straight-alpha RGBA. ``buf[:3]`` is a contiguous ``(3, H, W)`` colour block
and ``buf[3]`` is a contiguous ``(H, W)`` alpha plane.

The alpha plane being contiguous is the whole point: broadcasting it against
the colour block, ``(3, H, W) * (H, W)``, has a contiguous inner loop, so no
weight array ever has to be materialised. The interleaved equivalent needs
either a strided op (17.6 ms/4K) or a materialised ``(H, W, 4)`` weight
(17.9 ms/4K); the planar form costs 0.6 ms.

Semantics are identical to ``BlendingEngine`` -- straight-alpha Porter-Duff
'over' with an optional mask and layer opacity. This module changes the
memory layout and nothing else.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ..core.enums import BlendMode
from .planar_modes import POSITION_DEPENDENT, get_planar_blend_func

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Layout conversion
# ---------------------------------------------------------------------------

def to_planar(interleaved: np.ndarray) -> np.ndarray:
    """(H, W, 4) -> contiguous (4, H, W). Copies."""
    if interleaved.dtype != np.float32:
        interleaved = interleaved.astype(np.float32)
    return np.ascontiguousarray(interleaved.transpose(2, 0, 1))


def to_interleaved(planar: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """(4, H, W) -> (H, W, 4). Uses cv2.merge, which is ~4x faster than
    NumPy's transpose+copy for this shape (2.7 ms vs 10.2 ms at 4K)."""
    try:
        import cv2
        merged = cv2.merge([planar[0], planar[1], planar[2], planar[3]])
        if out is not None:
            np.copyto(out, merged)
            return out
        return merged
    except ImportError:
        result = np.ascontiguousarray(planar.transpose(1, 2, 0))
        if out is not None:
            np.copyto(out, result)
            return out
        return result


def empty_planar(height: int, width: int) -> np.ndarray:
    return np.empty((4, height, width), dtype=np.float32)


def zeros_planar(height: int, width: int) -> np.ndarray:
    return np.zeros((4, height, width), dtype=np.float32)


# ---------------------------------------------------------------------------
# The 'over' operator
# ---------------------------------------------------------------------------

def blend_planar_region(
    canvas: np.ndarray,
    over: np.ndarray,
    position: tuple[int, int],
    mode: BlendMode = BlendMode.NORMAL,
    opacity: float = 1.0,
    mask: np.ndarray | None = None,
    scratch: "PlanarScratch | None" = None,
    abs_origin: tuple[int, int] = (0, 0),
) -> None:
    """Blend planar *over* into planar *canvas* at *position*, in place.

    Parameters
    ----------
    canvas : (4, H, W) float32, modified in place.
    over : (4, h, w) float32 straight-alpha source at its native size.
    position : (x, y) top-left of *over* on the canvas.
    mode, opacity : blend mode and layer opacity.
    mask : optional (h, w) float32 mask in the *source's* coordinate space.
    scratch : optional reusable buffers, to avoid per-call allocation.
    abs_origin : absolute document position of *canvas*'s top-left. Only
        position-dependent modes (Dissolve) use it, but they need it: a
        pattern derived from the block's own size differs between a
        band-parallel render and a whole-canvas one.
    """
    ch, cw = canvas.shape[1], canvas.shape[2]
    lh, lw = over.shape[1], over.shape[2]
    lx, ly = position

    # Overlapping rectangle between source and canvas.
    sx, sy = max(0, -lx), max(0, -ly)
    dx, dy = max(0, lx), max(0, ly)
    w = min(lw - sx, cw - dx)
    h = min(lh - sy, ch - dy)
    if w <= 0 or h <= 0:
        return

    if mask is not None:
        mh, mw = mask.shape[:2]
        w = min(w, mw - sx)
        h = min(h, mh - sy)
        if w <= 0 or h <= 0:
            return

    base = canvas[:, dy:dy + h, dx:dx + w]
    src = over[:, sy:sy + h, sx:sx + w]

    # Effective source alpha = alpha * opacity * mask.
    src_a = src[3]
    if opacity < 1.0 or mask is not None:
        over_a = src_a * np.float32(opacity) if opacity < 1.0 else src_a.copy()
        if mask is not None:
            over_a *= mask[sy:sy + h, sx:sx + w]
    else:
        over_a = src_a

    base_a = base[3]

    # inv = base_a * (1 - over_a);  out_a = over_a + inv
    inv = np.subtract(np.float32(1.0), over_a)
    inv *= base_a
    out_a = np.add(over_a, inv)

    if mode == BlendMode.NORMAL:
        contrib = src[:3] * over_a
    else:
        fn = get_planar_blend_func(mode)
        if mode in POSITION_DEPENDENT:
            blended = fn(base[:3], src[:3],
                         (abs_origin[1] + dy, abs_origin[0] + dx))
        else:
            blended = fn(base[:3], src[:3])
        blended = np.clip(blended, 0.0, 1.0)
        contrib = blended * over_a

    # base_rgb = (base_rgb*inv + contrib) / max(out_a, eps)
    colour = base[:3]
    colour *= inv
    colour += contrib
    colour /= np.maximum(out_a, _EPS)
    base[3] = out_a


def multiply_alpha_planar(planar: np.ndarray, factor: np.ndarray) -> None:
    """Attenuate only the alpha plane in place (contiguous)."""
    planar[3] *= factor


def multiply_all_planar(planar: np.ndarray, factor: np.ndarray) -> None:
    """Attenuate every channel in place (contiguous, no materialised weight)."""
    planar *= factor


class PlanarScratch:
    """Reusable planar scratch buffers keyed by shape.

    The compositor allocates document-sized temporaries for clipping masks,
    groups and placed layers. Reusing them removes the dominant source of
    allocation churn during interactive rendering.
    """

    def __init__(self, max_per_shape: int = 3,
                 max_bytes: int = 256 << 20) -> None:
        self._max = max_per_shape
        # Bounded by TOTAL bytes as well as per-shape depth. The pool is
        # keyed by shape, and every distinct viewport size and mip level
        # produces a new key -- a couple of minutes of zooming and resizing
        # a 4K document created hundreds of them and the pool grew without
        # limit. Least-recently-used shapes are dropped first.
        self._max_bytes = max_bytes
        self._bytes = 0
        self._pools: "OrderedDict[tuple, list[np.ndarray]]" = OrderedDict()

    def acquire(self, shape: tuple[int, ...],
                dtype=np.float32, zero: bool = True) -> np.ndarray:
        key = (tuple(shape), np.dtype(dtype))
        pool = self._pools.get(key)
        if pool:
            self._pools.move_to_end(key)
            buf = pool.pop()
            self._bytes -= buf.nbytes
            if zero:
                buf.fill(0)
            return buf
        return (np.zeros(shape, dtype=dtype) if zero
                else np.empty(shape, dtype=dtype))

    def release(self, buf: np.ndarray | None) -> None:
        if buf is None:
            return
        key = (tuple(buf.shape), buf.dtype)
        pool = self._pools.setdefault(key, [])
        if len(pool) >= self._max:
            return              # this shape is already well stocked
        pool.append(buf)
        self._bytes += buf.nbytes
        self._pools.move_to_end(key)
        self._evict()

    def _evict(self) -> None:
        """Drop least-recently-used shapes until the pool fits its budget."""
        while self._bytes > self._max_bytes and len(self._pools) > 1:
            _, dropped = self._pools.popitem(last=False)
            for buf in dropped:
                self._bytes -= buf.nbytes

    def nbytes(self) -> int:
        return self._bytes

    def clear(self) -> None:
        self._pools.clear()
        self._bytes = 0
