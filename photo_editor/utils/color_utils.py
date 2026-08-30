"""Color-space conversion utilities (vectorised NumPy)."""

from __future__ import annotations

import numpy as np


def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0,1] → HSL [0,1].  Last axis = 3."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    l = (mx + mn) * 0.5
    d = mx - mn
    s = np.where(d > 0, np.where(l > 0.5, d / (2 - mx - mn + 1e-10), d / (mx + mn + 1e-10)), 0)
    h = np.zeros_like(l)
    safe = np.where(d > 0, d, 1)
    h = np.where((mx == r) & (d > 0), ((g - b) / safe) % 6, h)
    h = np.where((mx == g) & (d > 0), (b - r) / safe + 2, h)
    h = np.where((mx == b) & (d > 0), (r - g) / safe + 4, h)
    h /= 6.0
    return np.stack([h, s, l], axis=-1).astype(np.float32)


def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """Convert HSL [0,1] → RGB [0,1]."""
    h, s, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]
    c = (1 - np.abs(2 * l - 1)) * s
    x = c * (1 - np.abs((h * 6) % 2 - 1))
    m = l - c * 0.5
    h6 = (h * 6).astype(np.int32) % 6
    z = np.zeros_like(c)
    r = np.select([h6 == 0, h6 == 1, h6 == 2, h6 == 3, h6 == 4, h6 == 5], [c, x, z, z, x, c])
    g = np.select([h6 == 0, h6 == 1, h6 == 2, h6 == 3, h6 == 4, h6 == 5], [x, c, c, x, z, z])
    b = np.select([h6 == 0, h6 == 1, h6 == 2, h6 == 3, h6 == 4, h6 == 5], [z, z, x, c, c, x])
    return np.clip(np.stack([r + m, g + m, b + m], axis=-1), 0, 1).astype(np.float32)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0,1] → HSV [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    v = mx
    s = np.where(mx > 0, d / (mx + 1e-10), 0)
    safe = np.where(d > 0, d, 1)
    h = np.zeros_like(v)
    h = np.where((mx == r) & (d > 0), ((g - b) / safe) % 6, h)
    h = np.where((mx == g) & (d > 0), (b - r) / safe + 2, h)
    h = np.where((mx == b) & (d > 0), (r - g) / safe + 4, h)
    h /= 6.0
    return np.stack([h, s, v], axis=-1).astype(np.float32)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV [0,1] → RGB [0,1]."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h6 = ((h % 1.0) * 6.0)
    hi = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = np.select([hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5], [v, q, p, p, t, v])
    g = np.select([hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5], [t, v, v, q, p, p])
    b = np.select([hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5], [p, p, t, v, v, q])
    return np.clip(np.stack([r, g, b], axis=-1), 0, 1).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luminance."""
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


# ---------------------------------------------------------------------------
# Planar variants
# ---------------------------------------------------------------------------
# These take and return contiguous ``(3, H, W)`` blocks. Indexing a plane is
# then a contiguous view rather than a stride-3 gather, which is what makes
# the HSL-based blend modes affordable. The maths is identical to the
# interleaved functions above; only the layout and the sector selection
# differ (boolean copyto instead of np.select, which evaluates and
# materialises all six branches).

def rgb_to_hsl_planar(rgb: np.ndarray) -> np.ndarray:
    """(3, H, W) RGB [0,1] -> (3, H, W) HSL [0,1]."""
    r, g, b = rgb[0], rgb[1], rgb[2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    l = (mx + mn) * 0.5
    d = mx - mn
    pos = d > 0
    s = np.where(pos,
                 np.where(l > 0.5, d / (2 - mx - mn + 1e-10),
                          d / (mx + mn + 1e-10)),
                 0)
    safe = np.where(pos, d, 1)
    h = np.zeros_like(l)
    h = np.where((mx == r) & pos, ((g - b) / safe) % 6, h)
    h = np.where((mx == g) & pos, (b - r) / safe + 2, h)
    h = np.where((mx == b) & pos, (r - g) / safe + 4, h)
    h /= 6.0
    out = np.empty_like(rgb)
    out[0], out[1], out[2] = h, s, l
    return out


def hsl_to_rgb_planar(hsl: np.ndarray) -> np.ndarray:
    """(3, H, W) HSL [0,1] -> (3, H, W) RGB [0,1]."""
    h, s, l = hsl[0], hsl[1], hsl[2]
    c = (1 - np.abs(2 * l - 1)) * s
    x = c * (1 - np.abs((h * 6) % 2 - 1))
    m = l - c * 0.5
    h6 = (h * 6).astype(np.int32) % 6

    out = np.zeros_like(hsl)
    r, g, b = out[0], out[1], out[2]
    # Sector -> (r, g, b) source among {c, x, 0}, matching the np.select
    # tables in hsl_to_rgb exactly.
    for sector, (rs, gs, bs) in enumerate((
        (c, x, None), (x, c, None), (None, c, x),
        (None, x, c), (x, None, c), (c, None, x),
    )):
        sel = h6 == sector
        if not sel.any():
            continue
        if rs is not None:
            np.copyto(r, rs, where=sel)
        if gs is not None:
            np.copyto(g, gs, where=sel)
        if bs is not None:
            np.copyto(b, bs, where=sel)
    out += m
    np.clip(out, 0, 1, out=out)
    return out
