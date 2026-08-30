"""Normal and Dissolve blend modes."""

from __future__ import annotations

import numpy as np


def blend_normal(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    return overlay


# Dissolve uses a *stable* dither field.  Photoshop's dissolve pattern is
# fixed for a given document — it does not re-randomise on every redraw.
# An unseeded np.random.random() here made the composite non-deterministic,
# so a dissolve layer visibly flickered on every rendered frame and the
# render output could never be cached or regression-tested.
#
# The field is generated once per size from a fixed seed and cached.
_DITHER_CACHE: dict[tuple[int, int], np.ndarray] = {}
_DITHER_SEED = 0x5EED_D155
_DITHER_MAX_CACHED = 8


def _dither_field(height: int, width: int) -> np.ndarray:
    """Deterministic blue-ish noise field in [0, 1) for (height, width)."""
    key = (height, width)
    cached = _DITHER_CACHE.get(key)
    if cached is not None:
        return cached
    field = np.random.default_rng(_DITHER_SEED).random(
        (height, width), dtype=np.float32,
    )
    if len(_DITHER_CACHE) >= _DITHER_MAX_CACHED:
        _DITHER_CACHE.pop(next(iter(_DITHER_CACHE)))
    _DITHER_CACHE[key] = field
    return field


def blend_dissolve(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    h, w = overlay.shape[:2]
    noise = _dither_field(h, w)
    avg = overlay.mean(axis=-1)
    mask = (noise < avg)[..., np.newaxis]
    return np.where(mask, overlay, base)
