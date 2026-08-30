"""Normal and Dissolve blend modes."""

from __future__ import annotations

import numpy as np

# Dissolve's dither must be a property of the *document*, not of whatever
# region happens to be rendered. Two things broke that:
#
#  * An unseeded np.random.random() made it different on every frame, so a
#    dissolve layer visibly flickered and the composite could never be
#    cached or regression-tested.
#  * Generating a field sized to the region being blended made it different
#    per band once band-parallel rendering arrived, and different again for
#    every viewport ROI -- seams along band boundaries that moved as you
#    panned.
#
# So the field is a fixed tile, generated once from a fixed seed, and
# indexed by ABSOLUTE document coordinates with wraparound. Any region --
# a band, a ROI, the whole canvas -- slices the same global pattern.
_DITHER_TILE = 512
_DITHER_SEED = 0x5EED_D155
_dither_tile_cache: np.ndarray | None = None


def _dither_tile() -> np.ndarray:
    global _dither_tile_cache
    if _dither_tile_cache is None:
        _dither_tile_cache = np.random.default_rng(_DITHER_SEED).random(
            (_DITHER_TILE, _DITHER_TILE), dtype=np.float32,
        )
    return _dither_tile_cache


def dither_for(y0: int, x0: int, height: int, width: int) -> np.ndarray:
    """The global dither pattern over the region at absolute (y0, x0).

    Wraps the tile, so the pattern is continuous across region boundaries
    and identical however the frame was decomposed.
    """
    tile = _dither_tile()
    ys = (np.arange(y0, y0 + height) % _DITHER_TILE)
    xs = (np.arange(x0, x0 + width) % _DITHER_TILE)
    return tile[np.ix_(ys, xs)]


def blend_normal(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    return overlay


def blend_dissolve(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    h, w = overlay.shape[:2]
    noise = dither_for(0, 0, h, w)
    avg = overlay.mean(axis=-1)
    mask = (noise < avg)[..., np.newaxis]
    return np.where(mask, overlay, base)
