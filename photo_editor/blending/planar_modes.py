"""Blend-mode kernels in planar layout.

Why planar
----------
The interleaved ``(H, W, 4)`` layout forces every colour operation through a
``[..., :3]`` slice whose innermost stride is 4 floats. NumPy cannot
vectorise across that stride, and the penalty is severe: a MULTIPLY over a
4K layer measured **71.4 ms** interleaved versus **3.5 ms** planar -- 20x.

In planar ``(4, H, W)`` layout, ``arr[:3]`` is a contiguous ``(3, H, W)``
block and ``arr[3]`` is a contiguous ``(H, W)`` plane, so every kernel below
is a straight contiguous ufunc.

Contract
--------
Each kernel takes contiguous ``(3, H, W)`` float32 colour blocks and returns
a new ``(3, H, W)`` block. Inputs are **straight-alpha** colour values in
[0, 1] -- identical semantics to the interleaved implementations in this
package, which these must reproduce exactly.
"""

from __future__ import annotations

import numpy as np

from ..core.enums import BlendMode

# Luminance weights used by the "darker/lighter colour" modes.
_LUM_R, _LUM_G, _LUM_B = 0.299, 0.587, 0.114


def _luma(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma of a (3, H, W) block -> (H, W)."""
    out = rgb[0] * _LUM_R
    out += rgb[1] * _LUM_G
    out += rgb[2] * _LUM_B
    return out


# ---------------------------------------------------------------------------
# Normal / dissolve
# ---------------------------------------------------------------------------

def p_normal(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return over


def p_dissolve(base: np.ndarray, over: np.ndarray,
               origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Dissolve, dithered against the global pattern at *origin*.

    *origin* is the absolute document position of this block. Without it the
    pattern would be generated per region, so a band-parallel render and a
    whole-canvas render of the same document disagreed along band
    boundaries.
    """
    from .normal import dither_for
    y0, x0 = origin
    noise = dither_for(y0, x0, over.shape[1], over.shape[2])
    avg = over.mean(axis=0)
    return np.where(noise < avg, over, base)


# ---------------------------------------------------------------------------
# Darken group
# ---------------------------------------------------------------------------

def p_darken(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return np.minimum(base, over)


def p_multiply(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return base * over


def p_color_burn(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    safe = np.where(over > 0, over, 1e-6)
    return np.clip(1.0 - (1.0 - base) / safe, 0, 1)


def p_linear_burn(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return np.clip(base + over - 1.0, 0, 1)


def p_darker_color(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return np.where(_luma(over) < _luma(base), over, base)


# ---------------------------------------------------------------------------
# Lighten group
# ---------------------------------------------------------------------------

def p_lighten(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return np.maximum(base, over)


def p_screen(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - base) * (1.0 - over)


def p_color_dodge(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    safe = np.where(over < 1.0, 1.0 - over, 1e-6)
    return np.clip(base / safe, 0, 1)


def p_linear_dodge(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return np.clip(base + over, 0, 1)


def p_lighter_color(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return np.where(_luma(over) > _luma(base), over, base)


# ---------------------------------------------------------------------------
# Contrast group
# ---------------------------------------------------------------------------

def p_overlay(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    lo = 2 * base * over
    hi = 1 - 2 * (1 - base) * (1 - over)
    return np.where(base <= 0.5, lo, hi)


def p_soft_light(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    lo = base - (1 - 2 * over) * base * (1 - base)
    hi = base + (2 * over - 1) * (np.sqrt(np.clip(base, 0, None)) - base)
    return np.where(over <= 0.5, lo, hi)


def p_hard_light(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    lo = 2 * base * over
    hi = 1 - 2 * (1 - base) * (1 - over)
    return np.where(over <= 0.5, lo, hi)


def p_vivid_light(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    burn_d = np.where(over > 0, 2 * over, 1e-6)
    dodge_d = np.where(over < 1, 2 * (1 - over), 1e-6)
    burn = 1.0 - (1.0 - base) / burn_d
    dodge = base / dodge_d
    return np.clip(np.where(over <= 0.5, burn, dodge), 0, 1)


def p_linear_light(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return np.clip(base + 2 * over - 1.0, 0, 1)


def p_pin_light(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    lo = np.minimum(base, 2 * over)
    hi = np.maximum(base, 2 * over - 1)
    return np.where(over < 0.5, lo, hi)


def p_hard_mix(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return np.where(base + over >= 1.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# Comparative group
# ---------------------------------------------------------------------------

def p_difference(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return np.abs(base - over)


def p_exclusion(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return base + over - 2 * base * over


def p_subtract(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return np.clip(base - over, 0, 1)


def p_divide(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    safe = np.where(over > 0, over, 1e-6)
    return np.clip(base / safe, 0, 1)


# ---------------------------------------------------------------------------
# Colour group -- HSL round-trips, done entirely in planar layout so every
# channel access is a contiguous view rather than a stride-3 gather.
# `take` picks (hue, saturation, luminance) from the base and overlay HSL
# blocks; the four modes differ only in that choice.
# ---------------------------------------------------------------------------

def _hsl_mix(base: np.ndarray, over: np.ndarray, take) -> np.ndarray:
    from ..utils.color_utils import rgb_to_hsl_planar, hsl_to_rgb_planar
    b_hsl = rgb_to_hsl_planar(base)
    o_hsl = rgb_to_hsl_planar(over)
    mixed = np.empty_like(b_hsl)
    for i, plane in enumerate(take(b_hsl, o_hsl)):
        mixed[i] = plane
    return hsl_to_rgb_planar(mixed)


def p_hue(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return _hsl_mix(base, over, lambda b, o: (o[0], b[1], b[2]))


def p_saturation(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return _hsl_mix(base, over, lambda b, o: (b[0], o[1], b[2]))


def p_color(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return _hsl_mix(base, over, lambda b, o: (o[0], o[1], b[2]))


def p_luminosity(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    return _hsl_mix(base, over, lambda b, o: (b[0], b[1], o[2]))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PLANAR_BLEND_FUNCS = {
    BlendMode.NORMAL: p_normal,
    BlendMode.DISSOLVE: p_dissolve,
    BlendMode.DARKEN: p_darken,
    BlendMode.MULTIPLY: p_multiply,
    BlendMode.COLOR_BURN: p_color_burn,
    BlendMode.LINEAR_BURN: p_linear_burn,
    BlendMode.DARKER_COLOR: p_darker_color,
    BlendMode.LIGHTEN: p_lighten,
    BlendMode.SCREEN: p_screen,
    BlendMode.COLOR_DODGE: p_color_dodge,
    BlendMode.LINEAR_DODGE: p_linear_dodge,
    BlendMode.LIGHTER_COLOR: p_lighter_color,
    BlendMode.OVERLAY: p_overlay,
    BlendMode.SOFT_LIGHT: p_soft_light,
    BlendMode.HARD_LIGHT: p_hard_light,
    BlendMode.VIVID_LIGHT: p_vivid_light,
    BlendMode.LINEAR_LIGHT: p_linear_light,
    BlendMode.PIN_LIGHT: p_pin_light,
    BlendMode.HARD_MIX: p_hard_mix,
    BlendMode.DIFFERENCE: p_difference,
    BlendMode.EXCLUSION: p_exclusion,
    BlendMode.SUBTRACT: p_subtract,
    BlendMode.DIVIDE: p_divide,
    BlendMode.HUE: p_hue,
    BlendMode.SATURATION: p_saturation,
    BlendMode.COLOR: p_color,
    BlendMode.LUMINOSITY: p_luminosity,
}


# Modes whose result depends on WHERE the block sits in the document, and
# which therefore must be handed an absolute origin.
POSITION_DEPENDENT = frozenset({BlendMode.DISSOLVE})


def get_planar_blend_func(mode: BlendMode):
    return PLANAR_BLEND_FUNCS.get(mode, p_normal)
