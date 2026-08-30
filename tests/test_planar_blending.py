"""Planar blending must be numerically equivalent to the interleaved path.

The planar rewrite changes memory layout, not math. These tests pin that
down for every blend mode and every compositing feature, so a layout or
kernel bug cannot slip through as a subtle colour shift.
"""

from __future__ import annotations

import numpy as np
import pytest

from photo_editor.blending.blending_engine import BlendingEngine
from photo_editor.blending.planar import (
    PlanarScratch, blend_planar_region, to_interleaved, to_planar,
)
from photo_editor.core.enums import BlendMode

# One 8-bit level. Planar and interleaved perform the same float32
# operations in a different order, so tiny last-bit differences are
# acceptable; anything visible is not.
TOL = 1.0 / 255.0

H, W = 37, 53  # deliberately non-power-of-two, non-square


def _rand(h=H, w=W, seed=0, alpha=None):
    rng = np.random.default_rng(seed)
    img = rng.random((h, w, 4), dtype=np.float32)
    if alpha is not None:
        img[..., 3] = alpha
    return img


def _reference(base, over, pos, mode, opacity, mask):
    out = base.copy()
    BlendingEngine.blend_region_inplace(out, over, pos, mode, opacity, mask)
    return out


def _planar(base, over, pos, mode, opacity, mask):
    pb = to_planar(base)
    po = to_planar(over)
    blend_planar_region(pb, po, pos, mode, opacity, mask)
    return to_interleaved(pb)


def _assert_close(got, want, label):
    diff = np.abs(got - want)
    worst = float(diff.max())
    assert worst <= TOL, (
        f"{label}: max abs diff {worst:.6f} > {TOL:.6f} "
        f"at {np.unravel_index(int(diff.argmax()), diff.shape)}"
    )


@pytest.mark.parametrize("mode", list(BlendMode), ids=lambda m: m.name)
def test_planar_matches_interleaved_all_modes(mode):
    base = _rand(seed=1)
    over = _rand(seed=2)
    _assert_close(_planar(base, over, (0, 0), mode, 1.0, None),
                  _reference(base, over, (0, 0), mode, 1.0, None),
                  mode.name)


@pytest.mark.parametrize("opacity", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_planar_matches_with_opacity(opacity):
    base, over = _rand(seed=3), _rand(seed=4)
    for mode in (BlendMode.NORMAL, BlendMode.MULTIPLY, BlendMode.OVERLAY):
        _assert_close(_planar(base, over, (0, 0), mode, opacity, None),
                      _reference(base, over, (0, 0), mode, opacity, None),
                      f"{mode.name}@{opacity}")


def test_planar_matches_with_mask():
    base, over = _rand(seed=5), _rand(seed=6)
    mask = np.linspace(0, 1, H * W, dtype=np.float32).reshape(H, W)
    for mode in (BlendMode.NORMAL, BlendMode.SCREEN, BlendMode.COLOR):
        _assert_close(_planar(base, over, (0, 0), mode, 0.8, mask),
                      _reference(base, over, (0, 0), mode, 0.8, mask),
                      f"mask/{mode.name}")


@pytest.mark.parametrize("pos", [(0, 0), (5, 7), (-9, -4), (-100, 0),
                                 (W - 3, H - 3), (W + 50, H + 50), (0, -12)])
def test_planar_matches_at_offsets(pos):
    """Placement, clipping and fully-off-canvas cases."""
    base = _rand(seed=7)
    over = _rand(h=21, w=17, seed=8)
    _assert_close(_planar(base, over, pos, BlendMode.NORMAL, 1.0, None),
                  _reference(base, over, pos, BlendMode.NORMAL, 1.0, None),
                  f"pos{pos}")


def test_planar_matches_with_offset_and_mask():
    base = _rand(seed=9)
    over = _rand(h=21, w=17, seed=10)
    mask = np.linspace(0, 1, 21 * 17, dtype=np.float32).reshape(21, 17)
    for pos in ((3, 4), (-5, -6), (W - 5, H - 5)):
        _assert_close(_planar(base, over, pos, BlendMode.MULTIPLY, 0.6, mask),
                      _reference(base, over, pos, BlendMode.MULTIPLY, 0.6, mask),
                      f"mask+offset{pos}")


def test_planar_transparent_source_is_noop():
    base = _rand(seed=11)
    over = _rand(seed=12, alpha=0.0)
    _assert_close(_planar(base, over, (0, 0), BlendMode.NORMAL, 1.0, None),
                  base, "transparent source")


def test_planar_opaque_normal_replaces():
    base = _rand(seed=13)
    over = _rand(seed=14, alpha=1.0)
    got = _planar(base, over, (0, 0), BlendMode.NORMAL, 1.0, None)
    _assert_close(got, over, "opaque replace")


def test_blending_onto_empty_canvas():
    """Compositing onto a fully transparent canvas must preserve the source."""
    base = np.zeros((H, W, 4), dtype=np.float32)
    over = _rand(seed=15)
    got = _planar(base, over, (0, 0), BlendMode.NORMAL, 1.0, None)
    want = _reference(base, over, (0, 0), BlendMode.NORMAL, 1.0, None)
    _assert_close(got, want, "empty canvas")


def test_roundtrip_layout_conversion_is_lossless():
    img = _rand(seed=16)
    np.testing.assert_array_equal(to_interleaved(to_planar(img)), img)


def test_planar_buffers_are_contiguous():
    """The entire performance argument rests on contiguity."""
    p = to_planar(_rand(seed=17))
    assert p.flags["C_CONTIGUOUS"]
    assert p[:3].flags["C_CONTIGUOUS"], "colour block must be contiguous"
    assert p[3].flags["C_CONTIGUOUS"], "alpha plane must be contiguous"


def test_scratch_pool_reuses_buffers():
    pool = PlanarScratch(max_per_shape=2)
    a = pool.acquire((4, 8, 8))
    ptr = a.__array_interface__["data"][0]
    pool.release(a)
    b = pool.acquire((4, 8, 8))
    assert b.__array_interface__["data"][0] == ptr, "buffer was not reused"
    assert not b.any(), "reacquired buffer must be zeroed"


def test_scratch_pool_respects_cap():
    pool = PlanarScratch(max_per_shape=1)
    bufs = [pool.acquire((4, 4, 4)) for _ in range(3)]
    for b in bufs:
        pool.release(b)
    assert len(pool._pools[((4, 4, 4), np.dtype(np.float32))]) == 1


def test_dissolve_is_deterministic():
    """Dissolve must not re-randomise between frames (it used to)."""
    base, over = _rand(seed=18), _rand(seed=19)
    a = _planar(base, over, (0, 0), BlendMode.DISSOLVE, 1.0, None)
    b = _planar(base, over, (0, 0), BlendMode.DISSOLVE, 1.0, None)
    np.testing.assert_array_equal(a, b)
