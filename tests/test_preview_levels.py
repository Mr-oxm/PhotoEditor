"""Resolution-adaptive preview rendering.

Compositing a 4K document for a 1600x1000 viewport at full resolution does
8x the necessary pixel work and needs 126 MB of cache per layer. Rendering
at a mip level sized to the viewport is the single largest win available,
but only if the scaled composite is *the same picture*, just smaller.

These tests check that: correct output size, close agreement with a
downscaled full-resolution render, and no change at all to level 0 (the
export path).
"""

from __future__ import annotations

import numpy as np
import pytest

from photo_editor.blending.planar import to_interleaved
from photo_editor.engine.layer_cache import LayerRasterCache
from photo_editor.engine.render_pipeline import RenderPipeline, level_size
from tests.fidelity.scenes import all_scenes


def _downscale(img, level):
    import cv2
    h, w = img.shape[:2]
    nw, nh = level_size(w, h, level)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Level selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("w,h,max_size,expected", [
    (3840, 2160, 0, 0),        # disabled -> full resolution
    (3840, 2160, 4096, 0),     # already fits
    (3840, 2160, 2048, 1),     # 1920x1080
    (3840, 2160, 1024, 2),     # 960x540
    (3840, 2160, 512, 3),
    (1920, 1080, 2048, 0),
    (800, 600, 2048, 0),
])
def test_choose_level(w, h, max_size, expected):
    assert LayerRasterCache.choose_level(w, h, max_size) == expected


@pytest.mark.parametrize("level,expected", [
    (0, (3840, 2160)), (1, (1920, 1080)), (2, (960, 540)), (3, (480, 270)),
])
def test_level_size(level, expected):
    assert level_size(3840, 2160, level) == expected


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", [0, 1, 2])
def test_preview_output_size(level):
    doc = all_scenes()["many_layers"]()
    pipe = RenderPipeline()
    out = pipe.execute_planar(doc, level=level)
    ew, eh = level_size(doc.width, doc.height, level)
    assert out.shape == (4, eh, ew)


def test_level_zero_is_unchanged():
    """Export must be bit-identical to before the preview work."""
    from photo_editor.engine.planar_compositor import PlanarCompositor
    for name in ("many_layers", "group", "clipping_chain", "styles"):
        doc = all_scenes()[name]()
        a = PlanarCompositor().composite(doc.layers, doc.width, doc.height)
        b = PlanarCompositor().composite(doc.layers, doc.width, doc.height,
                                         level=0)
        np.testing.assert_array_equal(a, b, err_msg=name)


# ---------------------------------------------------------------------------
# Visual agreement
# ---------------------------------------------------------------------------

SCENES = ["many_layers", "opacity", "group", "nested_group", "clipping_chain",
          "legacy_mask", "mask_layer", "channels", "offcanvas",
          "hidden_empty", "adjustment_root", "blend_MULTIPLY",
          "blend_SCREEN", "blend_OVERLAY"]


@pytest.mark.parametrize("name", SCENES)
def test_scaled_preview_resembles_downscaled_full_render(name):
    """A level-1 composite must look like the full render, shrunk.

    Compositing at reduced resolution is not mathematically identical to
    compositing then shrinking -- blending is non-linear, and a half-scale
    source pixel is an average of four. The tolerance below is what that
    genuinely costs; anything larger means geometry or masks are being
    scaled wrongly.
    """
    doc = all_scenes()[name]()
    pipe = RenderPipeline()
    full = to_interleaved(pipe.execute_planar(doc, level=0))
    pipe.invalidate()
    scaled = to_interleaved(pipe.execute_planar(doc, level=1))

    reference = _downscale(full, 1)
    assert scaled.shape == reference.shape

    diff = np.abs(scaled - reference)
    assert float(diff.mean()) < 0.02, (
        f"{name}: mean abs diff {diff.mean():.4f} -- scaled preview does "
        f"not match the downscaled full render"
    )


@pytest.mark.parametrize("name", ["many_layers", "group", "offcanvas"])
def test_preview_geometry_is_positioned_correctly(name):
    """Catch position-scaling errors: content must land in the same place.

    Compares the centre of mass of the alpha channel, which shifts
    immediately if layer positions are scaled inconsistently.
    """
    doc = all_scenes()[name]()
    pipe = RenderPipeline()
    full = to_interleaved(pipe.execute_planar(doc, level=0))
    pipe.invalidate()
    scaled = to_interleaved(pipe.execute_planar(doc, level=1))

    def centroid(img):
        a = img[..., 3]
        total = a.sum()
        if total <= 0:
            return (0.5, 0.5)
        ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        return (float((ys * a).sum() / total / a.shape[0]),
                float((xs * a).sum() / total / a.shape[1]))

    fy, fx = centroid(full)
    sy, sx = centroid(scaled)
    assert abs(fy - sy) < 0.02 and abs(fx - sx) < 0.02, (
        f"{name}: alpha centroid moved from ({fx:.3f}, {fy:.3f}) to "
        f"({sx:.3f}, {sy:.3f}) -- layer positions are mis-scaled"
    )


def test_switching_levels_invalidates_the_cached_result():
    doc = all_scenes()["many_layers"]()
    pipe = RenderPipeline()
    a = pipe.execute_planar(doc, level=0)
    b = pipe.execute_planar(doc, level=1)
    assert a.shape != b.shape, "level change returned the stale cached buffer"
    c = pipe.execute_planar(doc, level=0)
    assert c.shape == a.shape


def test_uint8_path_honours_level():
    doc = all_scenes()["many_layers"]()
    pipe = RenderPipeline()
    out = pipe.execute_to_uint8(doc, level=1)
    ew, eh = level_size(doc.width, doc.height, 1)
    assert out.shape == (eh, ew, 4)
    assert out.dtype == np.uint8
