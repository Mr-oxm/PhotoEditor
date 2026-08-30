"""Sandwich caching must be invisible in the output.

Reusing the composited layers below the one being edited is only safe if
the result is identical to compositing everything. These tests run both
paths over every reference scene and at every possible focus layer, because
a wrong answer here is a silently mis-composited image rather than a slow
one.
"""

from __future__ import annotations

import numpy as np
import pytest

from photo_editor.blending.planar import to_interleaved
from photo_editor.core.enums import BlendMode, LayerType
from photo_editor.engine.render_pipeline import RenderPipeline
from photo_editor.engine.sandwich_cache import (
    SandwichCache, layer_signature, over_run_is_isolatable,
)
from tests.fidelity.scenes import all_scenes

TOL = 1.0 / 255.0
SCENES = sorted(all_scenes().items())


def _full(doc):
    pipe = RenderPipeline()
    pipe.end_interaction()
    return to_interleaved(pipe.execute_planar(doc, level=0)).copy()


def _interactive(doc, focus_id, frames=3):
    """Render *frames* interactive frames and return the last."""
    pipe = RenderPipeline()
    pipe.begin_interaction(focus_id)
    out = None
    for _ in range(frames):
        pipe.invalidate(focus_id)
        out = to_interleaved(pipe.execute_planar(doc, level=0)).copy()
    return out


@pytest.mark.parametrize("name,factory", SCENES, ids=[n for n, _ in SCENES])
def test_interactive_matches_full_for_every_focus(name, factory):
    doc = factory()
    expected = _full(factory())
    for layer in list(doc.layers):
        got = _interactive(factory(), layer.id)
        diff = np.abs(got - expected)
        assert float(diff.max()) <= TOL, (
            f"{name}: focusing layer '{layer.name}' changed the composite "
            f"(max diff {float(diff.max()):.6f})")


def test_repeated_interactive_frames_are_stable():
    """The cache must not drift as frames accumulate."""
    factory = all_scenes()["many_layers"]
    doc = factory()
    focus = list(doc.layers)[len(list(doc.layers)) // 2].id
    pipe = RenderPipeline()
    pipe.begin_interaction(focus)
    first = None
    for i in range(6):
        pipe.invalidate(focus)
        frame = to_interleaved(pipe.execute_planar(doc, level=0)).copy()
        if first is None:
            first = frame
        else:
            np.testing.assert_allclose(frame, first, atol=1e-6)


def test_editing_the_focus_layer_is_reflected():
    factory = all_scenes()["many_layers"]
    doc = factory()
    layers = list(doc.layers)
    focus = layers[-1]
    pipe = RenderPipeline()
    pipe.begin_interaction(focus.id)
    pipe.invalidate(focus.id)
    before = to_interleaved(pipe.execute_planar(doc, level=0)).copy()

    focus.position = (focus.position[0] + 20, focus.position[1] + 15)
    pipe.invalidate(focus.id)
    after = to_interleaved(pipe.execute_planar(doc, level=0))
    assert not np.allclose(before, after), "moving the focus layer did nothing"


def test_editing_a_layer_below_invalidates_the_cache():
    """The cache keys off the layers it covers, so a change to any of them
    must be picked up without anyone remembering to say so."""
    factory = all_scenes()["many_layers"]
    doc = factory()
    layers = list(doc.layers)
    below, focus = layers[1], layers[-1]

    pipe = RenderPipeline()
    pipe.begin_interaction(focus.id)
    pipe.invalidate(focus.id)
    before = to_interleaved(pipe.execute_planar(doc, level=0)).copy()

    below.visible = False
    pipe.invalidate(focus.id)
    after = to_interleaved(pipe.execute_planar(doc, level=0))
    assert not np.allclose(before, after), (
        "hiding a layer below the focus did not invalidate the sandwich")


def test_changing_focus_rebuilds():
    factory = all_scenes()["many_layers"]
    doc = factory()
    layers = list(doc.layers)
    pipe = RenderPipeline()
    expected = _full(factory())
    for layer in (layers[2], layers[-1], layers[4]):
        pipe.begin_interaction(layer.id)
        pipe.invalidate(layer.id)
        got = to_interleaved(pipe.execute_planar(doc, level=0))
        np.testing.assert_allclose(got, expected, atol=TOL)


def test_end_interaction_clears_the_cache():
    factory = all_scenes()["many_layers"]
    doc = factory()
    focus = list(doc.layers)[-1].id
    pipe = RenderPipeline()
    pipe.begin_interaction(focus)
    pipe.invalidate(focus)
    pipe.execute_planar(doc, level=0)
    assert pipe.sandwich.nbytes() > 0
    pipe.end_interaction()
    assert pipe.sandwich.nbytes() == 0


def test_interactive_works_at_preview_levels_and_rois():
    factory = all_scenes()["many_layers"]
    doc = factory()
    focus = list(doc.layers)[-1].id
    for level in (0, 1):
        for roi in (None, (10, 10, 80, 60)):
            reference = RenderPipeline()
            expected = to_interleaved(
                reference.execute_planar(factory(), level=level, roi=roi))
            pipe = RenderPipeline()
            pipe.begin_interaction(focus)
            pipe.invalidate(focus)
            got = to_interleaved(
                pipe.execute_planar(doc, level=level, roi=roi))
            np.testing.assert_allclose(
                got, expected, atol=TOL,
                err_msg=f"level={level} roi={roi}")


# ---------------------------------------------------------------------------
# Cache unit behaviour
# ---------------------------------------------------------------------------

def test_signature_changes_with_content():
    from photo_editor.core.layer import Layer
    layer = Layer(name="a", width=8, height=8)
    before = layer_signature(layer)
    layer.touch()
    assert layer_signature(layer) != before


@pytest.mark.parametrize("mutate", [
    lambda l: setattr(l, "position", (5, 5)),
    lambda l: setattr(l, "opacity", 0.5),
    lambda l: setattr(l, "blend_mode", BlendMode.MULTIPLY),
    lambda l: setattr(l, "visible", False),
    lambda l: setattr(l, "clipping_mask", True),
])
def test_signature_covers_every_contributing_attribute(mutate):
    from photo_editor.core.layer import Layer
    layer = Layer(name="a", width=8, height=8)
    before = layer_signature(layer)
    mutate(layer)
    assert layer_signature(layer) != before


def test_isolatable_run_detection():
    from photo_editor.core.layer import Layer
    plain = [Layer(name=f"L{i}", width=4, height=4) for i in range(3)]
    assert over_run_is_isolatable(plain, {})

    plain[1].blend_mode = BlendMode.MULTIPLY
    assert not over_run_is_isolatable(plain, {}), "non-NORMAL must disqualify"

    plain[1].blend_mode = BlendMode.NORMAL
    plain[2].clipping_mask = True
    assert not over_run_is_isolatable(plain, {}), "clipping must disqualify"

    plain[2].clipping_mask = False
    adj = Layer(name="adj", width=4, height=4, layer_type=LayerType.ADJUSTMENT)
    assert not over_run_is_isolatable(plain + [adj], {}), (
        "an adjustment layer consumes the canvas and must disqualify")


def test_cache_get_and_put():
    cache = SandwichCache()
    buf = np.ones((4, 8, 8), dtype=np.float32)
    assert cache.get_under(("k",)) is None
    cache.put_under(("k",), buf)
    got = cache.get_under(("k",))
    assert got is not None and np.array_equal(got, buf)
    assert cache.get_under(("other",)) is None
    cache.clear()
    assert cache.get_under(("k",)) is None


def test_cache_stores_a_copy():
    """The caller's buffer is reused between frames; the cache must not
    alias it or the cached 'under' would follow the live canvas."""
    cache = SandwichCache()
    buf = np.ones((4, 4, 4), dtype=np.float32)
    cache.put_under(("k",), buf)
    buf[:] = 9.0
    assert np.array_equal(cache.get_under(("k",)),
                          np.ones((4, 4, 4), dtype=np.float32))
