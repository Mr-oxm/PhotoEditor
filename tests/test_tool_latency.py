"""Interactive tools: what happens per mouse-move event.

Mouse-move events are not coalesced before they reach the tools, so
anything allocated or scanned per event is paid 60-120 times a second.
These tests pin the two contracts that keep that affordable: a stroke
caches its selection mask, and a transform drag uses the fast resampling
path.
"""

from __future__ import annotations

import numpy as np
import pytest

from photo_editor.core.document import Document
from photo_editor.tools.brush import BrushTool
from photo_editor.tools.eraser import EraserTool
from photo_editor.tools.transform_tool import TransformTool


def _doc(w=256, h=192, selection=False):
    doc = Document(w, h, name="tool-test")
    doc.layers.active_index = 0
    if selection:
        doc.selection.select_rect(20, 20, 100, 80)
    return doc


@pytest.mark.parametrize("tool_cls", [BrushTool, EraserTool])
def test_stroke_caches_the_selection_mask(tool_cls):
    """Rebuilding it per event allocated a whole layer-sized array each time."""
    doc = _doc(selection=True)
    tool = tool_cls()
    calls = []
    original = type(tool)._compute_sel_mask
    type(tool)._compute_sel_mask = staticmethod(
        lambda d: (calls.append(1), original(d))[1])
    try:
        tool.on_press(doc, 30, 30, 1.0)
        for i in range(10):
            tool.on_move(doc, 30 + i * 3, 30 + i * 2, 1.0)
        tool.on_release(doc, 60, 50)
    finally:
        type(tool)._compute_sel_mask = staticmethod(original)
    assert len(calls) == 1, (
        f"selection mask rebuilt {len(calls)} times for one stroke")


@pytest.mark.parametrize("tool_cls", [BrushTool, EraserTool])
def test_stroke_still_honours_the_selection(tool_cls):
    """Caching must not change what the stroke does."""
    doc = _doc(selection=True)
    layer = doc.layers.active_layer
    layer.begin_write()
    layer.pixels[:] = np.array([1, 1, 1, 1], dtype=np.float32)
    before_outside = layer.pixels[150, 200].copy()

    tool = tool_cls()
    tool.size = 40
    tool.on_press(doc, 60, 60, 1.0)
    tool.on_move(doc, 200, 150, 1.0)     # drags outside the selection
    tool.on_release(doc, 200, 150)

    # Pixels outside the selection must be untouched.
    assert np.allclose(layer.pixels[150, 200], before_outside), (
        "stroke painted outside the active selection")


@pytest.mark.parametrize("tool_cls", [BrushTool, EraserTool])
def test_mask_cache_is_dropped_after_the_stroke(tool_cls):
    doc = _doc(selection=True)
    tool = tool_cls()
    tool.on_press(doc, 30, 30, 1.0)
    assert tool._stroke_sel_valid
    tool.on_release(doc, 30, 30)
    assert not tool._stroke_sel_valid
    assert tool._stroke_sel_mask is None


def test_a_new_selection_is_picked_up_by_the_next_stroke():
    doc = _doc(selection=True)
    tool = BrushTool()
    tool.on_press(doc, 30, 30, 1.0)
    first = tool._stroke_sel_mask
    tool.on_release(doc, 30, 30)

    doc.selection.select_rect(0, 0, 10, 10)
    tool.on_press(doc, 5, 5, 1.0)
    second = tool._stroke_sel_mask
    tool.on_release(doc, 5, 5)
    assert not np.array_equal(first, second), (
        "the stroke reused a mask from the previous selection")


def test_transform_drag_uses_the_fast_resampling_path():
    """The quality path measured 293 ms per event on a 4K layer."""
    doc = _doc(512, 384)
    layer = doc.layers.active_layer
    layer.init_non_destructive()
    seen = []
    original = type(layer).compute_display

    def spy(self, *a, **kw):
        seen.append(kw.get("fast", False))
        return original(self, *a, **kw)

    type(layer).compute_display = spy
    try:
        tool = TransformTool()
        tool.mode = "scale"
        tool.on_press(doc, 100, 100, 1.0)
        tool.on_move(doc, 160, 140, 1.0)
        tool.on_move(doc, 200, 180, 1.0)
        drag_calls = list(seen)
        tool.on_release(doc, 200, 180)
        release_calls = seen[len(drag_calls):]
    finally:
        type(layer).compute_display = original

    assert drag_calls and all(drag_calls), (
        f"transform drag used the quality path: {drag_calls}")
    assert release_calls and release_calls[-1] is False, (
        "transform must re-derive at full quality on release")


def test_transform_release_leaves_a_full_quality_layer():
    doc = _doc(128, 96)
    layer = doc.layers.active_layer
    layer.begin_write()
    layer.pixels[:] = 0.5
    layer.init_non_destructive()
    tool = TransformTool()
    tool.mode = "scale"
    tool.on_press(doc, 40, 40, 1.0)
    tool.on_move(doc, 80, 70, 1.0)
    tool.on_release(doc, 80, 70)
    assert layer.pixels.dtype == np.float32
    assert layer.pixels.size > 0
