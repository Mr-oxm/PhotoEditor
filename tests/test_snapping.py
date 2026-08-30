"""Smart snapping geometry.

Pure geometry, so it can be tested exhaustively: the interesting cases are
which candidate wins, how the threshold scales with zoom, and that nothing
snaps when it should not.
"""

from __future__ import annotations

import pytest

from photo_editor.core.document import Document
from photo_editor.core.enums import LayerType
from photo_editor.core.layer import Layer
from photo_editor.core.snapping import (
    SnapCandidate, SnapEngine, SnapSource, collect_candidates, snap_rect,
)


def _v(*positions, source=SnapSource.CANVAS):
    return [SnapCandidate(p, source) for p in positions]


# ---------------------------------------------------------------------------
# Basic snapping
# ---------------------------------------------------------------------------

def test_left_edge_snaps():
    result = snap_rect((97.0, 50.0, 40.0, 30.0), _v(100.0), [], threshold=8)
    assert result.dx == pytest.approx(3.0)
    assert result.snapped


def test_right_edge_snaps():
    # right edge at 140 -> target 143
    result = snap_rect((100.0, 50.0, 40.0, 30.0), _v(143.0), [], threshold=8)
    assert result.dx == pytest.approx(3.0)


def test_centre_snaps():
    # centre x at 120 -> target 124
    result = snap_rect((100.0, 50.0, 40.0, 30.0), _v(124.0), [], threshold=8)
    assert result.dx == pytest.approx(4.0)


def test_nothing_snaps_beyond_the_threshold():
    result = snap_rect((100.0, 50.0, 40.0, 30.0), _v(200.0), [], threshold=8)
    assert result.dx == 0.0
    assert not result.snapped


def test_axes_snap_independently():
    result = snap_rect((97.0, 48.0, 40.0, 30.0),
                       _v(100.0), _v(50.0), threshold=8)
    assert result.dx == pytest.approx(3.0)
    assert result.dy == pytest.approx(2.0)
    assert len(result.lines) == 2


def test_nearest_candidate_wins():
    # Rect edges are x = 100 (left), 105 (centre), 110 (right).
    # Both candidates are near the left edge; the closer one must win.
    result = snap_rect((100.0, 0.0, 10.0, 10.0),
                       _v(96.0, 102.0), [], threshold=8)
    assert result.dx == pytest.approx(2.0)


def test_centre_can_win_over_an_edge():
    """All three of left, centre and right are candidates for snapping."""
    result = snap_rect((100.0, 0.0, 10.0, 10.0),
                       _v(105.0, 96.0), [], threshold=8)
    assert result.dx == pytest.approx(0.0)
    assert result.snapped, "an exact centre match still counts as a snap"


def test_canvas_beats_layer_when_equidistant():
    """Equal distances must resolve deterministically, or the snap flips
    back and forth between coincident targets during a drag."""
    candidates = [
        SnapCandidate(103.0, SnapSource.LAYER),
        SnapCandidate(103.0, SnapSource.CANVAS),
    ]
    result = snap_rect((100.0, 0.0, 10.0, 10.0), candidates, [], threshold=8)
    assert result.lines[0].source is SnapSource.CANVAS


def test_zero_threshold_disables_snapping():
    result = snap_rect((100.0, 0.0, 10.0, 10.0), _v(100.0), [], threshold=0)
    assert not result.snapped


# ---------------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------------

def _doc_with_layers():
    doc = Document(400, 300, name="snap")
    doc.layers.layers.clear()
    a = Layer(name="a", width=100, height=80)
    a.position = (50, 40)
    doc.layers.add(a)
    b = Layer(name="b", width=60, height=60)
    b.position = (250, 150)
    doc.layers.add(b)
    return doc, a, b


def test_canvas_edges_and_centre_are_candidates():
    doc, _, _ = _doc_with_layers()
    vertical, horizontal = collect_candidates(doc, moving_ids=set())
    vpos = {c.position for c in vertical if c.source is SnapSource.CANVAS}
    hpos = {c.position for c in horizontal if c.source is SnapSource.CANVAS}
    assert vpos == {0.0, 200.0, 400.0}
    assert hpos == {0.0, 150.0, 300.0}


def test_other_layers_contribute_candidates():
    doc, a, b = _doc_with_layers()
    vertical, _ = collect_candidates(doc, moving_ids={b.id})
    layer_x = {c.position for c in vertical if c.source is SnapSource.LAYER}
    assert layer_x == {50.0, 100.0, 150.0}      # left, centre, right of a


def test_a_layer_does_not_snap_to_itself():
    doc, a, b = _doc_with_layers()
    vertical, _ = collect_candidates(doc, moving_ids={a.id, b.id})
    assert not [c for c in vertical if c.source is SnapSource.LAYER]


def test_hidden_layers_are_ignored():
    doc, a, b = _doc_with_layers()
    a.visible = False
    vertical, _ = collect_candidates(doc, moving_ids={b.id})
    assert not [c for c in vertical if c.source is SnapSource.LAYER]


def test_adjustment_and_mask_layers_are_ignored():
    doc, a, b = _doc_with_layers()
    for lt in (LayerType.ADJUSTMENT, LayerType.FILTER, LayerType.MASK):
        extra = Layer(name=str(lt), width=10, height=10, layer_type=lt)
        extra.position = (300, 200)
        doc.layers.add(extra)
    vertical, _ = collect_candidates(doc, moving_ids={a.id, b.id})
    assert not [c for c in vertical if c.source is SnapSource.LAYER]


def test_guides_contribute_candidates():
    from PySide6.QtCore import Qt

    class G:
        def __init__(self, orientation, position):
            self.orientation = orientation
            self.position = position

    doc, a, b = _doc_with_layers()
    guides = [G(Qt.Orientation.Vertical, 123.0),
              G(Qt.Orientation.Horizontal, 77.0)]
    vertical, horizontal = collect_candidates(doc, moving_ids=set(),
                                              guides=guides)
    assert 123.0 in {c.position for c in vertical
                     if c.source is SnapSource.GUIDE}
    assert 77.0 in {c.position for c in horizontal
                    if c.source is SnapSource.GUIDE}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def test_threshold_is_in_screen_pixels():
    """A fixed document-space threshold would grab everything when zoomed
    out and nothing when zoomed in."""
    doc, a, b = _doc_with_layers()
    engine = SnapEngine(threshold_px=8.0)
    engine.begin(doc, moving_ids={b.id})

    rect = (44.0, 200.0, 60.0, 60.0)     # left edge 6 doc-px from layer a
    assert engine.snap(rect, zoom=1.0).snapped          # 8 px reach
    assert not engine.snap(rect, zoom=4.0).snapped      # 2 doc-px reach
    assert engine.snap(rect, zoom=0.25).snapped         # 32 doc-px reach


def test_engine_is_inert_until_begin():
    doc, a, b = _doc_with_layers()
    engine = SnapEngine()
    assert not engine.snap((50.0, 40.0, 10.0, 10.0)).snapped
    engine.begin(doc, {b.id})
    assert engine.active
    engine.end()
    assert not engine.active


def test_disabled_engine_never_snaps():
    doc, a, b = _doc_with_layers()
    engine = SnapEngine(enabled=False)
    engine.begin(doc, {b.id})
    assert not engine.snap((49.0, 40.0, 10.0, 10.0)).snapped


def test_snap_lines_span_both_objects():
    """The drawn line should reach across the two things it relates."""
    doc, a, b = _doc_with_layers()
    engine = SnapEngine(threshold_px=8.0)
    engine.begin(doc, {b.id})
    result = engine.snap((48.0, 200.0, 60.0, 60.0), zoom=1.0)
    line = next(l for l in result.lines if l.vertical)
    assert line.start <= 40.0     # top of layer a
    assert line.end >= 260.0      # bottom of the dragged rect


def test_qt_orientation_flag_enum_is_handled():
    """Qt.Orientation is a flag enum: int() raises and == 1 is False, so
    only .value is reliable. Getting this wrong silently sent every
    horizontal guide into the vertical candidate list."""
    from PySide6.QtCore import Qt
    from photo_editor.core.snapping import _guide_is_horizontal
    assert _guide_is_horizontal(Qt.Orientation.Horizontal)
    assert not _guide_is_horizontal(Qt.Orientation.Vertical)
    assert _guide_is_horizontal(1)
    assert not _guide_is_horizontal(2)
    assert _guide_is_horizontal("horizontal")
    assert not _guide_is_horizontal(None)


# ---------------------------------------------------------------------------
# Move tool integration
# ---------------------------------------------------------------------------

def test_move_tool_snapping_is_off_until_the_ui_enables_it():
    """A programmatic drag must move exactly as far as it was told."""
    from photo_editor.tools.move.move_tool import MoveTool
    tool = MoveTool()
    assert not tool.snap_engine.enabled


def test_move_tool_snaps_a_drag_when_enabled():
    from photo_editor.tools.move.move_tool import MoveTool

    doc, a, b = _doc_with_layers()
    doc.layers.active_index = doc.layers.layers.index(b)
    tool = MoveTool()
    tool.auto_select = False
    tool.snap_engine.enabled = True
    tool.snap_zoom = 1.0

    # Press in the middle of b, well away from its resize handles.
    start = (280, 180)
    tool.on_press(doc, *start, 1.0)
    assert tool._mode.name == "MOVE", f"pressed a handle, not the body: {tool._mode}"
    # Drag so b's left edge lands at 153 -- 3 px from a's right edge (150).
    tool.on_move(doc, start[0] - 97, start[1], 1.0)
    snapped_x = b.position[0]
    tool.on_release(doc, start[0] - 97, start[1])
    assert snapped_x == 150, (
        f"expected the left edge to snap to 150, got {snapped_x}")


def test_move_tool_reports_lines_only_while_snapped():
    from photo_editor.tools.move.move_tool import MoveTool

    doc, a, b = _doc_with_layers()
    doc.layers.active_index = doc.layers.layers.index(b)
    tool = MoveTool()
    tool.auto_select = False
    tool.snap_engine.enabled = True
    tool.snap_zoom = 1.0

    tool.on_press(doc, 280, 180, 1.0)
    tool.on_move(doc, 183, 180, 1.0)          # near layer a's right edge
    assert tool.snap_lines, "no alignment line reported for a snapped drag"
    tool.on_move(doc, 280, 180, 1.0)          # back to where nothing aligns
    tool.on_release(doc, 280, 180)
    assert not tool.snap_lines


def test_snap_state_is_cleared_on_release():
    from photo_editor.tools.move.move_tool import MoveTool

    doc, a, b = _doc_with_layers()
    doc.layers.active_index = doc.layers.layers.index(b)
    tool = MoveTool()
    tool.auto_select = False
    tool.snap_engine.enabled = True
    tool.on_press(doc, 280, 180, 1.0)
    tool.on_move(doc, 183, 180, 1.0)
    tool.on_release(doc, 183, 180)
    assert not tool.snap_engine.active
    assert tool.snap_lines == []
