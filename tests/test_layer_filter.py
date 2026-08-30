"""Layer filtering rules.

Pure logic, so the interesting cases -- what a match drags in with it --
can be tested directly.
"""

from __future__ import annotations

import pytest

from photo_editor.core.document import Document
from photo_editor.core.enums import LayerType
from photo_editor.core.layer import Layer
from photo_editor.core.layer_filter import (
    LayerFilter, match_count, visible_layer_ids,
)


def _doc():
    """A small hierarchy:  Sky(group) > [Clouds, Sun],  Ground,  Text."""
    doc = Document(64, 48, name="filter")
    doc.layers.layers.clear()
    sky = Layer(name="Sky", width=64, height=48, layer_type=LayerType.GROUP)
    doc.layers.add(sky)
    clouds = Layer(name="Clouds", width=32, height=24)
    clouds.parent_id = sky.id
    doc.layers.add(clouds)
    sun = Layer(name="Sun", width=16, height=16)
    sun.parent_id = sky.id
    doc.layers.add(sun)
    ground = Layer(name="Ground", width=64, height=16)
    doc.layers.add(ground)
    caption = Layer(name="Caption", width=40, height=10,
                    layer_type=LayerType.TEXT)
    doc.layers.add(caption)
    return doc, {l.name: l for l in doc.layers}


def _names(doc, ids):
    if ids is None:
        return None
    return {l.name for l in doc.layers if l.id in ids}


# ---------------------------------------------------------------------------
# No filter
# ---------------------------------------------------------------------------

def test_empty_filter_is_inactive():
    assert not LayerFilter().is_active


def test_no_filter_returns_none():
    doc, _ = _doc()
    assert visible_layer_ids(doc, None) is None
    assert visible_layer_ids(doc, LayerFilter()) is None


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def test_text_match_is_case_insensitive_substring():
    doc, _ = _doc()
    assert _names(doc, visible_layer_ids(doc, LayerFilter(text="clou"))) == {
        "Clouds", "Sky"}          # Sky comes along as the ancestor
    assert _names(doc, visible_layer_ids(doc, LayerFilter(text="CLOUDS"))) == {
        "Clouds", "Sky"}


def test_a_match_brings_its_ancestors_for_context():
    """A layer three groups deep should appear in place, not contextless."""
    doc, _ = _doc()
    shown = _names(doc, visible_layer_ids(doc, LayerFilter(text="Sun")))
    assert shown == {"Sun", "Sky"}


def test_matching_a_group_brings_its_whole_subtree():
    doc, _ = _doc()
    shown = _names(doc, visible_layer_ids(doc, LayerFilter(text="Sky")))
    assert shown == {"Sky", "Clouds", "Sun"}


def test_no_matches_shows_nothing():
    doc, _ = _doc()
    assert visible_layer_ids(doc, LayerFilter(text="zzz")) == set()


def test_whitespace_only_text_is_inactive():
    assert not LayerFilter(text="   ").is_active


# ---------------------------------------------------------------------------
# Kinds
# ---------------------------------------------------------------------------

def test_kind_filter():
    doc, _ = _doc()
    shown = _names(doc, visible_layer_ids(doc, LayerFilter(kinds={"text"})))
    assert shown == {"Caption"}


def test_kind_filter_combines_with_text():
    doc, _ = _doc()
    f = LayerFilter(text="c", kinds={"text"})
    assert _names(doc, visible_layer_ids(doc, f)) == {"Caption"}


def test_multiple_kinds_are_a_union():
    doc, _ = _doc()
    f = LayerFilter(kinds={"text", "group"})
    shown = _names(doc, visible_layer_ids(doc, f))
    # the group brings its children with it
    assert shown == {"Caption", "Sky", "Clouds", "Sun"}


@pytest.mark.parametrize("kind,layer_type", [
    ("raster", LayerType.RASTER),
    ("shape", LayerType.SHAPE),
    ("adjustment", LayerType.ADJUSTMENT),
    ("filter", LayerType.FILTER),
    ("mask", LayerType.MASK),
])
def test_every_kind_maps_to_a_layer_type(kind, layer_type):
    from photo_editor.core.layer_filter import KIND_GROUPS
    assert layer_type in KIND_GROUPS[kind]


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------

def test_visible_only():
    doc, by_name = _doc()
    by_name["Ground"].visible = False
    shown = _names(doc, visible_layer_ids(doc, LayerFilter(visible_only=True)))
    assert "Ground" not in shown
    assert "Caption" in shown


def test_locked_only():
    doc, by_name = _doc()
    by_name["Ground"].locked = True
    shown = _names(doc, visible_layer_ids(doc, LayerFilter(locked_only=True)))
    assert shown == {"Ground"}


def test_with_effects_only():
    doc, by_name = _doc()
    by_name["Ground"].add_mask()
    shown = _names(doc,
                   visible_layer_ids(doc, LayerFilter(with_effects_only=True)))
    assert shown == {"Ground"}


# ---------------------------------------------------------------------------
# Counting and clearing
# ---------------------------------------------------------------------------

def test_match_count_ignores_ancestors():
    """The count should say how many layers matched, not how many rows show."""
    doc, _ = _doc()
    assert match_count(doc, LayerFilter(text="Sun")) == 1
    assert len(visible_layer_ids(doc, LayerFilter(text="Sun"))) == 2


def test_match_count_with_no_filter_is_everything():
    doc, _ = _doc()
    assert match_count(doc, None) == len(doc.layers.layers)


def test_clear_deactivates():
    f = LayerFilter(text="x", kinds={"text"}, visible_only=True)
    assert f.is_active
    f.clear()
    assert not f.is_active


# ---------------------------------------------------------------------------
# Panel integration
# ---------------------------------------------------------------------------

def _panel(qtbot):
    from photo_editor.ui.panels.layers.panel import LayersPanel
    panel = LayersPanel()
    qtbot.addWidget(panel)
    return panel


def _rows(panel):
    return [i for i in panel._row_layer_ids if i != "__sep__"]


def test_panel_shows_every_layer_by_default(qtbot):
    doc, _ = _doc()
    panel = _panel(qtbot)
    panel.refresh(doc)
    assert len(_rows(panel)) == len(doc.layers.layers)


def test_panel_narrows_to_matches(qtbot):
    doc, by_name = _doc()
    panel = _panel(qtbot)
    panel.refresh(doc)
    panel._search.setText("Caption")
    shown = {l.name for l in doc.layers if l.id in set(_rows(panel))}
    assert shown == {"Caption"}


def test_panel_keeps_ancestors_visible(qtbot):
    doc, _ = _doc()
    panel = _panel(qtbot)
    panel.refresh(doc)
    panel._search.setText("Sun")
    shown = {l.name for l in doc.layers if l.id in set(_rows(panel))}
    assert shown == {"Sun", "Sky"}, shown


def test_panel_reports_the_match_count(qtbot):
    doc, _ = _doc()
    panel = _panel(qtbot)
    panel.refresh(doc)
    panel._search.setText("Sun")
    assert panel._filter_status.isVisibleTo(panel)
    assert "1 of 5" in panel._filter_status.text()


def test_panel_reports_no_matches(qtbot):
    doc, _ = _doc()
    panel = _panel(qtbot)
    panel.refresh(doc)
    panel._search.setText("nothing-here")
    assert _rows(panel) == []
    assert "No layers match" in panel._filter_status.text()


def test_clearing_the_filter_restores_every_layer(qtbot):
    doc, _ = _doc()
    panel = _panel(qtbot)
    panel.refresh(doc)
    panel._search.setText("Sun")
    assert len(_rows(panel)) == 2
    panel.clear_filter()
    assert len(_rows(panel)) == len(doc.layers.layers)
    assert not panel._filter_status.isVisibleTo(panel)


def test_filtering_expands_collapsed_groups(qtbot):
    """Collapsing away the rows the query just selected would be perverse."""
    doc, by_name = _doc()
    panel = _panel(qtbot)
    panel.refresh(doc)
    panel._collapsed_groups.add(by_name["Sky"].id)
    panel.refresh(doc, force=True)
    assert by_name["Sun"].id not in _rows(panel)     # collapsed away

    panel._search.setText("Sun")
    assert by_name["Sun"].id in _rows(panel), (
        "filtering did not reveal a match inside a collapsed group")


def test_kind_combo_filters(qtbot):
    doc, _ = _doc()
    panel = _panel(qtbot)
    panel.refresh(doc)
    index = panel._kind_combo.findData("text")
    panel._kind_combo.setCurrentIndex(index)
    shown = {l.name for l in doc.layers if l.id in set(_rows(panel))}
    assert shown == {"Caption"}
