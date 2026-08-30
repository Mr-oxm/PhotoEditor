"""Caches in the UI layer: they must be correct before they are fast.

A stale thumbnail or a stale channel preview is a visible bug, so each of
these caches is tested for invalidation as well as for reuse.
"""

from __future__ import annotations

import numpy as np
import pytest

from photo_editor.core.document import Document
from photo_editor.core.enums import LayerType
from photo_editor.core.layer import Layer


def _doc():
    doc = Document(64, 48, name="ui-cache")
    doc.layers.active_index = 0
    return doc


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------

def test_thumbnail_is_cached_between_calls(qtbot):
    from photo_editor.ui.panels.layers.thumbnails import make_thumbnail
    doc = _doc()
    layer = doc.layers.active_layer
    first = make_thumbnail(layer)
    assert make_thumbnail(layer) is first, "thumbnail was regenerated"


def test_thumbnail_refreshes_after_a_pixel_edit(qtbot):
    """The bug this replaces: thumbnails were keyed on layer.id alone and
    invalidate_thumbnail() had no callers, so they never updated."""
    from photo_editor.ui.panels.layers.thumbnails import make_thumbnail
    doc = _doc()
    layer = doc.layers.active_layer
    layer.begin_write()
    layer.pixels[:] = np.array([1, 0, 0, 1], dtype=np.float32)
    red = make_thumbnail(layer)

    layer.begin_write()
    layer.pixels[:] = np.array([0, 0, 1, 1], dtype=np.float32)
    blue = make_thumbnail(layer)

    assert blue is not red, "thumbnail did not update after an edit"
    assert (red.toImage().pixelColor(4, 4).red()
            != blue.toImage().pixelColor(4, 4).red())


def test_group_thumbnail_follows_its_children(qtbot):
    from photo_editor.ui.panels.layers.thumbnails import make_group_thumbnail
    doc = _doc()
    group = Layer(name="g", width=64, height=48, layer_type=LayerType.GROUP)
    doc.layers.add(group)
    child = Layer(name="c", width=32, height=24)
    child.parent_id = group.id
    px = np.zeros((24, 32, 4), dtype=np.float32)
    px[..., 0] = 1.0
    px[..., 3] = 1.0
    child.pixels = px
    doc.layers.add(child)

    first = make_group_thumbnail(doc, group)
    child.begin_write()
    child.pixels[..., 0] = 0.0
    child.pixels[..., 2] = 1.0
    second = make_group_thumbnail(doc, group)
    assert second is not first, "group thumbnail ignored a child edit"


def test_invalidate_thumbnail_drops_every_version(qtbot):
    from photo_editor.ui.panels.layers import thumbnails
    doc = _doc()
    layer = doc.layers.active_layer
    thumbnails.make_thumbnail(layer)
    layer.begin_write()
    layer.pixels[:] = 0.5
    thumbnails.make_thumbnail(layer)
    assert any(k[0] == layer.id for k in thumbnails._thumb_cache)
    thumbnails.invalidate_thumbnail(layer.id)
    assert not any(k[0] == layer.id for k in thumbnails._thumb_cache)


# ---------------------------------------------------------------------------
# Icons
# ---------------------------------------------------------------------------

def test_icons_are_cached(qtbot):
    from photo_editor.ui.icons.layers import (
        clear_icon_cache, icon_eye, icon_lock, icon_mask,
    )
    clear_icon_cache()
    assert icon_eye(True) is icon_eye(True)
    assert icon_eye(False) is icon_eye(False)
    assert icon_eye(True) is not icon_eye(False)
    assert icon_lock(True) is icon_lock(True)
    assert icon_mask(True) is icon_mask(True)


def test_clear_icon_cache_forces_a_redraw(qtbot):
    from photo_editor.ui.icons.layers import clear_icon_cache, icon_eye
    clear_icon_cache()
    first = icon_eye(True)
    clear_icon_cache()
    assert icon_eye(True) is not first


# ---------------------------------------------------------------------------
# Channels panel
# ---------------------------------------------------------------------------

def test_channels_panel_skips_unchanged_refreshes(qtbot):
    """It used to re-derive four previews -- and re-composite the active
    group -- on every rendered frame."""
    from photo_editor.ui.panels.channels_panel import ChannelsPanel
    panel = ChannelsPanel()
    qtbot.addWidget(panel)
    doc = _doc()

    calls = []
    original = panel._update_ui_values
    panel._update_ui_values = lambda: (calls.append(1), original())[1]

    panel.refresh(doc)
    assert len(calls) == 1
    panel.refresh(doc)
    panel.refresh(doc)
    assert len(calls) == 1, "unchanged refresh still rebuilt the previews"

    doc.layers.active_layer.begin_write()
    doc.layers.active_layer.pixels[:] = 0.3
    panel.refresh(doc)
    assert len(calls) == 2, "a pixel edit did not refresh the panel"


def test_channels_panel_follows_channel_toggles(qtbot):
    from photo_editor.ui.panels.channels_panel import ChannelsPanel
    panel = ChannelsPanel()
    qtbot.addWidget(panel)
    doc = _doc()
    calls = []
    original = panel._update_ui_values
    panel._update_ui_values = lambda: (calls.append(1), original())[1]

    panel.refresh(doc)
    doc.layers.active_layer.channel_r = False
    panel.refresh(doc)
    assert len(calls) == 2
