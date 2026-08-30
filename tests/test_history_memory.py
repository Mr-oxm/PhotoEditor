"""Undo history: correctness under structural sharing, and its memory bound.

Sharing buffers between snapshots is what makes multi-layer undo affordable,
and it is also exactly where aliasing bugs hide: if a restored layer ends up
holding the *same* array a snapshot holds, the next edit silently corrupts
history. These tests cover both the memory claim and the aliasing hazard.
"""

from __future__ import annotations

import numpy as np
import pytest

from photo_editor.core.document import Document
from photo_editor.core.history import HistoryManager, HistoryState

MB = 1 << 20


def _edit(layer, value, region=None):
    """Mutate a layer in place exactly as a painting tool does.

    Tools must take private ownership of the buffer before writing, because
    undo snapshots hold references to it. begin_write() does that and bumps
    the content version in one step.
    """
    layer.begin_write()
    if region is None:
        layer.pixels[:] = value
    else:
        layer.pixels[region] = value


def _doc(layers: int = 4, w: int = 64, h: int = 48) -> Document:
    doc = Document(w, h, name="hist")
    for i in range(layers - 1):
        doc.add_layer(f"L{i}")
    doc.history.clear()
    return doc


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def test_undo_restores_pixels():
    """Snapshots capture state *before* an edit -- tools call save_snapshot
    on mouse-down. Undo then returns the document to that pre-edit state."""
    doc = _doc(2)
    layer = doc.layers.layers[1]
    _edit(layer, 0.25)

    doc.save_snapshot("stroke")      # pre-edit, as a tool would
    _edit(layer, 0.75)

    doc.undo()
    restored = doc.layers.get(layer.id)
    assert np.allclose(restored.pixels, 0.25), "undo did not restore pixels"


def test_undo_redo_round_trip():
    doc = _doc(2)
    layer = doc.layers.layers[1]
    _edit(layer, 0.1)

    doc.save_snapshot("stroke")      # pre-edit
    _edit(layer, 0.9)

    doc.undo()
    assert np.allclose(doc.layers.get(layer.id).pixels, 0.1)
    doc.redo()
    assert np.allclose(doc.layers.get(layer.id).pixels, 0.9)


def test_restored_layer_does_not_alias_history():
    """The critical aliasing hazard: because snapshots share buffers, an
    in-place edit after an undo must not retroactively rewrite the stored
    state it was restored from."""
    doc = _doc(2)
    layer = doc.layers.layers[1]
    _edit(layer, 0.2)
    doc.save_snapshot("stroke")
    _edit(layer, 0.8)

    doc.undo()
    restored = doc.layers.get(layer.id)
    stored = doc.history.states[0].layer_data[layer.id]
    stored_before = stored.copy()

    # Mutate in place, exactly as a painting tool would.
    _edit(restored, 0.5)

    np.testing.assert_array_equal(
        stored, stored_before,
        "in-place edit after undo corrupted a stored history state",
    )


def test_unchanged_layers_share_buffers():
    """Only the edited layer should be copied into the new snapshot."""
    doc = _doc(5)
    doc.save_snapshot("base")
    first = doc.history.states[-1]

    edited = doc.layers.layers[2]
    _edit(edited, 0.42)
    doc.save_snapshot("edit-one-layer")
    second = doc.history.states[-1]

    shared = sum(
        1 for k, v in second.layer_data.items()
        if k in first.layer_data and v is first.layer_data[k]
    )
    copied = len(second.layer_data) - shared
    assert copied <= 2, (
        f"{copied} buffers copied for a one-layer edit; expected <= 2 "
        f"(pixels of the edited layer). Sharing is not working."
    )
    assert shared >= 3, f"only {shared} buffers shared"


def test_snapshot_after_structural_change_is_correct():
    """add_layer snapshots *after* inserting, so the pre-add state is the
    one before it; undo has to step back past the live state first."""
    doc = _doc(3)
    n_before = len(doc.layers.layers)
    doc.save_snapshot("before-add")
    doc.add_layer("new")
    assert len(doc.layers.layers) == n_before + 1
    doc.undo()
    doc.undo()
    assert len(doc.layers.layers) == n_before, "undo did not remove the layer"


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def test_history_respects_byte_budget():
    mgr = HistoryManager(max_states=1000, budget_bytes=4 * MB)
    for i in range(40):
        st = HistoryState(name=f"s{i}")
        st.layer_data["a"] = np.zeros((256, 256, 4), dtype=np.float32)  # 1 MB
        st.layer_versions["a"] = i
        mgr.push(st)
    assert mgr.total_bytes() <= 4 * MB, mgr.stats()
    assert len(mgr.states) >= 2, "budget eviction must leave undo usable"


def test_budget_counts_shared_arrays_once():
    mgr = HistoryManager(budget_bytes=1 << 30)
    shared = np.zeros((256, 256, 4), dtype=np.float32)  # 1 MB
    for i in range(10):
        st = HistoryState(name=f"s{i}")
        st.layer_data["a"] = shared        # same object every time
        st.layer_versions["a"] = 0
        mgr.push(st)
    assert mgr.total_bytes() == shared.nbytes, (
        f"shared array counted {mgr.total_bytes() / shared.nbytes:.0f} times"
    )


def test_repeated_snapshots_of_static_document_stay_flat():
    """Snapshotting without editing must not grow memory linearly."""
    doc = _doc(6)
    doc.save_snapshot("s0")
    after_first = doc.history.total_bytes()
    for i in range(15):
        doc.save_snapshot(f"s{i + 1}")
    after_many = doc.history.total_bytes()
    assert after_many == after_first, (
        f"16 no-op snapshots grew history from {after_first} to {after_many} bytes"
    )


def test_history_stats_shape():
    doc = _doc(2)
    doc.save_snapshot("x")
    stats = doc.history.stats()
    assert {"states", "index", "bytes", "mb", "budget_mb"} <= set(stats)
    assert stats["states"] == 1


@pytest.mark.parametrize("n_layers", [4, 12])
def test_single_layer_edit_cost_is_independent_of_stack_depth(n_layers):
    """Snapshot cost after an edit should scale with changed layers, not total."""
    doc = _doc(n_layers)
    doc.save_snapshot("base")
    base_bytes = doc.history.total_bytes()

    layer = doc.layers.layers[1]
    _edit(layer, 0.33)
    doc.save_snapshot("edit")

    growth = doc.history.total_bytes() - base_bytes
    one_layer = doc.layers.layers[1].pixels.nbytes
    assert growth <= one_layer * 2, (
        f"editing one layer grew history by {growth} bytes "
        f"({growth / one_layer:.1f} layers' worth) with {n_layers} layers"
    )


def test_history_cost_is_independent_of_layer_count():
    """The whole point of copy-on-write: a 20-layer document must not cost
    20x a 1-layer document to undo."""
    costs = {}
    for n in (2, 8):
        doc = _doc(n)
        for i in range(6):
            doc.save_snapshot(f"s{i}")
            layer = doc.layers.layers[1]
            _edit(layer, 0.1 * i)
        costs[n] = doc.history.owned_bytes(doc.live_buffer_ids())
    ratio = costs[8] / max(1, costs[2])
    assert ratio < 1.5, (
        f"history grew {ratio:.1f}x going from 2 to 8 layers "
        f"({costs[2]} -> {costs[8]} bytes); sharing is not working"
    )


def test_owned_bytes_excludes_live_buffers():
    doc = _doc(3)
    doc.save_snapshot("only")
    live = doc.live_buffer_ids()
    # Straight after a snapshot every stored array is still the live one.
    assert doc.history.owned_bytes(live) == 0, (
        "a snapshot that copied nothing should own no memory"
    )
    assert doc.history.total_bytes() > 0


def test_frozen_buffers_are_read_only():
    """Enforcement: a missed begin_write() must fail loudly, not silently
    corrupt the snapshot that shares the buffer."""
    doc = _doc(2)
    doc.save_snapshot("frozen")
    layer = doc.layers.layers[1]
    with pytest.raises(ValueError, match="read-only"):
        layer.pixels[:] = 0.5
    layer.begin_write()
    layer.pixels[:] = 0.5      # now fine
    assert np.allclose(layer.pixels, 0.5)
