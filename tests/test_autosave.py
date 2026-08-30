"""Autosave and crash recovery.

The contract that matters: work is never lost after a crash, and work is
never *falsely* offered for recovery after a clean exit. Both directions
are tested, along with the "does no work when idle" property that keeps
autosave from becoming its own performance problem.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from photo_editor.core.document import Document
from photo_editor.utils import autosave as A


@pytest.fixture(autouse=True)
def isolated_recovery_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BASERA_RECOVERY_DIR", str(tmp_path / "recovery"))
    yield


def _dirty_doc(name="Doc"):
    doc = Document(64, 48, name=name)
    doc.layers.active_index = 0
    layer = doc.layers.active_layer
    layer.begin_write()
    layer.pixels[:] = 0.5
    doc.save_snapshot("edit")
    doc.mark_dirty()
    return doc


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def test_dirty_document_is_autosaved():
    mgr = A.AutosaveManager(session_id="s1", min_interval_seconds=0)
    doc = _dirty_doc()
    path = mgr.maybe_save("d1", doc)
    assert path is not None and path.exists()


def test_clean_document_is_not_autosaved():
    mgr = A.AutosaveManager(session_id="s1", min_interval_seconds=0)
    doc = Document(32, 32)
    doc.mark_clean()
    assert mgr.maybe_save("d1", doc) is None


def test_unchanged_document_is_not_written_twice():
    """An idle session must do no work -- autosave should not itself
    become a periodic 4 GB write."""
    mgr = A.AutosaveManager(session_id="s1", min_interval_seconds=0)
    doc = _dirty_doc()
    assert mgr.maybe_save("d1", doc) is not None
    assert mgr.maybe_save("d1", doc) is None, "unchanged document re-saved"


def test_further_edits_trigger_another_autosave():
    mgr = A.AutosaveManager(session_id="s1", min_interval_seconds=0)
    doc = _dirty_doc()
    mgr.maybe_save("d1", doc)
    doc.save_snapshot("another edit")
    assert mgr.maybe_save("d1", doc) is not None


def test_minimum_interval_is_respected():
    mgr = A.AutosaveManager(session_id="s1", min_interval_seconds=3600)
    doc = _dirty_doc()
    assert mgr.maybe_save("d1", doc) is not None
    doc.save_snapshot("edit 2")
    assert mgr.maybe_save("d1", doc) is None, "ignored the minimum interval"


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def test_crashed_session_is_recoverable():
    mgr = A.AutosaveManager(session_id="crashed", min_interval_seconds=0)
    doc = _dirty_doc("Important Work")
    mgr.save("d1", doc)
    # Simulate a crash: the process vanishes without calling release().
    (A.recovery_dir() / f"crashed{A._LIVE_SUFFIX}").unlink()

    found = A.find_recoverable()
    assert len(found) == 1
    assert found[0].name == "Important Work"
    assert found[0].doc_path.exists()


def test_clean_shutdown_leaves_nothing_to_recover():
    mgr = A.AutosaveManager(session_id="clean", min_interval_seconds=0)
    mgr.save("d1", _dirty_doc())
    mgr.release()
    assert A.find_recoverable() == []


def test_live_session_is_not_offered_for_recovery():
    """A second window must not offer to 'recover' documents this one has
    open right now."""
    mgr = A.AutosaveManager(session_id="live", min_interval_seconds=0)
    mgr.save("d1", _dirty_doc())
    assert A.find_recoverable(exclude_session="live") == []
    assert A.find_recoverable() == [], "a live session was offered for recovery"


def test_recovered_document_loads_with_its_pixels():
    from photo_editor.utils.project_io import load_basera_project
    mgr = A.AutosaveManager(session_id="crashed", min_interval_seconds=0)
    doc = _dirty_doc()
    doc.layers.active_layer.begin_write()
    doc.layers.active_layer.pixels[:] = np.array([0.25, 0.5, 0.75, 1.0],
                                                 dtype=np.float32)
    mgr.save("d1", doc)
    (A.recovery_dir() / f"crashed{A._LIVE_SUFFIX}").unlink()

    entry = A.find_recoverable()[0]
    restored = load_basera_project(entry.doc_path)
    assert np.allclose(restored.layers.layers[0].pixels[0, 0],
                       [0.25, 0.5, 0.75, 1.0], atol=1e-4)


def test_forget_removes_a_documents_autosave():
    mgr = A.AutosaveManager(session_id="s1", min_interval_seconds=0)
    mgr.save("d1", _dirty_doc())
    mgr.forget("d1")
    (A.recovery_dir() / f"s1{A._LIVE_SUFFIX}").unlink()
    assert A.find_recoverable() == []


def test_discard_removes_the_files():
    mgr = A.AutosaveManager(session_id="crashed", min_interval_seconds=0)
    mgr.save("d1", _dirty_doc())
    (A.recovery_dir() / f"crashed{A._LIVE_SUFFIX}").unlink()
    entry = A.find_recoverable()[0]
    entry.discard()
    assert not entry.doc_path.exists()
    assert A.find_recoverable() == []


def test_orphaned_sidecar_is_cleaned_up():
    mgr = A.AutosaveManager(session_id="crashed", min_interval_seconds=0)
    mgr.save("d1", _dirty_doc())
    (A.recovery_dir() / f"crashed{A._LIVE_SUFFIX}").unlink()
    entry = A.find_recoverable()[0]
    entry.doc_path.unlink()          # document gone, sidecar left behind
    assert A.find_recoverable() == []
    assert not entry.sidecar.exists()


def test_corrupt_sidecar_is_ignored():
    bad = A.recovery_dir() / f"junk{A._SIDECAR_SUFFIX}"
    bad.write_text("not json at all")
    assert A.find_recoverable() == []


def test_entries_are_newest_first():
    for i, session in enumerate(("old", "new")):
        mgr = A.AutosaveManager(session_id=session, min_interval_seconds=0)
        mgr.save("d1", _dirty_doc(f"Doc{i}"))
        (A.recovery_dir() / f"{session}{A._LIVE_SUFFIX}").unlink()
        time.sleep(0.01)
    found = A.find_recoverable()
    assert [e.name for e in found] == ["Doc1", "Doc0"]


def test_age_description_is_human():
    entry = A.RecoveryEntry(
        doc_path=A.recovery_dir() / "x", sidecar=A.recovery_dir() / "y",
        name="n", original_path=None, saved_at=time.time() - 300)
    assert "minutes ago" in entry.describe_age()


def test_original_path_is_remembered():
    mgr = A.AutosaveManager(session_id="crashed", min_interval_seconds=0)
    doc = _dirty_doc()
    doc.file_path = "/somewhere/project.basera"
    mgr.save("d1", doc)
    (A.recovery_dir() / f"crashed{A._LIVE_SUFFIX}").unlink()
    assert A.find_recoverable()[0].original_path == "/somewhere/project.basera"


# ---------------------------------------------------------------------------
# Integration with the window
# ---------------------------------------------------------------------------

def test_window_autosaves_dirty_documents(qtbot):
    from photo_editor.ui.main_window import MainWindow
    win = MainWindow(dev_mode=True)
    qtbot.addWidget(win)
    try:
        layer = win._doc.layers.active_layer
        layer.begin_write()
        layer.pixels[:] = 0.5
        win._doc.save_snapshot("edit")
        win._doc.mark_dirty()
        win._autosave._min_interval = 0

        win._run_autosave()
        qtbot.waitUntil(lambda: not getattr(win, "_autosave_busy", False),
                        timeout=15000)
        assert list(A.recovery_dir().glob(f"*{A._DOC_SUFFIX}")), (
            "a dirty document was not autosaved")
    finally:
        # Leaving the document dirty makes teardown's close raise the modal
        # unsaved-changes prompt, which blocks forever without a user.
        win._autosave_timer.stop()
        win._doc.mark_clean()


def test_window_release_clears_autosaves_on_close(qtbot):
    from photo_editor.ui.main_window import MainWindow
    win = MainWindow(dev_mode=True)
    qtbot.addWidget(win)
    try:
        win._autosave.save("d0", _dirty_doc())
        assert list(A.recovery_dir().glob(f"*{A._DOC_SUFFIX}"))
        win._autosave_timer.stop()
        win._autosave.release()
        assert A.find_recoverable() == []
    finally:
        win._autosave_timer.stop()
        win._doc.mark_clean()


def test_recovery_dialog_lists_and_discards(qtbot):
    from photo_editor.ui.dialogs.recovery_dialog import RecoveryDialog
    mgr = A.AutosaveManager(session_id="crashed", min_interval_seconds=0)
    mgr.save("d1", _dirty_doc("Recovered Doc"))
    (A.recovery_dir() / f"crashed{A._LIVE_SUFFIX}").unlink()

    entries = A.find_recoverable()
    dlg = RecoveryDialog(entries)
    qtbot.addWidget(dlg)
    assert dlg._list.count() == 1
    assert dlg.selected_entries() == entries      # pre-selected

    dlg._on_discard()
    assert dlg._list.count() == 0
    assert A.find_recoverable() == []
