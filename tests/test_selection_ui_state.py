import numpy as np
import pytest

from photo_editor.core.selection import Selection
from photo_editor.ui.services.selection_ui_state import apply_selection_overlay


class FakeCanvas:
    def __init__(self) -> None:
        self.selection_masks = []
        self.versions = []

    def set_selection_mask(self, mask, version=None) -> None:
        self.selection_masks.append(mask)
        self.versions.append(version)


def test_apply_selection_overlay_shows_non_empty_mask() -> None:
    canvas = FakeCanvas()
    mask = np.array([[0.0, 1.0]], dtype=np.float32)

    apply_selection_overlay(canvas, mask)

    assert canvas.selection_masks == [mask]


def test_apply_selection_overlay_hides_empty_or_missing_mask() -> None:
    canvas = FakeCanvas()

    apply_selection_overlay(canvas, np.zeros((1, 1), dtype=np.float32))
    apply_selection_overlay(canvas, None)

    assert canvas.selection_masks == [None, None]


def test_apply_selection_overlay_forwards_version() -> None:
    """The version lets the canvas skip re-tracing unchanged contours."""
    canvas = FakeCanvas()
    mask = np.array([[0.0, 1.0]], dtype=np.float32)
    apply_selection_overlay(canvas, mask, version=7, is_empty=False)
    assert canvas.versions == [7]


def test_apply_selection_overlay_trusts_precomputed_emptiness() -> None:
    """Passing is_empty avoids a full-document mask.max() per frame."""
    canvas = FakeCanvas()
    mask = np.ones((4, 4), dtype=np.float32)
    apply_selection_overlay(canvas, mask, version=1, is_empty=True)
    assert canvas.selection_masks == [None]


# ---------------------------------------------------------------------------
# Selection change tracking
# ---------------------------------------------------------------------------

def test_selection_version_advances_on_change() -> None:
    sel = Selection(32, 24)
    v0 = sel.version
    sel.select_rect(2, 2, 10, 10)
    assert sel.version > v0
    v1 = sel.version
    sel.invert()
    assert sel.version > v1


def test_selection_bounds_are_cached_and_correct() -> None:
    sel = Selection(32, 24)
    sel.select_rect(4, 6, 10, 8)
    assert sel.bounds == (4, 6, 13, 13)
    # Cached: repeated reads must not recompute (same object identity).
    first = sel.bounds
    assert sel.bounds is first


def test_selection_bounds_follow_changes() -> None:
    sel = Selection(32, 24)
    sel.select_rect(0, 0, 5, 5)
    assert sel.bounds == (0, 0, 4, 4)
    sel.select_rect(10, 10, 4, 4)
    assert sel.bounds == (10, 10, 13, 13)


def test_selection_is_empty() -> None:
    sel = Selection(16, 16)
    assert sel.is_empty          # no mask at all
    sel.select_all()
    assert not sel.is_empty
    sel.deselect()
    assert sel.is_empty


def test_selection_bounds_none_for_empty_mask() -> None:
    sel = Selection(16, 16)
    sel.select_rect(0, 0, 0, 0)   # selects nothing
    assert sel.bounds is None
    assert sel.is_empty


def test_touch_invalidates_cached_bounds() -> None:
    """In-place mask edits must be signalled, like layer pixels."""
    sel = Selection(16, 16)
    sel.select_rect(0, 0, 4, 4)
    assert sel.bounds == (0, 0, 3, 3)
    sel._mask[10:14, 10:14] = 1.0
    sel.touch()
    assert sel.bounds == (0, 0, 13, 13)


@pytest.mark.parametrize("mode", ["add", "subtract", "intersect", "new"])
def test_selection_modes_bump_the_version(mode):
    """Every selection combine mode must move Selection.version.

    The canvas skips re-tracing marching-ants contours and the transform box
    reuses cached bounds when the version is unchanged, so a mode that
    assigned Selection._mask directly left both showing the *previous*
    selection until something else happened to invalidate them.
    """
    from photo_editor.core.document import Document
    from photo_editor.tools.selection_tools import _apply_mode

    doc = Document(64, 48)
    doc.selection.select_rect(0, 0, 20, 20)
    before_version = doc.selection.version

    new_mask = np.zeros((48, 64), dtype=np.float32)
    new_mask[10:30, 10:30] = 1.0
    _apply_mode(doc, new_mask, mode)

    assert doc.selection.version > before_version, (
        f"'{mode}' did not bump the selection version")

    # The cached bounds must match the mask as it now is. Comparing against
    # the previous value would not do: subtracting a corner legitimately
    # leaves the bounding box unchanged. Staleness is the thing under test.
    mask = doc.selection._mask
    if mask is not None and (mask > 0.5).any():
        rows = np.flatnonzero(np.any(mask > 0.5, axis=1))
        cols = np.flatnonzero(np.any(mask > 0.5, axis=0))
        expected = (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))
    else:
        expected = None
    assert doc.selection.bounds == expected, (
        f"'{mode}' left the cached bounds stale: "
        f"{doc.selection.bounds} != {expected}")
