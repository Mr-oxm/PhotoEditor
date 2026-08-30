"""Shared helpers for selection overlay state on the canvas."""

from __future__ import annotations


def apply_selection_overlay(canvas, mask, version: int | None = None,
                            is_empty: bool | None = None) -> None:
    """Show the selection mask only when it contains visible pixels.

    *version* lets the canvas skip re-tracing contours when the selection is
    unchanged. Without it, every rendered frame ran ``mask.max()`` plus
    ``cv2.findContours`` over the full document mask and rebuilt one
    ``QPointF`` per contour point in Python -- work whose result was
    identical to the previous frame's for the whole of a drag.
    """
    if mask is None:
        canvas.set_selection_mask(None)
        return
    empty = is_empty if is_empty is not None else not (mask.max() > 0)
    canvas.set_selection_mask(None if empty else mask, version=version)
