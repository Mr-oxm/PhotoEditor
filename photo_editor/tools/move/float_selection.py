"""Floating-selection mixin for the Move tool.

When the user has an active selection and starts a Move drag, the pixels
inside the selection are *cut* into a temporary floating buffer and only
those pixels follow the cursor.  The rest of the layer becomes the
"base-with-hole" beneath the float.

On ``commit_float`` the floating buffer is composited back onto the layer
at its final offset, and the selection mask is translated to match.

Exported symbol
---------------
FloatSelectionMixin
    Mixin class containing ``commit_float``, ``_clear_float``, and
    ``_composite_float``.  Expects the host class to carry the floating-
    selection state attributes set up in ``MoveTool.__init__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...core.document import Document


class FloatSelectionMixin:
    """Mixin that manages the floating-selection state for the Move tool.

    Attribute contract (must be initialised by the host ``__init__``)::

        _floating: bool
        _float_pixels: np.ndarray | None
        _float_base: np.ndarray | None
        _float_orig: np.ndarray | None
        _float_dx: int
        _float_dy: int
        _float_committed_dx: int
        _float_committed_dy: int
        _active_layer: Any
    """

    def commit_float(self, doc: "Document | None" = None) -> None:
        """Permanently apply the floating selection to the layer pixels.

        Called when switching tools or starting a non-float interaction.
        The base pixels under the float's new position are replaced.
        """
        if not self._floating or self._float_pixels is None or self._active_layer is None:
            self._clear_float()
            return
        layer = self._active_layer
        total_dx = self._float_committed_dx
        total_dy = self._float_committed_dy
        # Restore base-with-hole, then composite float at final position
        layer.begin_write()
        layer.pixels[:] = self._float_base
        self._composite_float(layer.pixels, total_dx, total_dy)
        # Translate the selection mask to follow the float's final position
        if doc is not None and doc.selection.active and (total_dx != 0 or total_dy != 0):
            doc.selection.translate(int(total_dx), int(total_dy))
        self._clear_float()

    def _clear_float(self) -> None:
        """Reset all floating-selection state to idle."""
        self._floating = False
        self._float_pixels = None
        self._float_base = None
        self._float_orig = None
        self._float_dx = 0
        self._float_dy = 0
        self._float_committed_dx = 0
        self._float_committed_dy = 0

    def _composite_float(self, target: np.ndarray, dx: int, dy: int) -> None:
        """Alpha-composite the floating pixels onto *target* at offset *(dx, dy)*.

        Pixels outside the target bounds are silently clipped.
        """
        fp = self._float_pixels
        if fp is None:
            return
        h, w = target.shape[:2]
        fh, fw = fp.shape[:2]
        sy0 = max(0, -dy)
        sx0 = max(0, -dx)
        sy1 = min(fh, h - dy)
        sx1 = min(fw, w - dx)
        d_y0 = max(0, dy)
        d_x0 = max(0, dx)
        d_y1 = d_y0 + (sy1 - sy0)
        d_x1 = d_x0 + (sx1 - sx0)
        if sy1 > sy0 and sx1 > sx0:
            src = fp[sy0:sy1, sx0:sx1]
            roi = target[d_y0:d_y1, d_x0:d_x1]
            alpha = src[..., 3:4]
            roi[:] = roi * (1.0 - alpha) + src * alpha
            np.clip(roi, 0, 1, out=roi)
