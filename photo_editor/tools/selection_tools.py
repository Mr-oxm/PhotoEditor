"""Selection tools — Rect, Ellipse, Lasso, and Magic Wand selections.

Each tool supports four selection modes:
  - new       : Replace the current selection
  - add       : Add to the current selection  (Shift)
  - subtract  : Subtract from the current selection  (Alt)
  - intersect : Intersect with the current selection  (Shift+Alt)

The mode is stored on each tool instance as ``.mode`` and can be driven by
the properties bar or by keyboard modifiers at press time.
"""

from __future__ import annotations

import cv2
import numpy as np

from .tool_base import Tool
from ..core.document import Document


# ── Helpers ──────────────────────────────────────────────────────────────────

_MODES = ("new", "add", "subtract", "intersect")


def _apply_mode(doc: Document, new_mask: np.ndarray, mode: str) -> None:
    """Combine *new_mask* into the document selection according to *mode*."""
    sel = doc.selection
    if mode == "new" or sel._mask is None:
        sel._set_mask(new_mask)
        return
    old = sel._mask
    # Ensure same shape — pad / crop if needed
    if old.shape != new_mask.shape:
        h, w = doc.height, doc.width
        if old.shape != (h, w):
            tmp = np.zeros((h, w), dtype=np.float32)
            oh, ow = min(old.shape[0], h), min(old.shape[1], w)
            tmp[:oh, :ow] = old[:oh, :ow]
            old = tmp
        if new_mask.shape != (h, w):
            tmp = np.zeros((h, w), dtype=np.float32)
            nh, nw = min(new_mask.shape[0], h), min(new_mask.shape[1], w)
            tmp[:nh, :nw] = new_mask[:nh, :nw]
            new_mask = tmp
    # Through _set_mask, so Selection.version moves. The canvas skips
    # re-tracing marching-ants contours and the transform box reuses cached
    # bounds when the version is unchanged, so a direct assignment left both
    # showing the previous selection.
    if mode == "add":
        sel._set_mask(np.maximum(old, new_mask))
    elif mode == "subtract":
        sel._set_mask(np.clip(old - new_mask, 0, 1))
    elif mode == "intersect":
        sel._set_mask(np.minimum(old, new_mask))
    else:
        sel._set_mask(new_mask)


# ======================================================================
# Rectangular selection
# ======================================================================


class RectSelectTool(Tool):
    """Drag to create a rectangular selection."""

    def __init__(self) -> None:
        super().__init__("Rectangular Select")
        self.feather: int = 0
        self.mode: str = "new"       # new / add / subtract / intersect
        self._start_x: int = 0
        self._start_y: int = 0
        self._dragging: bool = False

    def on_press(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        self._start_x, self._start_y = x, y
        self._dragging = True

    def on_move(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        pass  # live preview is handled by canvas drag-rect

    def on_release(self, doc: Document, x: int, y: int) -> None:
        if not self._dragging:
            return
        self._dragging = False
        rx = min(self._start_x, x)
        ry = min(self._start_y, y)
        rw = abs(x - self._start_x)
        rh = abs(y - self._start_y)
        if rw == 0 or rh == 0:
            if self.mode == "new":
                doc.selection.deselect()
            return
        mask = np.zeros((doc.height, doc.width), dtype=np.float32)
        x1, y1 = max(0, rx), max(0, ry)
        x2, y2 = min(doc.width, rx + rw), min(doc.height, ry + rh)
        mask[y1:y2, x1:x2] = 1.0
        if self.feather > 0:
            ksize = self.feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), self.feather / 3.0)
        _apply_mode(doc, mask, self.mode)


# ======================================================================
# Elliptical selection
# ======================================================================


class EllipseSelectTool(Tool):
    """Drag to create an elliptical selection."""

    def __init__(self) -> None:
        super().__init__("Elliptical Select")
        self.feather: int = 0
        self.mode: str = "new"
        self._start_x: int = 0
        self._start_y: int = 0
        self._dragging: bool = False

    def on_press(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        self._start_x, self._start_y = x, y
        self._dragging = True

    def on_move(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        pass

    def on_release(self, doc: Document, x: int, y: int) -> None:
        if not self._dragging:
            return
        self._dragging = False
        cx = (self._start_x + x) // 2
        cy = (self._start_y + y) // 2
        rx = abs(x - self._start_x) // 2
        ry = abs(y - self._start_y) // 2
        if rx == 0 or ry == 0:
            if self.mode == "new":
                doc.selection.deselect()
            return
        mask = np.zeros((doc.height, doc.width), dtype=np.float32)
        yy, xx = np.ogrid[:doc.height, :doc.width]
        ellipse = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2
        mask[ellipse <= 1.0] = 1.0
        if self.feather > 0:
            ksize = self.feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), self.feather / 3.0)
        _apply_mode(doc, mask, self.mode)


# ======================================================================
# Free-hand lasso selection
# ======================================================================


class LassoTool(Tool):
    """Free-hand selection by collecting points along a drag path."""

    def __init__(self) -> None:
        super().__init__("Lasso")
        self.feather: int = 0
        self.mode: str = "new"
        self._points: list[tuple[int, int]] = []
        self._drawing: bool = False

    def on_press(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        self._points = [(x, y)]
        self._drawing = True

    def on_move(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        if self._drawing:
            self._points.append((x, y))

    def on_release(self, doc: Document, x: int, y: int) -> None:
        if not self._drawing:
            return
        self._drawing = False
        self._points.append((x, y))

        if len(self._points) < 3:
            if self.mode == "new":
                doc.selection.deselect()
            return

        # Rasterise polygon into a document-sized mask
        mask = np.zeros((doc.height, doc.width), dtype=np.float32)
        pts = np.array(self._points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1.0)

        if self.feather > 0:
            ksize = self.feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), self.feather / 3.0)
        _apply_mode(doc, mask, self.mode)
        self._points.clear()


# ======================================================================
# Magic Wand (flood-fill tolerance selection)
# ======================================================================


class MagicWandTool(Tool):
    """Selects connected pixels similar to the clicked colour."""

    def __init__(self) -> None:
        super().__init__("Magic Wand")
        self.tolerance: int = 32   # 0–255 scale
        self.contiguous: bool = True
        self.feather: int = 0
        self.mode: str = "new"
        self.sample_all: bool = False  # True = sample merged, False = active layer

    def _flood_mask(self, pixels: np.ndarray, sx: int, sy: int) -> np.ndarray:
        """Flood-fill based contiguous selection (layer-local coords)."""
        h, w = pixels.shape[:2]
        tol = self.tolerance
        # Use 3-channel RGB for cv2.floodFill reliability
        rgb = np.clip(pixels[..., :3] * 255, 0, 255).astype(np.uint8)
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        lo_diff = (int(tol), int(tol), int(tol))
        hi_diff = (int(tol), int(tol), int(tol))
        cv2.floodFill(rgb, ff_mask, (sx, sy), 255,
                      loDiff=lo_diff, upDiff=hi_diff,
                      flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        return (ff_mask[1:-1, 1:-1].astype(np.float32) / 255.0)

    def _global_mask(self, pixels: np.ndarray, sx: int, sy: int) -> np.ndarray:
        """Non-contiguous: select all pixels within tolerance of seed colour."""
        seed = pixels[sy, sx, :3]  # Compare RGB only
        diff = np.abs(pixels[..., :3] - seed).max(axis=-1)
        return (diff <= self.tolerance / 255.0).astype(np.float32)

    def on_press(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        layer = doc.layers.active_layer
        if layer is None:
            return
        # Convert document coords to layer-local pixel coords
        lx, ly = layer.position
        px, py = x - lx, y - ly
        h, w = layer.pixels.shape[:2]
        if px < 0 or px >= w or py < 0 or py >= h:
            return

        if self.contiguous:
            local_mask = self._flood_mask(layer.pixels, px, py)
        else:
            local_mask = self._global_mask(layer.pixels, px, py)

        # Embed layer-local mask into a document-sized mask
        doc_mask = np.zeros((doc.height, doc.width), dtype=np.float32)
        # Compute overlap region
        dst_y1 = max(0, ly)
        dst_y2 = min(doc.height, ly + h)
        dst_x1 = max(0, lx)
        dst_x2 = min(doc.width, lx + w)
        src_y1 = dst_y1 - ly
        src_y2 = dst_y2 - ly
        src_x1 = dst_x1 - lx
        src_x2 = dst_x2 - lx
        if dst_y2 > dst_y1 and dst_x2 > dst_x1:
            doc_mask[dst_y1:dst_y2, dst_x1:dst_x2] = local_mask[src_y1:src_y2, src_x1:src_x2]

        if self.feather > 0:
            ksize = self.feather * 2 + 1
            doc_mask = cv2.GaussianBlur(doc_mask, (ksize, ksize), self.feather / 3.0)

        _apply_mode(doc, doc_mask, self.mode)

    def on_move(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        pass

    def on_release(self, doc: Document, x: int, y: int) -> None:
        pass
