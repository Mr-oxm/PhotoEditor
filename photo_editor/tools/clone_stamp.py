"""Clone Stamp tool — copies pixels from a source region to the destination."""

import numpy as np

from .tool_base import Tool
from ..core.document import Document


class CloneStampTool(Tool):
    """Samples pixels from one area and paints them onto another."""

    def __init__(self) -> None:
        super().__init__("Clone Stamp")
        self.size: int = 30
        self.hardness: float = 0.7
        self.opacity: float = 1.0
        self.spacing: float = 0.25

        # Source configuration (set via alt-click)
        self.source_x: int = 0
        self.source_y: int = 0
        self.source_set: bool = False

        self._offset_x: int = 0
        self._offset_y: int = 0
        self._last_x: int = 0
        self._last_y: int = 0
        self._drawing: bool = False
        self._offset_locked: bool = False

    # ------------------------------------------------------------------
    # Public: set source point (called externally on alt-click)
    # ------------------------------------------------------------------

    def set_source(self, x: int, y: int) -> None:
        """Call this when the user alt-clicks to set the clone source."""
        self.source_x = x
        self.source_y = y
        self.source_set = True
        self._offset_locked = False

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def generate_preview_dab(self) -> np.ndarray | None:
        """Return an RGBA uint8 dab showing the clone stamp brush circle."""
        d = max(self.size, 1)
        r = d / 2.0
        center = r - 0.5
        dab = np.zeros((d, d, 4), dtype=np.uint8)
        yy, xx = np.mgrid[0:d, 0:d]
        dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2).astype(np.float32)
        mask = np.clip(1.0 - dist / max(r, 1), 0, 1)
        mask = mask ** (1.0 / max(self.hardness, 0.01))
        mask *= self.opacity
        # Neutral grey preview so the cursor is visible on any background
        dab[..., 0] = 128
        dab[..., 1] = 128
        dab[..., 2] = 128
        dab[..., 3] = np.clip(mask * 100, 0, 255).astype(np.uint8)
        return dab

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_radius(self, pressure: float) -> int:
        return max(1, int((self.size / 2) * pressure))

    def _stroke_points(self, x0: int, y0: int, x1: int, y1: int, step: float):
        dx = x1 - x0
        dy = y1 - y0
        dist = max(1.0, np.hypot(dx, dy))
        steps = int(dist / max(step, 1))
        for i in range(steps + 1):
            t = i / max(steps, 1)
            yield int(x0 + dx * t), int(y0 + dy * t)

    def _clone_circle(self, target: np.ndarray, cx: int, cy: int,
                      radius: int, sel_mask: np.ndarray | None = None) -> None:
        """Stamp source pixels onto *target* centred at (cx, cy)."""
        h, w = target.shape[:2]
        sx = cx + self._offset_x
        sy = cy + self._offset_y

        y0d, y1d = max(0, cy - radius), min(h, cy + radius + 1)
        x0d, x1d = max(0, cx - radius), min(w, cx + radius + 1)
        y0s = y0d + (sy - cy)
        x0s = x0d + (sx - cx)
        y1s = y1d + (sy - cy)
        x1s = x1d + (sx - cx)

        # Clip source region to valid bounds
        if y0s < 0:
            y0d -= y0s; y0s = 0
        if x0s < 0:
            x0d -= x0s; x0s = 0
        if y1s > h:
            y1d -= (y1s - h); y1s = h
        if x1s > w:
            x1d -= (x1s - w); x1s = w
        if y1d <= y0d or x1d <= x0d or y1s <= y0s or x1s <= x0s:
            return

        # Ensure same shape
        sh = min(y1d - y0d, y1s - y0s)
        sw = min(x1d - x0d, x1s - x0s)
        if sh <= 0 or sw <= 0:
            return

        src_patch = target[y0s:y0s + sh, x0s:x0s + sw].copy()

        yy, xx = np.mgrid[y0d:y0d + sh, x0d:x0d + sw]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
        mask = np.clip(1.0 - dist / max(radius, 1), 0, 1)
        mask = mask ** (1.0 / max(self.hardness, 0.01))
        mask *= self.opacity
        # Clip to selection
        if sel_mask is not None:
            mask *= sel_mask[y0d:y0d + sh, x0d:x0d + sw]
        mask = mask[..., np.newaxis]

        target[y0d:y0d + sh, x0d:x0d + sw] = (
            target[y0d:y0d + sh, x0d:x0d + sw] * (1 - mask) + src_patch * mask
        )

    def _stamp_along(self, doc: Document, x0: int, y0: int, x1: int, y1: int,
                     pressure: float) -> None:
        layer = doc.layers.active_layer
        if layer is None or layer.locked:
            return
        lx, ly = layer.position
        radius = self._effective_radius(pressure)
        step = max(1.0, radius * 2 * self.spacing)
        sel_mask = self._get_sel_mask(doc)
        for px, py in self._stroke_points(x0, y0, x1, y1, step):
            self._clone_circle(layer.pixels, px - lx, py - ly, radius,
                               sel_mask=sel_mask)

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------


    def on_press(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        if not self.source_set:
            return
        self._rasterize_if_needed(doc)
        doc.save_snapshot("Clone Stamp")
        if not self._offset_locked:
            self._offset_x = self.source_x - x
            self._offset_y = self.source_y - y
            self._offset_locked = True
        self._drawing = True
        self._last_x, self._last_y = x, y
        self._stamp_along(doc, x, y, x, y, pressure)

    def on_move(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        if not self._drawing:
            return
        self._stamp_along(doc, self._last_x, self._last_y, x, y, pressure)
        self._last_x, self._last_y = x, y

    def on_release(self, doc: Document, x: int, y: int) -> None:
        self._drawing = False
