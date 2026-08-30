"""Eraser tool — removes pixel data by reducing alpha on the active layer.

On MASK layers, erasing paints towards black (fully hidden) instead of
reducing alpha, matching the Affinity / Photoshop convention.

Uses the active brush preset's tip image (from BrushManager) when available;
falls back to a simple circular dab otherwise.
"""

import numpy as np

from .tool_base import Tool
from ..core.document import Document
from ..core.enums import LayerType


class EraserTool(Tool):
    """Round eraser that reduces alpha (opacity) of existing pixels."""

    def __init__(self) -> None:
        super().__init__("Eraser")
        self.size: int = 20
        self.hardness: float = 0.8
        self.opacity: float = 1.0
        self.spacing: float = 0.25
        self._last_x: int = 0
        self._last_y: int = 0
        self._drawing: bool = False

    # ------------------------------------------------------------------
    # Active tip helper
    # ------------------------------------------------------------------

    def _get_active_tip(self) -> np.ndarray | None:
        """Return the tip image of the active brush preset, or None."""
        try:
            from ..core.brush_engine import BrushManager
            mgr = BrushManager.instance()
            preset = mgr.active_preset
            if preset is not None and preset.tip_image is not None:
                return preset.tip_image
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def generate_preview_dab(self) -> np.ndarray | None:
        """Return an alpha-mask dab showing eraser intensity."""
        tip = self._get_active_tip()
        if tip is not None:
            return self._preview_from_tip(tip)
        return self._preview_circle()

    def _preview_circle(self) -> np.ndarray:
        d = max(self.size, 1)
        r = d / 2.0
        center = r - 0.5
        dab = np.zeros((d, d, 4), dtype=np.uint8)
        yy, xx = np.mgrid[0:d, 0:d]
        dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2).astype(np.float32)
        mask = np.clip(1.0 - dist / max(r, 1), 0, 1)
        mask = mask ** (1.0 / max(self.hardness, 0.01))
        mask *= self.opacity
        dab[..., 0] = 255
        dab[..., 1] = 255
        dab[..., 2] = 255
        dab[..., 3] = np.clip(mask * 255, 0, 255).astype(np.uint8)
        return dab

    def _preview_from_tip(self, tip: np.ndarray) -> np.ndarray:
        """Build a preview dab using the active tip image."""
        import cv2
        th, tw = tip.shape[:2]
        d = max(self.size, 1)
        scale = d / max(th, tw, 1)
        new_h = max(1, int(th * scale))
        new_w = max(1, int(tw * scale))
        if new_h != th or new_w != tw:
            scaled = cv2.resize(tip, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            scaled = tip
        dab = np.zeros((new_h, new_w, 4), dtype=np.uint8)
        dab[..., 0] = 255
        dab[..., 1] = 255
        dab[..., 2] = 255
        dab[..., 3] = np.clip(scaled.astype(np.float32) * self.opacity / 255.0 * 255, 0, 255).astype(np.uint8)
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

    def _erase_circle(self, target: np.ndarray, cx: int, cy: int,
                      radius: int, hardness: float, opacity: float,
                      sel_mask: np.ndarray | None = None) -> None:
        """Reduce alpha in a circular region."""
        h, w = target.shape[:2]
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        if y1 <= y0 or x1 <= x0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
        mask = np.clip(1.0 - dist / max(radius, 1), 0, 1)
        mask = mask ** (1.0 / max(hardness, 0.01))
        mask *= opacity
        # Clip to selection
        if sel_mask is not None:
            mask *= sel_mask[y0:y1, x0:x1]
        # Reduce the alpha channel
        target[y0:y1, x0:x1, 3] *= (1.0 - mask)
        # Also fade RGB towards transparent to avoid colour fringing
        for c in range(3):
            target[y0:y1, x0:x1, c] *= (1.0 - mask)
        np.clip(target[y0:y1, x0:x1], 0, 1, out=target[y0:y1, x0:x1])

    def _erase_along(self, doc: Document, x0: int, y0: int, x1: int, y1: int,
                     pressure: float) -> None:
        layer = doc.layers.active_layer
        if layer is None or layer.locked:
            return
        layer.begin_write()
        lx, ly = layer.position
        radius = self._effective_radius(pressure)
        eff_opacity = self.opacity * pressure
        step = max(1.0, radius * 2 * self.spacing)
        sel_mask = self._get_sel_mask(doc)

        # Check for active brush tip
        tip = self._get_active_tip()

        if layer.layer_type == LayerType.MASK:
            # On mask layers, erase = paint black (RGB → 0, keep alpha=1)
            black = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            if tip is not None:
                tip_size = max(1, int(self.size * pressure))
                for px, py in self._stroke_points(x0, y0, x1, y1, step):
                    self._stamp_tip(layer.pixels, px - lx, py - ly,
                                    tip, tip_size, black,
                                    eff_opacity, hardness=self.hardness,
                                    sel_mask=sel_mask)
            else:
                for px, py in self._stroke_points(x0, y0, x1, y1, step):
                    self._stamp_circle(layer.pixels, px - lx, py - ly, radius,
                                       black, self.hardness, eff_opacity,
                                       sel_mask=sel_mask)
        else:
            if tip is not None:
                tip_size = max(1, int(self.size * pressure))
                for px, py in self._stroke_points(x0, y0, x1, y1, step):
                    self._erase_tip(layer.pixels, px - lx, py - ly,
                                    tip, tip_size, eff_opacity,
                                    sel_mask=sel_mask)
            else:
                for px, py in self._stroke_points(x0, y0, x1, y1, step):
                    self._erase_circle(layer.pixels, px - lx, py - ly, radius,
                                       self.hardness, eff_opacity, sel_mask=sel_mask)

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def on_press(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        self._rasterize_if_needed(doc)
        doc.save_snapshot("Eraser Stroke")
        self._drawing = True
        self._last_x, self._last_y = x, y
        self._erase_along(doc, x, y, x, y, pressure)

    def on_move(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        if not self._drawing:
            return
        self._erase_along(doc, self._last_x, self._last_y, x, y, pressure)
        self._last_x, self._last_y = x, y

    def on_release(self, doc: Document, x: int, y: int) -> None:
        self._drawing = False
