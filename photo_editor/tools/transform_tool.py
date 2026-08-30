"""Transform tool — interactive drag-based scaling and rotation via TransformEngine.

Uses the Layer's non-destructive transform system: display pixels are
always re-derived from the original source, so repeated transforms
never degrade quality.
"""

import math

import numpy as np

from .tool_base import Tool
from ..core.document import Document
from ..transforms.transform_engine import TransformEngine


class TransformTool(Tool):
    """Drag to scale; Shift-drag (far from centre) to rotate the active layer."""

    def __init__(self) -> None:
        super().__init__("Transform")
        self.mode: str = "scale"  # "scale" | "rotate" | "free"
        self._engine = TransformEngine()

        self._start_x: int = 0
        self._start_y: int = 0
        self._dragging: bool = False
        self._center_x: float = 0
        self._center_y: float = 0
        # Saved base params at drag start
        self._base_scale_x: float = 1.0
        self._base_scale_y: float = 1.0
        self._base_angle: float = 0.0

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def on_press(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        layer = doc.layers.active_layer
        if layer is None or layer.locked:
            return
        doc.save_snapshot("Transform")

        # Initialise non-destructive source (idempotent)
        layer.init_non_destructive()

        self._start_x, self._start_y = x, y
        h, w = layer.pixels.shape[:2]
        lx, ly = layer.position
        self._center_x = lx + w / 2.0
        self._center_y = ly + h / 2.0
        self._base_scale_x = layer.transform_scale_x
        self._base_scale_y = layer.transform_scale_y
        self._base_angle = layer.transform_angle
        self._dragging = True

    def on_move(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        if not self._dragging:
            return
        layer = doc.layers.active_layer
        if layer is None or layer._source_pixels is None:
            return

        if self.mode == "rotate":
            a0 = math.atan2(self._start_y - self._center_y,
                            self._start_x - self._center_x)
            a1 = math.atan2(y - self._center_y, x - self._center_x)
            delta = math.degrees(a1 - a0)
            layer.transform_angle = self._base_angle + delta
            layer.compute_display(fast=True)
            layer.position = (int(self._center_x - layer.width / 2),
                              int(self._center_y - layer.height / 2))

        elif self.mode == "free":
            a0 = math.atan2(self._start_y - self._center_y,
                            self._start_x - self._center_x)
            a1 = math.atan2(y - self._center_y, x - self._center_x)
            delta = math.degrees(a1 - a0)
            d0 = max(1.0, math.hypot(self._start_x - self._center_x,
                                     self._start_y - self._center_y))
            d1 = max(1.0, math.hypot(x - self._center_x, y - self._center_y))
            s = d1 / d0
            layer.transform_scale_x = self._base_scale_x * s
            layer.transform_scale_y = self._base_scale_y * s
            layer.transform_angle = self._base_angle + delta
            layer.compute_display(fast=True)
            layer.position = (int(self._center_x - layer.width / 2),
                              int(self._center_y - layer.height / 2))

        else:  # scale
            dx = x - self._start_x
            dy = y - self._start_y
            src_w = layer.source_width
            src_h = layer.source_height
            sx = max(0.05, self._base_scale_x + dx / max(src_w, 1))
            sy = max(0.05, self._base_scale_y + dy / max(src_h, 1))
            layer.transform_scale_x = sx
            layer.transform_scale_y = sy
            layer.compute_display(fast=True)
            layer.position = (int(self._center_x - layer.width / 2),
                              int(self._center_y - layer.height / 2))

    def on_release(self, doc: Document, x: int, y: int) -> None:
        if self._dragging:
            self._dragging = False
            # The drag used nearest-neighbour sampling to stay interactive
            # (293 ms/event at 4K with the quality path). Re-derive the
            # layer at full quality now that the gesture has ended.
            layer = doc.layers.active_layer
            if layer is not None and layer._source_pixels is not None:
                layer.compute_display(fast=False)
                layer.position = (int(self._center_x - layer.width / 2),
                                  int(self._center_y - layer.height / 2))
