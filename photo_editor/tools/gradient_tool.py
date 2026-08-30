"""Gradient tool — real-time preview, interactive Affinity-style handles.

Draw a gradient by clicking and dragging.  While dragging the gradient
is rendered live on the canvas.  On release a control line with colour
stop circles appears; the endpoints can then be dragged to reshape the
gradient interactively.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .tool_base import Tool
from ..core.color import Color, GradientStop
from ..core.document import Document


class GradientTool(Tool):
    """Draws a gradient with real-time preview and interactive handles."""

    HANDLE_RADIUS: int = 14  # document-pixel hit radius

    def __init__(self) -> None:
        super().__init__("Gradient")

        # ---- configurable properties (synced from properties bar) -----------
        self.gradient_type: str = "linear"   # linear | radial | conical | diamond
        self.opacity: float = 1.0

        self._stops: list[GradientStop] = [
            GradientStop(0.0, Color.black()),
            GradientStop(1.0, Color.white()),
        ]

        # ---- drag state -----------------------------------------------------
        self._start_x: int = 0
        self._start_y: int = 0
        self._end_x: int = 0
        self._end_y: int = 0
        self._dragging: bool = False

        # ---- handle-editing state -------------------------------------------
        self._editing: bool = False
        self._dragging_handle: str | None = None   # "start" / "end" / None
        self._saved_pixels: np.ndarray | None = None
        self._target_layer = None
        self._layer_pos: tuple[int, int] = (0, 0)

        # ---- callbacks (set by main_window) ---------------------------------
        self._preview_cb: Callable[[], None] | None = None
        self._handles_cb: Callable | None = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def stops(self) -> list[GradientStop]:
        return list(self._stops)

    @stops.setter
    def stops(self, value: list[GradientStop]) -> None:
        self._stops = sorted(value, key=lambda s: s.position)
        if self._editing:
            self._reapply_gradient()

    @property
    def is_editing(self) -> bool:
        return self._editing

    def set_preview_callback(self, cb: Callable[[], None] | None) -> None:
        self._preview_cb = cb

    def set_handles_callback(self, cb: Callable | None) -> None:
        self._handles_cb = cb

    def reverse_gradient(self) -> None:
        """Reverse the gradient stop order."""
        self._stops = [
            GradientStop(1.0 - s.position, s.color)
            for s in reversed(self._stops)
        ]
        if self._editing:
            self._reapply_gradient()

    # ------------------------------------------------------------------
    # Gradient generators
    # ------------------------------------------------------------------

    @staticmethod
    def _linear_map(h: int, w: int, x0: int, y0: int,
                    x1: int, y1: int) -> np.ndarray:
        dx, dy = float(x1 - x0), float(y1 - y0)
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return np.zeros((h, w), dtype=np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        t = ((xx - x0) * dx + (yy - y0) * dy) / length_sq
        return np.clip(t, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _radial_map(h: int, w: int, cx: int, cy: int,
                    ex: int, ey: int) -> np.ndarray:
        radius = max(1.0, np.hypot(float(ex - cx), float(ey - cy)))
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
        return np.clip(dist / radius, 0.0, 1.0)

    @staticmethod
    def _conical_map(h: int, w: int, cx: int, cy: int,
                     ex: int, ey: int) -> np.ndarray:
        base = math.atan2(ey - cy, ex - cx)
        yy, xx = np.mgrid[0:h, 0:w]
        angles = np.arctan2(yy - cy, xx - cx).astype(np.float32)
        t = (angles - base) / (2.0 * math.pi)
        return np.mod(t, 1.0).astype(np.float32)

    @staticmethod
    def _diamond_map(h: int, w: int, cx: int, cy: int,
                     ex: int, ey: int) -> np.ndarray:
        radius = max(1.0, float(abs(ex - cx) + abs(ey - cy)))
        yy, xx = np.mgrid[0:h, 0:w]
        dist = (np.abs(xx - cx) + np.abs(yy - cy)).astype(np.float32)
        return np.clip(dist / radius, 0.0, 1.0)

    def _render_gradient(self, h: int, w: int,
                         x0: int, y0: int,
                         x1: int, y1: int,
                         *, downsample: bool = False) -> np.ndarray:
        """Return [H, W, 4] float32 gradient image using current stops.

        When *downsample* is True (used during interactive drag), the
        gradient map is computed at a reduced resolution then upscaled.
        This is dramatically cheaper for large layers.
        """
        gtype = self.gradient_type

        # Compute at reduced size during drag for large canvases
        rh, rw = h, w
        if downsample and (h > 512 or w > 512):
            scale = max(h, w) / 512.0
            rh = max(1, int(h / scale))
            rw = max(1, int(w / scale))
            rx0, ry0 = int(x0 / scale), int(y0 / scale)
            rx1, ry1 = int(x1 / scale), int(y1 / scale)
        else:
            rx0, ry0, rx1, ry1 = x0, y0, x1, y1

        if gtype == "radial":
            t = self._radial_map(rh, rw, rx0, ry0, rx1, ry1)
        elif gtype == "conical":
            t = self._conical_map(rh, rw, rx0, ry0, rx1, ry1)
        elif gtype == "diamond":
            t = self._diamond_map(rh, rw, rx0, ry0, rx1, ry1)
        else:
            t = self._linear_map(rh, rw, rx0, ry0, rx1, ry1)

        positions = np.array([s.position for s in self._stops], dtype=np.float32)
        colours = np.array(
            [[s.color.r, s.color.g, s.color.b, s.color.a] for s in self._stops],
            dtype=np.float32,
        )
        gradient = np.empty((rh, rw, 4), dtype=np.float32)
        t_flat = t.ravel()
        for ch in range(4):
            gradient[..., ch] = np.interp(t_flat, positions, colours[:, ch]).reshape(rh, rw)

        # Upscale back to full resolution if we downsampled
        if downsample and (rh != h or rw != w):
            import cv2
            gradient = cv2.resize(gradient, (w, h), interpolation=cv2.INTER_LINEAR)

        return gradient

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_gradient(self, base: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Blend *gradient* onto *base* with current opacity."""
        if self.opacity < 1.0:
            result = base * (1.0 - self.opacity) + gradient * self.opacity
        else:
            result = gradient.copy()
        np.clip(result, 0.0, 1.0, out=result)
        return result

    def _reapply_gradient(self) -> None:
        """Re-render gradient from saved pixels (handle-editing path)."""
        if self._saved_pixels is None or self._target_layer is None:
            return
        layer = self._target_layer
        if layer.locked:
            return
        lx, ly = self._layer_pos
        h, w = self._saved_pixels.shape[:2]
        grad = self._render_gradient(
            h, w,
            self._start_x - lx, self._start_y - ly,
            self._end_x - lx, self._end_y - ly,
        )
        layer.pixels[:] = self._apply_gradient(self._saved_pixels, grad)
        layer.touch()
        self._emit_handles(True)
        if self._preview_cb:
            self._preview_cb()

    def _emit_handles(self, visible: bool) -> None:
        if self._handles_cb:
            self._handles_cb(
                (self._start_x, self._start_y) if visible else None,
                (self._end_x, self._end_y) if visible else None,
                self._stops if visible else [],
                visible,
            )

    def _hit_handle(self, x: int, y: int) -> str | None:
        r2 = self.HANDLE_RADIUS ** 2
        if (x - self._start_x) ** 2 + (y - self._start_y) ** 2 <= r2:
            return "start"
        if (x - self._end_x) ** 2 + (y - self._end_y) ** 2 <= r2:
            return "end"
        return None

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def on_press(self, doc: Document, x: int, y: int,
                 pressure: float = 1.0) -> None:
        layer = doc.layers.active_layer
        if layer is None or layer.locked:
            return
        self._rasterize_if_needed(doc)

        # If already editing, check for handle grab first
        if self._editing:
            hit = self._hit_handle(x, y)
            if hit:
                self._dragging_handle = hit
                return
            # Click away from handles → start a new gradient
            self._editing = False
            self._saved_pixels = None
            self._target_layer = None
            self._emit_handles(False)

        # Begin new gradient drag
        doc.save_snapshot("Gradient")
        self._start_x, self._start_y = x, y
        self._end_x, self._end_y = x, y
        self._dragging = True
        self._target_layer = layer
        self._layer_pos = layer.position
        self._saved_pixels = layer.pixels.copy()

    def on_move(self, doc: Document, x: int, y: int,
                pressure: float = 1.0) -> None:
        # Handle drag
        if self._dragging_handle:
            if self._dragging_handle == "start":
                self._start_x, self._start_y = x, y
            else:
                self._end_x, self._end_y = x, y
            self._reapply_gradient()
            return

        if not self._dragging:
            return

        self._end_x, self._end_y = x, y

        # Live preview — render gradient on layer from saved state
        layer = self._target_layer
        if layer is None or self._saved_pixels is None:
            return
        lx, ly = self._layer_pos
        h, w = self._saved_pixels.shape[:2]
        grad = self._render_gradient(
            h, w,
            self._start_x - lx, self._start_y - ly,
            x - lx, y - ly,
            downsample=True,
        )
        layer.pixels[:] = self._apply_gradient(self._saved_pixels, grad)
        layer.touch()
        self._emit_handles(True)
        if self._preview_cb:
            self._preview_cb()

    def on_release(self, doc: Document, x: int, y: int) -> None:
        if self._dragging_handle:
            self._dragging_handle = None
            return

        if not self._dragging:
            return
        self._dragging = False
        self._end_x, self._end_y = x, y

        layer = self._target_layer
        if layer is None or self._saved_pixels is None:
            return

        # No actual drag — cancel
        if self._start_x == x and self._start_y == y:
            layer.pixels[:] = self._saved_pixels
            layer.touch()
            self._saved_pixels = None
            self._target_layer = None
            self._emit_handles(False)
            return

        # Final render
        lx, ly = self._layer_pos
        h, w = self._saved_pixels.shape[:2]
        grad = self._render_gradient(
            h, w,
            self._start_x - lx, self._start_y - ly,
            x - lx, y - ly,
        )
        layer.pixels[:] = self._apply_gradient(self._saved_pixels, grad)
        layer.touch()

        # Keep saved state for handle editing
        self._editing = True
        self._emit_handles(True)

    def deactivate(self) -> None:
        """Hide handles when switching away, but preserve editing state."""
        self._dragging = False
        self._dragging_handle = None
        # Only hide the overlay handles; keep _editing, _saved_pixels,
        # _target_layer, and start/end coords so they can be restored.
        self._emit_handles(False)
        super().deactivate()
