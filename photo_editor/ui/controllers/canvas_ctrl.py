"""Canvas input — press, move, release, hover, double-click."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtWidgets import QApplication

from ...core.enums import ToolType
from .base import ControllerBase
from ..services.rasterize_guard import ensure_active_layer_rasterized_for_tool


class CanvasController(ControllerBase):
    """Handles canvas mouse input: press, move, release, hover, double-click."""

    SEL_TOOLS = {
        ToolType.RECT_SELECT, ToolType.ELLIPSE_SELECT,
        ToolType.LASSO, ToolType.MAGIC_WAND,
    }

    def __init__(self) -> None:
        super().__init__()
        self._dragging = False
        self._sel_moving = False
        self._sel_move_start: tuple[int, int] = (0, 0)
        self._sel_move_orig_mask: object = None
        self._sel_move_total_dx: int = 0
        self._sel_move_total_dy: int = 0
        self._drag_start: tuple[int, int] | None = None

    def wire(self, main_window) -> None:
        super().wire(main_window)
        mw = main_window
        mw._canvas.cursor_moved.connect(self.on_hover)
        mw._canvas.tool_pressed.connect(self.on_press)
        mw._canvas.tool_moved.connect(self.on_move)
        mw._canvas.tool_released.connect(self.on_release)
        mw._canvas.tool_double_clicked.connect(self.on_double_click)

    def on_hover(self, x: int, y: int) -> None:
        mw = self._mw
        if hasattr(mw, '_h_ruler') and mw._rulers_visible:
            dr = mw._canvas._doc_rect()
            if mw._canvas._doc_w > 0 and mw._canvas._doc_h > 0:
                wx = dr.left() + (x / mw._canvas._doc_w) * dr.width()
                wy = dr.top() + (y / mw._canvas._doc_h) * dr.height()
                mw._h_ruler.set_cursor_position(wx)
                mw._v_ruler.set_cursor_position(wy)
        if mw._tools.active_type == ToolType.TEXT:
            self.signals.text_hover_cursor_requested.emit(x, y)
        elif mw._tools.active_type == ToolType.NODE:
            tool = mw._tools.active_tool
            if tool is not None and hasattr(tool, "pick_segments") and tool.pick_segments.active:
                tool.pick_segments_hover(x, y)
                mw._canvas.update()
            elif tool is not None and hasattr(tool, "update_hover"):
                doc = mw._doc
                if doc is not None and tool.update_hover(doc, x, y):
                    mw._canvas.update()
        elif mw._tools.active_type in (ToolType.CLONE_STAMP, ToolType.HEALING_BRUSH):
            tool = mw._tools.active_tool
            if tool is not None and tool.source_set:
                if tool._offset_locked:
                    ox, oy = tool._offset_x, tool._offset_y
                else:
                    ox = tool.source_x - x
                    oy = tool.source_y - y
                mw._canvas.set_source_offset((ox, oy))
            self.signals.clone_preview_requested.emit(x, y)

    def _prime_snapping(self) -> None:
        """Give the Move tool the view state its snapping needs.

        The snap threshold is in screen pixels, so it depends on zoom; the
        guides are view state held by the window. Holding a modifier
        suspends snapping for the duration of a drag, which is the usual
        convention for nudging something into a position that is *near* an
        alignment without taking it.
        """
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt

        mw = self.mw
        tool = mw._tools.active_tool
        engine = getattr(tool, "snap_engine", None)
        if engine is None:
            return
        mods = QApplication.keyboardModifiers()
        # Alt already sets the clone/heal source, so snapping is suspended
        # with Ctrl (Cmd on macOS) -- the usual convention for placing
        # something *near* an alignment without being taken to it.
        suppressed = bool(mods & Qt.KeyboardModifier.ControlModifier)
        engine.enabled = getattr(mw, "_snap_enabled", True) and not suppressed
        tool.snap_zoom = mw._canvas.zoom
        tool.snap_guides = getattr(mw, "_guides", None)

    def _sync_snap_lines(self) -> None:
        """Mirror the Move tool's snap lines onto the canvas overlay."""
        tool = self._mw._tools.active_tool
        self._mw._canvas.set_snap_lines(getattr(tool, "snap_lines", []) or [])

    def on_press(self, x: int, y: int, pressure: float) -> None:
        mw = self._mw
        self._dragging = True
        self._prime_snapping()

        modifiers = QApplication.keyboardModifiers()
        if modifiers & Qt.KeyboardModifier.AltModifier:
            tool_type = mw._tools.active_type
            if tool_type in (ToolType.CLONE_STAMP, ToolType.HEALING_BRUSH):
                tool = mw._tools.active_tool
                if tool is not None:
                    tool.set_source(x, y)
                    mw._canvas.set_source_position((x, y))
                    mw._status.show_activity(f"Source set at ({x}, {y})", 2000)
                self._dragging = False
                return

        if not ensure_active_layer_rasterized_for_tool(
            mw,
            mw._doc,
            mw._tools.active_type,
            self.ctx.refresh,
        ):
            self._dragging = False
            return

        tool_type = mw._tools.active_type
        if tool_type == ToolType.ZOOM:
            tool = mw._tools.active_tool
            if tool is not None:
                if modifiers & Qt.KeyboardModifier.AltModifier:
                    tool.zoom_out()
                else:
                    tool.on_press(mw._doc, x, y, pressure)
            self._dragging = False
            return

        if tool_type in self.SEL_TOOLS:
            tool = mw._tools.active_tool
            if tool is not None and hasattr(tool, "mode"):
                shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
                alt = bool(modifiers & Qt.KeyboardModifier.AltModifier)
                if shift and alt:
                    tool.mode = "intersect"
                elif shift:
                    tool.mode = "add"
                elif alt:
                    tool.mode = "subtract"

                if (tool_type in (ToolType.RECT_SELECT, ToolType.ELLIPSE_SELECT, ToolType.LASSO)
                        and mw._doc and mw._doc.selection._mask is not None
                        and not shift and not alt):
                    mask = mw._doc.selection._mask
                    if (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]
                            and mask[y, x] > 0.5):
                        mw._doc.save_snapshot("Move Selection")
                        self._sel_moving = True
                        self._sel_move_start = (x, y)
                        self._sel_move_orig_mask = mw._doc.selection._mask.copy()
                        self._sel_move_total_dx = 0
                        self._sel_move_total_dy = 0
                        self._dragging = True
                        return

        # Pass shift state to Move tool for multi-select
        if tool_type == ToolType.MOVE:
            tool = mw._tools.active_tool
            if tool is not None and hasattr(tool, "shift_held"):
                tool.shift_held = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

        mw._tools.on_press(mw._doc, x, y, pressure)
        if tool_type in (ToolType.RECT_SELECT, ToolType.ELLIPSE_SELECT):
            self._drag_start = (x, y)
        elif tool_type == ToolType.TEXT:
            self.signals.text_overlay_requested.emit()
        elif tool_type in (ToolType.CLONE_STAMP, ToolType.HEALING_BRUSH):
            tool = mw._tools.active_tool
            if tool is not None and tool.source_set:
                mw._canvas.set_source_offset((tool._offset_x, tool._offset_y))
                mw._canvas.set_source_drawing(True)

    def on_move(self, x: int, y: int, pressure: float) -> None:
        mw = self._mw
        if self._sel_moving and mw._doc and self._sel_move_orig_mask is not None:
            ox, oy = self._sel_move_start
            self._sel_move_total_dx += x - ox
            self._sel_move_total_dy += y - oy
            self._sel_move_start = (x, y)
            orig = self._sel_move_orig_mask
            h, w = orig.shape
            dx, dy = self._sel_move_total_dx, self._sel_move_total_dy
            new_mask = np.zeros_like(orig)
            sx0 = max(0, -dx)
            sy0 = max(0, -dy)
            sx1 = min(w, w - dx)
            sy1 = min(h, h - dy)
            dx0 = max(0, dx)
            dy0 = max(0, dy)
            dx1 = dx0 + (sx1 - sx0)
            dy1 = dy0 + (sy1 - sy0)
            if sx1 > sx0 and sy1 > sy0:
                new_mask[dy0:dy1, dx0:dx1] = orig[sy0:sy1, sx0:sx1]
            mw._doc.selection._set_mask(new_mask)
            self.signals.selection_overlay_requested.emit()
            self.signals.canvas_update_requested.emit()
            return

        tool_type = mw._tools.active_type
        mw._tools.on_move(mw._doc, x, y, pressure)

        if tool_type == ToolType.MOVE:
            # Repaint canvas for marquee overlay and transform box updates
            self.signals.canvas_update_requested.emit()
            self.signals.transform_box_requested.emit()

        if tool_type in (ToolType.RECT_SELECT, ToolType.ELLIPSE_SELECT) and self._drag_start is not None:
            sx, sy = self._drag_start
            dr = mw._canvas._doc_rect()
            zx = dr.left() + (min(sx, x) / mw._canvas._doc_w) * dr.width()
            zy = dr.top() + (min(sy, y) / mw._canvas._doc_h) * dr.height()
            zw = abs(x - sx) / mw._canvas._doc_w * dr.width()
            zh = abs(y - sy) / mw._canvas._doc_h * dr.height()
            is_ellipse = (tool_type == ToolType.ELLIPSE_SELECT)
            mw._canvas.set_drag_rect(QRectF(zx, zy, zw, zh), ellipse=is_ellipse)
        elif tool_type == ToolType.LASSO:
            tool = mw._tools.active_tool
            if tool is not None and hasattr(tool, '_points') and tool._drawing:
                mw._canvas.set_lasso_points(list(tool._points))
        elif tool_type in (ToolType.PEN, ToolType.NODE, ToolType.VECTOR_SHAPE):
            mw._canvas.update()
            if self._dragging:
                self._sync_snap_lines()
            mw._schedule_render()
        elif tool_type == ToolType.TEXT:
            self.signals.text_overlay_requested.emit()
            if not self._dragging:
                self.signals.text_hover_cursor_requested.emit(x, y)
        else:
            mw._schedule_render()

    def on_release(self, x: int, y: int) -> None:
        self._mw._canvas.set_snap_lines([])
        mw = self._mw
        self._dragging = False
        if self._sel_moving:
            self._sel_moving = False
            self._sel_move_orig_mask = None
            self.signals.selection_overlay_requested.emit()
            self.signals.canvas_update_requested.emit()
            self.signals.history_refresh_requested.emit()
            return

        tool_type = mw._tools.active_type
        mw._tools.on_release(mw._doc, x, y)
        mw._canvas.set_drag_rect(None)
        mw._canvas.set_lasso_points(None)
        self._drag_start = None

        if tool_type in (ToolType.CLONE_STAMP, ToolType.HEALING_BRUSH):
            mw._canvas.set_source_drawing(False)

        mw._refresh_canvas_only()
        mw._schedule_panel_refresh()

        if tool_type == ToolType.MOVE:
            self.signals.transform_box_requested.emit()

        if tool_type == ToolType.TEXT:
            self.signals.text_overlay_requested.emit()

        _VEC_TOOLS = {ToolType.PEN: "pen", ToolType.NODE: "node",
                      ToolType.VECTOR_SHAPE: "shape"}
        if tool_type in _VEC_TOOLS:
            self.signals.properties_panel_requested.emit()

    def on_double_click(self, x: int, y: int) -> None:
        mw = self._mw
        tool_type = mw._tools.active_type
        tool = mw._tools.active_tool
        if tool_type == ToolType.PEN and tool is not None:
            if hasattr(tool, 'finish_open_path'):
                tool.finish_open_path(mw._doc)
                mw._refresh()
                return
        if tool_type == ToolType.NODE and tool is not None:
            if hasattr(tool, 'pick_segments') and tool.pick_segments.active:
                return  # suppress node insertion while picking segments
            if hasattr(tool, 'insert_node_on_segment'):
                tool.insert_node_on_segment(mw._doc, x, y)
                mw._refresh()
                return
        # Double-click a selected vector layer with the Move tool → switch to Node tool
        if tool_type == ToolType.MOVE and mw._doc is not None:
            layer = mw._doc.layers.active_layer
            if layer is not None and layer.layer_type.name == "SHAPE":
                vl = getattr(layer, "_vector_data", None)
                if vl is not None:
                    self.signals.tool_selection_requested.emit(ToolType.NODE)
                    return
