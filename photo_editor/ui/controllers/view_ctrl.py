"""View state — zoom, pan, rulers, guides, grid."""

from __future__ import annotations

from PySide6.QtCore import QPointF

from .base import ControllerBase
from ..services.guide_ui_state import apply_guides, apply_preview_guide
from ..widgets.rulers import RULER_SIZE
from ...core.enums import LayerType


class ViewController(ControllerBase):
    """Handles zoom, pan, view toggles (grid/rulers/guides), and guide management."""

    def __init__(self) -> None:
        super().__init__()

    def wire(self, main_window) -> None:
        """Connect to main window and wire menu/panel/canvas signals."""
        super().wire(main_window)
        mw = main_window

        # Menu: zoom and view toggles
        a = mw._menu.actions_map
        a["zoom_in"].triggered.connect(lambda: self.on_zoom(1.25))
        a["zoom_out"].triggered.connect(lambda: self.on_zoom(1 / 1.25))
        a["zoom_fit"].triggered.connect(self.on_zoom_fit)
        a["zoom_100"].triggered.connect(self.on_zoom_100)
        a["toggle_grid"].triggered.connect(self.on_toggle_grid)
        a["toggle_rulers"].triggered.connect(self.on_toggle_rulers)
        a["toggle_guides"].triggered.connect(self.on_toggle_guides)
        if "toggle_snapping" in a:
            a["toggle_snapping"].setCheckable(True)
            a["toggle_snapping"].setChecked(True)
            a["toggle_snapping"].triggered.connect(self.on_toggle_snapping)
        
        from ..theme import THEMES
        for key in THEMES.keys():
            action_key = f"theme_{key.lower().replace(' ', '_')}"
            if action_key in a:
                a[action_key].triggered.connect(lambda _, k=key: self.on_set_theme(k))

        # Props panel zoom bar
        mw._props_panel.zoom_action.connect(self.on_zoom_action)

        # Ruler guide signals (rulers created in _build_ui)
        for ruler in (mw._h_ruler, mw._v_ruler):
            ruler.guide_created.connect(self.on_guide_created)
            ruler.guide_moved.connect(self.on_guide_moved)
            ruler.guide_deleted.connect(self.on_guide_deleted)

        # Canvas guide drag
        mw._canvas.guide_drag_moved.connect(self.on_canvas_guide_drag_moved)
        mw._canvas.guide_drag_released.connect(self.on_canvas_guide_drag_released)

    def on_zoom(self, factor: float) -> None:
        mw = self.mw
        mw._canvas.zoom_by(factor)
        mw._status.set_zoom(mw._canvas.zoom)
        self.update_rulers()

    def on_zoom_fit(self) -> None:
        mw = self.mw
        self.ctx.zoom_to_fit()
        mw._status.set_zoom(mw._canvas.zoom)

    def on_zoom_100(self) -> None:
        mw = self.mw
        mw._canvas.set_zoom(1.0)
        mw._status.set_zoom(1.0)

    def on_zoom_action(self, action: str) -> None:
        """Handle zoom actions from the zoom properties bar."""
        mw = self.mw
        canvas = mw._canvas
        if action == "zoom_in":
            canvas.zoom_by(1.5)
        elif action == "zoom_out":
            canvas.zoom_by(1 / 1.5)
        elif action == "fit":
            canvas.zoom_to_fit()
        elif action == "reset":
            canvas.set_zoom(1.0)
            canvas.set_pan(QPointF(0, 0))
        mw._status.set_zoom(canvas.zoom)

    def on_zoom_tool(self, factor: float, doc_pos: tuple[int, int] | None = None) -> None:
        """Called when the zoom tool requests a zoom change."""
        mw = self.mw
        anchor = None
        if doc_pos is not None and mw._canvas.zoom_to_mouse:
            dr = mw._canvas._doc_rect()
            anchor = mw._canvas._doc_to_widget(dr, float(doc_pos[0]), float(doc_pos[1]))
        mw._canvas.zoom_by(factor, anchor=anchor)
        mw._status.set_zoom(mw._canvas.zoom)
        self.update_rulers()

    def on_pan_tool(self, dx_screen: float, dy_screen: float) -> None:
        """Called when the pan tool requests a pan delta (screen pixels)."""
        canvas = self.mw._canvas
        canvas._pan += QPointF(dx_screen, dy_screen)
        canvas.update()

    def on_toggle_grid(self) -> None:
        mw = self.mw
        mw._show_grid = not getattr(mw, "_show_grid", False)
        state = "on" if mw._show_grid else "off"
        self.ctx.show_status_message(f"Grid {state} (not yet implemented)", 2000)

    def on_toggle_rulers(self) -> None:
        mw = self.mw
        mw._rulers_visible = not mw._rulers_visible
        mw._ruler_corner.setVisible(mw._rulers_visible)
        mw._h_ruler.setVisible(mw._rulers_visible)
        mw._v_ruler.setVisible(mw._rulers_visible)
        state = "on" if mw._rulers_visible else "off"
        self.ctx.show_status_message(f"Rulers {state}", 2000)

    def on_toggle_snapping(self) -> None:
        """Turn smart snapping on or off for the session.

        Snapping can also be suspended for a single drag by holding Ctrl
        (Cmd on macOS), which is the usual way to place something just
        beside an alignment rather than on it.
        """
        mw = self.mw
        mw._snap_enabled = not getattr(mw, "_snap_enabled", True)
        action = mw._menu.actions_map.get("toggle_snapping")
        if action is not None:
            action.setChecked(mw._snap_enabled)
        mw._canvas.set_snap_lines([])
        state = "on" if mw._snap_enabled else "off"
        mw._status.show_activity(f"Snap to objects {state}", 1500)

    def on_toggle_guides(self) -> None:
        mw = self.mw
        mw._show_guides = not getattr(mw, "_show_guides", True)
        apply_guides(mw._canvas, mw._h_ruler, mw._v_ruler, mw._guides if mw._show_guides else [])
        state = "on" if mw._show_guides else "off"
        self.ctx.show_status_message(f"Guides {state}", 2000)

    def on_set_theme(self, theme_name: str) -> None:
        mw = self.mw
        from ..theme import ThemeManager, THEMES
        if theme_name in THEMES:
            ThemeManager.instance().set_theme(theme_name)
            self.ctx.show_status_message(f"Theme set to {theme_name}", 2000)

    def on_guide_created(self, guide) -> None:
        mw = self.mw
        mw._guides.append(guide)
        apply_preview_guide(mw._canvas, mw._h_ruler, mw._v_ruler, mw._guides, None)

    def on_guide_moved(self, guide, new_pos: float) -> None:
        mw = self.mw
        guide.position = new_pos
        if guide not in mw._guides:
            apply_preview_guide(mw._canvas, mw._h_ruler, mw._v_ruler, mw._guides, guide)
        else:
            apply_guides(mw._canvas, mw._h_ruler, mw._v_ruler, mw._guides)

    def on_guide_deleted(self, guide) -> None:
        mw = self.mw
        if guide in mw._guides:
            mw._guides.remove(guide)
        apply_preview_guide(mw._canvas, mw._h_ruler, mw._v_ruler, mw._guides, None)

    def on_canvas_guide_drag_moved(self, guide, new_pos: float) -> None:
        mw = self.mw
        guide.position = new_pos
        apply_guides(mw._canvas, mw._h_ruler, mw._v_ruler, mw._guides)

    def on_canvas_guide_drag_released(self, guide, pos: float, delete: bool) -> None:
        mw = self.mw
        if delete:
            if guide in mw._guides:
                mw._guides.remove(guide)
        else:
            guide.position = pos
        apply_guides(mw._canvas, mw._h_ruler, mw._v_ruler, mw._guides)

    def update_rulers(self) -> None:
        """Sync rulers with current canvas zoom/pan state and document unit."""
        mw = self.mw
        if not hasattr(mw, '_h_ruler') or not mw._rulers_visible:
            return
        dr = mw._canvas._doc_rect()
        dw = mw._canvas._doc_w or 1
        dh = mw._canvas._doc_h or 1

        h_zoom = dr.width() / dw
        h_origin = dr.left()
        mw._h_ruler.set_view_params(h_zoom, h_origin, dw)

        v_zoom = dr.height() / dh
        v_origin = dr.top()
        mw._v_ruler.set_view_params(v_zoom, v_origin, dh)

        mw._h_ruler.set_perp_view_params(v_zoom, v_origin + RULER_SIZE, dh)
        mw._v_ruler.set_perp_view_params(h_zoom, h_origin + RULER_SIZE, dw)

        # Pass the active document's unit and DPI so rulers show the right unit
        if mw._doc is not None:
            unit = getattr(mw._doc, "unit", "px")
            dpi = getattr(mw._doc, "dpi", 72)
            mw._h_ruler.set_unit(unit, dpi)
            mw._v_ruler.set_unit(unit, dpi)

        layer = mw._doc.layers.active_layer if mw._doc else None
        if layer and layer.layer_type not in (LayerType.GROUP, LayerType.MASK):
            lx, ly = layer.position
            # layer.width/height, not layer.pixels.shape: the pixels property
            # triggers a full non-destructive-transform recompute when the
            # layer is dirty, and this runs on the view-change path.
            lw, lh = layer.width, layer.height
            mw._h_ruler.set_layer_bounds(float(lx), float(lx + lw))
            mw._v_ruler.set_layer_bounds(float(ly), float(ly + lh))
        else:
            mw._h_ruler.set_layer_bounds(None, None)
            mw._v_ruler.set_layer_bounds(None, None)

        apply_guides(mw._canvas, mw._h_ruler, mw._v_ruler, mw._guides)
