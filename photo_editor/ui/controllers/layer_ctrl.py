"""Layer operations — add, delete, group, mask, reorder."""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt, QTimer

from ...commands import (
    AddGroupCommand,
    AddLayerCommand,
    AddMaskLayerCommand,
    ApplyMaskLayerCommand,
    AttachAdjustmentToLayerCommand,
    AttachMaskToLayerCommand,
    ClipToLayerCommand,
    ConvertToMaskCommand,
    DropAsMaskCommand,
    DuplicateLayerCommand,
    FlattenCommand,
    InvertMaskLayerCommand,
    MergeDownCommand,
    MoveLayerCommand,
    RemoveLayerCommand,
    RemoveMaskLayerCommand,
    RenameLayerCommand,
    ReorderLayersCommand,
)
from ...core.services.document_resize import resize_canvas, resize_image
from ...core.enums import BlendMode, LayerType, ToolType
from .base import ControllerBase
from ..dialogs.layer_styles_dialog import LayerStylesDialog
from ..services.layer_panel_state import (
    reordered_stack_order,
    selected_indices_from_layer_ids,
    sync_panel_selection,
)
from ..services.rasterize_guard import rasterize_active_text_layer

NUMPAD_MAP = {
    Qt.Key.Key_0: 0, Qt.Key.Key_1: 1, Qt.Key.Key_2: 2,
    Qt.Key.Key_3: 3, Qt.Key.Key_4: 4, Qt.Key.Key_5: 5,
    Qt.Key.Key_6: 6, Qt.Key.Key_7: 7, Qt.Key.Key_8: 8,
    Qt.Key.Key_9: 9,
}


class LayerController(ControllerBase):
    """Handles layer add/delete/group/mask/reorder and property changes."""

    def __init__(self) -> None:
        super().__init__()
        self._blend_preview_original: BlendMode | None = None
        self._numpad_first: int | None = None
        self._numpad_timer: QTimer | None = None

    def wire(self, main_window) -> None:
        """Connect to main window and wire menu/panel signals."""
        super().wire(main_window)
        mw = main_window

        # Menu
        a = mw._menu.actions_map
        a["new_layer"].triggered.connect(self.on_add_layer)
        a["new_vector_layer"].triggered.connect(self.on_add_vector_layer)
        a["new_group"].triggered.connect(self.on_add_group)
        a["dup_layer"].triggered.connect(self.on_dup_layer)
        a["del_layer"].triggered.connect(self.on_del_layer)
        a["add_mask"].triggered.connect(self.on_add_mask)
        a["add_mask_black"].triggered.connect(self.on_add_mask_black)
        a["add_mask_standalone"].triggered.connect(self.on_add_mask_standalone)
        a["remove_mask_layer"].triggered.connect(self.on_remove_mask_layer)
        a["apply_mask_layer"].triggered.connect(self.on_apply_mask_layer)
        a["invert_mask_layer"].triggered.connect(self.on_invert_mask_layer)
        a["convert_to_mask"].triggered.connect(self.on_convert_to_mask)
        a["toggle_vis"].triggered.connect(self.on_toggle_vis_selected)
        a["flatten"].triggered.connect(self.on_flatten)
        a["merge_down"].triggered.connect(self.on_merge_down)
        a["resize_canvas"].triggered.connect(self.on_resize_canvas)
        a["resize_image"].triggered.connect(self.on_resize_image)

        # Layers panel
        lp = mw._layers_panel
        lp.layer_selected.connect(self.on_layer_selected)
        lp.add_requested.connect(self.on_add_layer)
        lp.duplicate_requested.connect(self.on_dup_layer)
        lp.delete_requested.connect(self.on_del_layer)
        lp.group_requested.connect(self.on_add_group)
        lp.mask_requested.connect(self.on_add_mask)
        lp.styles_requested.connect(self.on_layer_styles)
        lp.opacity_changed.connect(self.on_opacity)
        lp.blend_mode_changed.connect(self.on_blend_mode)
        lp.blend_mode_hovered.connect(self.on_blend_hover)
        lp.blend_mode_hover_ended.connect(self.on_blend_hover_end)
        lp.visibility_toggled.connect(self.on_toggle_vis)
        lp.lock_toggled.connect(self.on_toggle_lock)
        lp.layers_reordered.connect(self.on_layers_reordered)
        lp.layers_reparented.connect(self.on_layers_reparented)
        lp.layers_reordered_into_group.connect(self.on_layers_reordered_into_group)
        lp.layers_unparented.connect(self.on_layers_unparented)
        lp.mask_dropped_on_layer.connect(self.on_mask_dropped_on_layer)
        lp.adj_filter_dropped_on_layer.connect(self.on_adj_filter_dropped_on_layer)
        lp.clip_to_layer.connect(self.on_clip_to_layer)
        lp.layer_dropped_as_mask.connect(self.on_layer_dropped_as_mask)
        lp.rename_requested.connect(self.on_rename_layer)
        lp.adjustment_layer_requested.connect(self.on_adjustment_layer_requested)
        lp.edit_adjustment_requested.connect(self.on_edit_adjustment_requested)
        lp.filter_layer_requested.connect(self.on_filter_layer_requested)
        lp.edit_filter_requested.connect(self.on_edit_filter_requested)
        lp.multi_selection_changed.connect(self.on_panel_multi_selection)

        # Move tool auto-select
        move_tool = mw._tools._tools.get(ToolType.MOVE)
        if move_tool is not None:
            move_tool.on_layer_auto_selected = self.on_move_auto_select
            move_tool.on_deselect_all = self.on_move_deselect_all
            move_tool.on_marquee_select = self.on_move_marquee_select

    def on_layer_styles(self) -> None:
        """Open the Layer Styles dialog for the active layer."""
        mw = self._mw
        if not mw._doc:
            return
        layer = mw._doc.layers.active_layer
        if not layer:
            return

        old_styles = copy.deepcopy(layer.styles)
        dlg = LayerStylesDialog(existing_styles=layer.styles, parent=mw)

        _connected_panels: set[str] = set()

        def _preview() -> None:
            layer._styles = dlg.get_styles()
            mw._refresh()
            _wire_panels()

        def _wire_panels() -> None:
            for key, panel in dlg._panels.items():
                if key not in _connected_panels:
                    panel.changed.connect(_preview)
                    _connected_panels.add(key)

        _wire_panels()

        def _accept(styles: list) -> None:
            layer._styles = list(styles)
            mw._doc.save_snapshot(f"Layer Style \u2013 {layer.name}")
            mw._refresh()

        def _reject() -> None:
            layer._styles = old_styles
            mw._refresh()

        dlg.styles_accepted.connect(_accept)
        dlg.rejected.connect(_reject)
        dlg.exec()

    def on_adjustment_layer_requested(self, name: str) -> None:
        self.signals.adjustment_layer_requested.emit(name)

    def on_edit_adjustment_requested(self, layer_id: str) -> None:
        self.signals.edit_adjustment_requested.emit(layer_id)

    def on_filter_layer_requested(self, name: str) -> None:
        self.signals.filter_layer_requested.emit(name)

    def on_edit_filter_requested(self, layer_id: str) -> None:
        self.signals.edit_filter_requested.emit(layer_id)

    def on_add_layer(self) -> None:
        if self.doc:
            self.ctx.execute_command(AddLayerCommand())

    def on_add_vector_layer(self) -> None:
        if self.doc:
            self.doc.add_vector_layer()
            self.ctx.refresh()

    def on_add_group(self) -> None:
        if self.doc is None:
            return
        selected = self.ctx.selected_layer_ids()
        self.ctx.execute_command(AddGroupCommand(layer_ids=selected if len(selected) >= 1 else None))

    def on_dup_layer(self) -> None:
        if self.doc is None or not self.doc.layers.active_layer:
            return
        if self.doc.selection.active and self.doc.selection.mask is not None:
            self.signals.duplicate_selection_requested.emit()
        else:
            self.ctx.execute_command(DuplicateLayerCommand(self.doc.layers.active_layer.id))

    def on_del_layer(self) -> None:
        if self.doc is None or len(self.doc.layers) <= 1:
            return
        selected = self.ctx.selected_layer_ids()
        if selected:
            for lid in selected:
                if len(self.doc.layers) <= 1:
                    break
                if self.doc.layers.get(lid):
                    self.ctx.execute_command(RemoveLayerCommand(lid))
        elif self.doc.layers.active_layer:
            self.ctx.execute_command(RemoveLayerCommand(self.doc.layers.active_layer.id))

    def on_add_mask(self) -> None:
        if self.doc and self.doc.layers.active_layer:
            if self.doc.selection.active and self.doc.selection.mask is not None:
                self.doc.selection_to_mask_layer()
                self.ctx.refresh()
            else:
                self.ctx.execute_command(AddMaskLayerCommand(fill_white=True))

    def on_add_mask_black(self) -> None:
        if self.doc and self.doc.layers.active_layer:
            self.ctx.execute_command(AddMaskLayerCommand(fill_white=False))

    def on_add_mask_standalone(self) -> None:
        if self.doc:
            self.ctx.execute_command(AddMaskLayerCommand(standalone=True))

    def on_remove_mask_layer(self) -> None:
        if self.doc is None:
            return
        active = self.doc.layers.active_layer
        if active and active.layer_type == LayerType.MASK:
            self.ctx.execute_command(RemoveMaskLayerCommand(active.id))

    def on_apply_mask_layer(self) -> None:
        if self.doc is None:
            return
        active = self.doc.layers.active_layer
        if active and active.layer_type == LayerType.MASK:
            self.ctx.execute_command(ApplyMaskLayerCommand(active.id))

    def on_invert_mask_layer(self) -> None:
        if self.doc is None:
            return
        active = self.doc.layers.active_layer
        if active and active.layer_type == LayerType.MASK:
            self.ctx.execute_command(InvertMaskLayerCommand(active.id))

    def on_convert_to_mask(self) -> None:
        if self.doc is None:
            return
        active = self.doc.layers.active_layer
        if active and active.layer_type not in (LayerType.MASK, LayerType.GROUP):
            self.ctx.execute_command(ConvertToMaskCommand(active.id))

    def on_layer_selected(self, stack_index: int) -> None:
        if self.doc:
            self.doc.layers.active_index = stack_index

            if self.mw._tools.active_type == ToolType.NODE:
                al = self.doc.layers.active_layer
                if al:
                    vl = getattr(al, "_vector_data", None)
                    if vl and not vl.selected_objects() and vl.objects:
                        vl.objects[-1].selected = True
                        self.signals.canvas_update_requested.emit()

            self.signals.transform_box_requested.emit()
            self.signals.transform_panel_refresh_requested.emit()
            self.signals.channels_panel_refresh_requested.emit()
            self.signals.properties_panel_requested.emit()

            # Refresh boolean toolbar when a single layer is selected
            if self.mw._tools.active_type == ToolType.NODE:
                tool = self.mw._tools.active_tool
                if tool is not None and hasattr(tool, '_sync_bool_selection'):
                    tool._sync_bool_selection(self.doc)
                self.signals.vector_bool_state_requested.emit()

    def on_move_auto_select(self, stack_index: int) -> None:
        if not self.doc:
            return
        # Sync the layers panel selection with the model's multi-selection
        sync_panel_selection(self.doc, self.mw._layers_panel)
        self.signals.transform_panel_refresh_requested.emit()
        self.signals.channels_panel_refresh_requested.emit()
        self.signals.transform_box_requested.emit()

    def on_move_deselect_all(self) -> None:
        """Called when the Move tool clicks on empty canvas — deselect all."""
        if not self.doc:
            return
        self.ctx.refresh_layers_panel()
        self.signals.transform_box_requested.emit()
        self.signals.canvas_update_requested.emit()

    def on_move_marquee_select(self, indices: list[int]) -> None:
        """Called after a marquee drag-select completes in the Move tool."""
        if not self.doc:
            return
        sync_panel_selection(self.doc, self.mw._layers_panel)
        self.signals.transform_box_requested.emit()
        self.signals.channels_panel_refresh_requested.emit()
        self.signals.canvas_update_requested.emit()

    def on_panel_multi_selection(self, layer_ids: list) -> None:
        """Sync panel multi-selection back to LayerStack and update bbox."""
        mw = self.mw
        if not self.doc:
            return
        stack = self.doc.layers
        new_sel = selected_indices_from_layer_ids(layer_ids, stack.layers)
        stack._selected_indices = new_sel
        # Keep active_index pointing at something sensible
        if new_sel and stack.active_index not in new_sel:
            stack._active_index = max(new_sel)
        elif not new_sel:
            stack._active_index = -1
        self.signals.transform_box_requested.emit()
        self.signals.channels_panel_refresh_requested.emit()

        # Refresh boolean toolbar state when selection changes from the panel
        if mw._tools.active_type == ToolType.NODE:
            tool = mw._tools.active_tool
            if tool is not None and hasattr(tool, '_sync_bool_selection'):
                tool._sync_bool_selection(self.doc)
            self.signals.vector_bool_state_requested.emit()

    def on_opacity(self, val: float) -> None:
        if self.doc and self.doc.layers.active_layer:
            self.doc.layers.active_layer.opacity = val
            self.ctx.refresh_canvas_only()

    def on_blend_mode(self, mode: BlendMode) -> None:
        self._blend_preview_original = None
        if self.doc and self.doc.layers.active_layer:
            self.doc.layers.active_layer.blend_mode = mode
            self.ctx.refresh_canvas_only()

    def on_blend_hover(self, mode: BlendMode) -> None:
        if not self.doc or not self.doc.layers.active_layer:
            return
        if self._blend_preview_original is None:
            self._blend_preview_original = self.doc.layers.active_layer.blend_mode
        self.doc.layers.active_layer.blend_mode = mode
        self.ctx.refresh_canvas_only()

    def on_blend_hover_end(self) -> None:
        if not self.doc or not self.doc.layers.active_layer:
            return
        if self._blend_preview_original is not None:
            self.doc.layers.active_layer.blend_mode = self._blend_preview_original
            self._blend_preview_original = None
            self.ctx.refresh_canvas_only()

    def on_toggle_vis(self, layer_id: str) -> None:
        if self.doc:
            layer = self.doc.layers.get(layer_id)
            if layer:
                layer.visible = not layer.visible
                self.ctx.refresh_canvas_only()
                self.ctx.refresh_layers_panel(thumbnails=False)

    def on_toggle_lock(self, layer_id: str) -> None:
        if self.doc:
            layer = self.doc.layers.get(layer_id)
            if layer:
                layer.locked = not layer.locked
                self.ctx.refresh_layer_controls()
                self.signals.transform_box_requested.emit()

    def on_toggle_vis_selected(self) -> None:
        self.ctx.toggle_selected_layer_visibility()

    def on_rename_layer(self, layer_id: str, new_name: str) -> None:
        if self.doc:
            self.ctx.execute_command(RenameLayerCommand(layer_id, new_name))

    def on_layers_reordered(self, layer_ids: list[str], target_visual_row: int) -> None:
        if not self.doc:
            return
        display_ids = self.ctx.layer_row_ids()
        new_stack_order = reordered_stack_order(display_ids, layer_ids, target_visual_row)
        # reordered_stack_order derives the WHOLE stack order from the rows
        # the panel is showing, so a partial view yields a partial order and
        # reorder_by_ids would drop every layer missing from it. The panel
        # disables dragging while filtered; this is the backstop.
        if len(new_stack_order) != len(self.doc.layers.layers):
            self.ctx.show_status_message(
                "Clear the layer filter before reordering", 3000)
            return
        self.ctx.execute_command(ReorderLayersCommand(new_stack_order))

    def on_layers_reparented(self, layer_ids: list[str], group_id: str) -> None:
        if self.doc is None:
            return
        self.ctx.execute_command(MoveLayerCommand(layer_ids, target_parent_id=group_id))

    def on_layers_reordered_into_group(
        self, layer_ids: list[str], group_id: str, target_visual_row: int,
    ) -> None:
        """Reparent layers into a group AND reorder to a specific position."""
        if self.doc is None:
            return
        display_ids = self.ctx.layer_row_ids()
        new_stack_order = reordered_stack_order(
            display_ids, layer_ids, target_visual_row,
        )
        doc = self.doc
        doc.layers.reparent(layer_ids, group_id)
        doc.layers.reorder_by_ids(new_stack_order)
        doc.save_snapshot("Move into Group")
        doc.mark_dirty()
        self.ctx.refresh()

    def on_layers_unparented(self, layer_ids: list[str], target_visual_row: int) -> None:
        if self.doc is None:
            return
        # Get current display order BEFORE mutating the stack
        display_ids = self.ctx.layer_row_ids()
        # Compute the desired stack order first, then unparent + reorder
        # atomically in a single snapshot.
        new_stack_order = reordered_stack_order(
            display_ids, layer_ids, target_visual_row,
        )
        doc = self.doc
        doc.layers.reparent(layer_ids, None)
        doc.layers.reorder_by_ids(new_stack_order)
        doc.save_snapshot("Unparent Layer")
        doc.mark_dirty()
        self.ctx.refresh()

    def on_mask_dropped_on_layer(self, mask_id: str, target_id: str) -> None:
        if self.doc is None:
            return
        self.ctx.execute_command(AttachMaskToLayerCommand(mask_id, target_id))

    def on_adj_filter_dropped_on_layer(self, adj_id: str, target_id: str) -> None:
        if self.doc is None:
            return
        self.ctx.execute_command(AttachAdjustmentToLayerCommand(adj_id, target_id))

    def on_clip_to_layer(self, layer_id: str, target_id: str) -> None:
        if self.doc is None:
            return
        self.ctx.execute_command(ClipToLayerCommand(layer_id, target_id))

    def on_layer_dropped_as_mask(self, layer_id: str, target_id: str) -> None:
        if self.doc is None:
            return
        self.ctx.execute_command(DropAsMaskCommand(layer_id, target_id))

    def on_flatten(self) -> None:
        if self.doc:
            self.ctx.execute_command(FlattenCommand())

    def on_merge_down(self) -> None:
        if self.doc:
            success = self.ctx.execute_command(MergeDownCommand())
            if success is False:
                self.ctx.show_status_message(
                    "Cannot merge down — no suitable layer below", 3000
                )

    def on_resize_canvas(self) -> None:
        if not self.doc:
            return
        from ..dialogs.new_document import NewDocumentDialog
        dlg = NewDocumentDialog(self.mw)
        dlg.setWindowTitle("Canvas Size")
        dlg._width.setValue(self.doc.width)
        dlg._height.setValue(self.doc.height)
        if dlg.exec():
            w, h, _ = dlg.get_values()
            if w > 0 and h > 0:
                resize_canvas(self.doc, w, h)
                self.ctx.refresh()

    def rasterize_active_layer(self) -> None:
        """Convert the active text layer into a plain raster layer."""
        rasterize_active_text_layer(self.doc, self.ctx.refresh)

    def handle_numpad_opacity(self, key: int) -> bool:
        """Process a numpad digit for opacity. Returns True if consumed."""
        mw = self._mw
        digit = NUMPAD_MAP.get(key)
        if digit is None:
            return False
        if not mw._doc or not mw._doc.layers.active_layer:
            return False

        if self._numpad_timer is None:
            self._numpad_timer = QTimer(mw)
            self._numpad_timer.setSingleShot(True)
            self._numpad_timer.setInterval(500)
            self._numpad_timer.timeout.connect(self._numpad_commit)

        if self._numpad_first is not None:
            pct = self._numpad_first * 10 + digit
            pct = max(0, min(100, pct))
            self._numpad_first = None
            self._numpad_timer.stop()
            self._set_opacity_pct(pct)
            return True
        self._numpad_first = digit
        self._numpad_timer.start()
        return True

    def _numpad_commit(self) -> None:
        if self._numpad_first is not None:
            d = self._numpad_first
            pct = 100 if d == 0 else d * 10
            self._numpad_first = None
            self._set_opacity_pct(pct)

    def _set_opacity_pct(self, pct: int) -> None:
        mw = self._mw
        if mw._doc and mw._doc.layers.active_layer:
            mw._doc.layers.active_layer.opacity = pct / 100.0
            self.ctx.refresh_layer_controls()
            self.ctx.refresh_canvas_only()
            self.ctx.show_status_message(f"Opacity: {pct}%", 1500)

    def on_resize_image(self) -> None:
        if not self.doc:
            return
        from ..dialogs.new_document import NewDocumentDialog
        dlg = NewDocumentDialog(self.mw)
        dlg.setWindowTitle("Image Size")
        dlg._width.setValue(self.doc.width)
        dlg._height.setValue(self.doc.height)
        if dlg.exec():
            import cv2
            new_w, new_h, _ = dlg.get_values()
            if new_w < 1 or new_h < 1:
                return
            resize_image(
                self.doc,
                new_w,
                new_h,
                lambda pixels, size: cv2.resize(pixels, size, interpolation=cv2.INTER_AREA),
            )
            self.ctx.refresh()
