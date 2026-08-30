"""Virtualized layer list with custom pointer-based drag & drop.

Replaces the old QListWidget approach with a QScrollArea + manual row layout
so that only visible rows (plus overscan) are instantiated.  All drag visual
feedback is rendered by :class:`DragOverlay` — the tree is never re-rendered
during a drag.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QPointF,
    QRectF,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QWidget,
)

from .base import (
    GAP_ANIM_MS,
    EJECT_HOLD_MS,
    INDENT_WIDTH,
    MAX_INDENT_DEPTH,
    OVERSCAN_ROWS,
    ROLE_IS_ADJ_FILTER,
    ROLE_IS_CLIPPED,
    ROLE_IS_GROUP,
    ROLE_IS_MASK,
    ROLE_IS_SEP,
    ROLE_INDENT,
    ROLE_LAYER_ID,
    ROLE_PARENT_ID,
    ROW_HEIGHT,
    SEP_HEIGHT,
    THUMB_SIZE,
)
from .drag_manager import (
    DragState,
    DropMode,
    get_drop_index,
    get_drop_mode,
    infer_target_depth,
    is_descendant_of,
)
from .drag_overlay import DragOverlay
from .layer_delegate import LayerItemDelegate
from ...theme import ThemeManager
from ...styles import render_qss

# Minimum pointer displacement (px) before a press becomes a drag.
_DRAG_THRESHOLD = 5


class LayerListWidget(QListWidget):
    """QListWidget subclass with custom pointer-driven drag & drop.

    The built-in QListWidget DnD is disabled; all drag handling is done via
    pointer events so we can:
      - keep drag state in a mutable ref (no re-renders during drag)
      - composite visual overlays (insertion line, gap, badges) in a single
        overlay widget (never re-render the tree)
      - support the three-mode drop: reorder / nest / clip

    The existing QListWidget item machinery is still used for selection, row
    indexing, keyboard nav, etc.  Virtualization is achieved by skipping
    expensive thumbnail generation for off-screen rows (handled in panel.py)
    and by memoizing row widgets.
    """

    # ---- signals (unchanged API) -------------------------------------------
    layers_reordered = Signal(list, int)
    layers_dropped_in_group = Signal(list, str)
    layers_reordered_into_group = Signal(list, str, int)  # (ids, group_id, visual_row)
    layers_unparented = Signal(list, int)  # (layer_ids, target_visual_row)
    mask_dropped_on_layer = Signal(str, str)
    adj_filter_dropped_on_layer = Signal(str, str)
    clip_to_layer = Signal(str, str)  # (dragged_id, target_id) — legacy
    layer_dropped_as_mask = Signal(str, str)  # raster layer → shape-mask child
    delete_key_pressed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Disable built-in DnD — we handle it ourselves
        self.setDragDropMode(QListWidget.DragDropMode.NoDragDrop)
        self.setDefaultDropAction(Qt.DropAction.IgnoreAction)
        self.setDragEnabled(False)
        self.setAcceptDrops(False)

        self.setItemDelegate(LayerItemDelegate(self))

        # Reordering is refused while the panel shows a filtered subset:
        # a drop position among a subset says nothing about where the hidden
        # layers belong, and the reorder is computed from the visible rows.
        self.reorder_enabled = True

        # ---- drag state (ref, not reactive state) --------------------------
        self._drag = DragState()
        self._children_map: dict[str, list[str]] = {}

        # ---- overlay for drag visuals --------------------------------------
        self._overlay = DragOverlay(self.viewport())
        self._overlay.set_drag(self._drag)
        self._overlay.hide()

        # Eject timer
        self._eject_timer = QTimer(self)
        self._eject_timer.setSingleShot(True)
        self._eject_timer.setInterval(EJECT_HOLD_MS)
        self._eject_timer.timeout.connect(self._on_eject_timeout)

        # Throttle overlay repaints to ~60 fps
        self._repaint_pending = False

        # For ghost row styling
        self._ghost_rows: set[int] = set()

        # Track old mask drop target for style cleanup
        self._mask_drop_target: QListWidgetItem | None = None

        ThemeManager.instance().theme_changed.connect(self._apply_theme)
        self._apply_theme(ThemeManager.instance().active_palette)

    def _apply_theme(self, palette: dict) -> None:
        self.setStyleSheet(render_qss("layer_list.qss", palette))

    # ---- children map (set by panel.refresh) --------------------------------

    def set_children_map(self, children_map: dict[str, list[str]]) -> None:
        """Provide group→children mapping for circular-drop detection."""
        self._children_map = children_map

    # ---- overlay geometry ---------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._overlay.setGeometry(self.viewport().rect())

    def _sync_overlay_geometry(self) -> None:
        """Recalculate row rects for the overlay."""
        rects: list[QRectF] = []
        indents: list[int] = []
        for i in range(self.count()):
            item = self.item(i)
            if item is None:
                continue
            rect = self.visualItemRect(item)
            rects.append(QRectF(rect))
            indents.append(item.data(ROLE_INDENT) or 0)
        self._overlay.set_row_geometry(rects, indents)

    # ---- ghost row helpers --------------------------------------------------

    def set_ghost_rows(self, rows: set[int]) -> None:
        for r in self._ghost_rows - rows:
            w = self.itemWidget(self.item(r)) if r < self.count() else None
            if w:
                w.setGraphicsEffect(None)
                w.setStyleSheet(w.styleSheet().replace("border: 1px dashed #666;", ""))
        self._ghost_rows = rows
        for r in rows:
            if r < self.count():
                w = self.itemWidget(self.item(r))
                if w:
                    # Apply ghost appearance: reduced opacity + dashed border
                    effect = QGraphicsOpacityEffect(w)
                    effect.setOpacity(0.3)
                    w.setGraphicsEffect(effect)

    def clear_ghost_rows(self) -> None:
        for r in self._ghost_rows:
            if r < self.count():
                w = self.itemWidget(self.item(r))
                if w:
                    w.setGraphicsEffect(None)
        self._ghost_rows.clear()

    # ---- thumbnail rect for a given row ------------------------------------

    def _thumbnail_rect_for_row(self, row: int) -> QRectF | None:
        """Return the thumbnail QLabel rect in viewport coords for *row*."""
        item = self.item(row)
        if item is None:
            return None
        widget = self.itemWidget(item)
        if widget is None:
            return None
        # Find the thumbnail label inside the row widget
        thumb = getattr(widget, "_thumb_label", None)
        if thumb is None:
            return None
        # Map thumbnail geometry to viewport coords
        pos = thumb.mapTo(self.viewport(), thumb.rect().topLeft())
        return QRectF(pos.x(), pos.y(), thumb.width(), thumb.height())

    # =====================================================================
    # Pointer-event-driven drag & drop
    # =====================================================================

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        pos = event.position().toPoint()
        item = self.itemAt(pos)

        # Let normal selection work first
        super().mousePressEvent(event)

        if item is None:
            return
        if item.data(ROLE_IS_SEP):
            return

        # Check if the click is on an interactive control (button) — skip drag
        widget = self.itemWidget(item)
        if widget:
            child = widget.childAt(widget.mapFrom(self.viewport(), pos))
            if isinstance(child, QPushButton):
                return

        layer_id = item.data(ROLE_LAYER_ID)
        if not layer_id:
            return
        if not self.reorder_enabled:
            return          # selection still works; dragging does not

        # Prepare drag ref
        self._drag.reset()
        self._drag.dragging = True
        self._drag.start_x = pos.x()
        self._drag.start_y = pos.y()
        self._drag.pointer_x = pos.x()
        self._drag.pointer_y = pos.y()
        self._drag.source_parent_id = item.data(ROLE_PARENT_ID) or None

        # Collect all selected items
        selected = self.selectedItems()
        if not selected or item not in selected:
            selected = [item]
        ids = []
        rows_set = set()
        any_locked = False
        for sel_item in selected:
            lid = sel_item.data(ROLE_LAYER_ID)
            if lid and lid != "__sep__":
                ids.append(lid)
                rows_set.add(self.row(sel_item))
                # Check lock status from the row widget
                w = self.itemWidget(sel_item)
                if w and hasattr(w, "_lock_icon"):
                    any_locked = True

        self._drag.dragged_ids = ids
        self._drag.source_indices = sorted(rows_set)
        self._drag.dragged_locked = any_locked

    def mouseMoveEvent(self, event) -> None:
        if not self._drag.dragging:
            # Not in drag-prep — let QListWidget handle normally
            super().mouseMoveEvent(event)
            return

        # Suppress QListWidget rubber-band selection once we own the gesture
        event.accept()

        pos = event.position().toPoint()
        self._drag.pointer_x = pos.x()
        self._drag.pointer_y = pos.y()

        # Check drag threshold
        if not self._drag.drag_started:
            dx = pos.x() - self._drag.start_x
            dy = pos.y() - self._drag.start_y
            if (dx * dx + dy * dy) < _DRAG_THRESHOLD * _DRAG_THRESHOLD:
                return
            # Locked layers cannot be dragged
            if self._drag.dragged_locked:
                self._drag.reset()
                self.setCursor(Qt.CursorShape.ForbiddenCursor)
                return
            self._drag.drag_started = True
            self._start_drag_visuals()
            self.viewport().grabMouse()

        # Update drop target
        self._update_drop_target(pos.x(), pos.y())

        # Schedule overlay repaint (throttled)
        self._schedule_overlay_repaint()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return

        if not self._drag.drag_started:
            self._drag.reset()
            super().mouseReleaseEvent(event)
            return

        self.viewport().releaseMouse()
        self.unsetCursor()
        self._eject_timer.stop()

        if self._drag.committed or not self._drag.dragging:
            self._end_drag_visuals()
            self._drag.reset()
            return

        # Commit the drag
        self._commit_drop()
        self._end_drag_visuals()
        self._drag.reset()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self._drag.drag_started:
            # Cancel drag
            self.viewport().releaseMouse()
            self.unsetCursor()
            self._eject_timer.stop()
            self._end_drag_visuals()
            self._drag.reset()
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_key_pressed.emit()
            return
        super().keyPressEvent(event)

    # =====================================================================
    # Drag visual start / end
    # =====================================================================

    def _start_drag_visuals(self) -> None:
        """Set up overlay and ghost rows at drag start."""
        self._sync_overlay_geometry()
        self._overlay.setGeometry(self.viewport().rect())
        self._overlay.show()
        self._overlay.raise_()

        # Set ghost rows
        self.set_ghost_rows(set(self._drag.source_indices))

        # Setup drag chip preview
        if self._drag.dragged_ids:
            first_id = self._drag.dragged_ids[0]
            for i in range(self.count()):
                item = self.item(i)
                if item and item.data(ROLE_LAYER_ID) == first_id:
                    w = self.itemWidget(item)
                    if w:
                        thumb = getattr(w, "_thumb_label", None)
                        pm = thumb.pixmap() if thumb and thumb.pixmap() else None
                        name_lbl = getattr(w, "_name_label", None)
                        name = name_lbl.text() if name_lbl else "Layer"
                        self._overlay.set_preview(pm, name)
                    break

    def _end_drag_visuals(self) -> None:
        """Clean up overlay and ghost rows."""
        self._overlay.hide()
        self.clear_ghost_rows()
        self._overlay.set_invalid(False)

    def _schedule_overlay_repaint(self) -> None:
        if not self._repaint_pending:
            self._repaint_pending = True
            QTimer.singleShot(0, self._do_overlay_repaint)

    def _do_overlay_repaint(self) -> None:
        self._repaint_pending = False
        if self._drag.drag_started:
            self._overlay.schedule_repaint()

    # =====================================================================
    # Drop target computation
    # =====================================================================

    def _update_drop_target(self, px: float, py: float) -> None:
        """Recompute the drop target from pointer position.

        All updates go into ``self._drag`` (the ref), NOT widget state.
        """
        dragged_set = set(self._drag.dragged_ids)

        # Find which row the pointer is over
        target_item = self.itemAt(int(px), int(py))
        target_row = -1
        target_id: str | None = None
        target_indent = 0
        is_group = False

        if target_item:
            target_row = self.row(target_item)
            target_id = target_item.data(ROLE_LAYER_ID)
            target_indent = target_item.data(ROLE_INDENT) or 0
            is_group = bool(target_item.data(ROLE_IS_GROUP))

            # Skip separators
            if target_item.data(ROLE_IS_SEP):
                target_id = None
                target_row = -1

        # --- Invalid: drag onto self ---
        if target_id and target_id in dragged_set:
            self._drag.drop_mode = None
            self._drag.drop_target_id = None
            self._drag.drop_target_row = target_row
            self._drag.insert_index = -1
            self._overlay.set_invalid(True)
            return

        # --- Invalid: drag group onto own descendant ---
        if target_id and is_descendant_of(target_id, dragged_set, self._children_map):
            self._drag.drop_mode = None
            self._drag.drop_target_id = None
            self._drag.drop_target_row = target_row
            self._drag.insert_index = -1
            self._overlay.set_invalid(True)
            return

        self._overlay.set_invalid(False)

        # --- Unparent gesture ---
        # A nested layer is unparented when the user drags it to a
        # position that is at a shallower depth than its current nesting.
        # Two triggers:
        #   1. The pointer X is left of the source indent (classic drag-left).
        #   2. The target row under the pointer is at root level (indent 0)
        #      and the pointer is in the REORDER zone (top/bottom 25%).
        source_depth = 0
        if self._drag.source_parent_id:
            src_row = self._drag.source_indices[0] if self._drag.source_indices else -1
            if src_row >= 0 and src_row < self.count():
                src_item = self.item(src_row)
                source_depth = (src_item.data(ROLE_INDENT) or 0) if src_item else 0

        inferred_depth = infer_target_depth(px)

        _want_unparent = False
        if self._drag.source_parent_id and source_depth > 0:
            # Trigger 1: pointer dragged left of source indent
            if inferred_depth < source_depth:
                _want_unparent = True
            # Trigger 2: hovering a root-level row in its reorder zone
            elif target_indent == 0 and target_item is not None:
                row_rect = QRectF(self.visualItemRect(target_item))
                _zone_h = row_rect.height() * 0.25
                _local_y = py - row_rect.top()
                if _local_y < _zone_h or _local_y > row_rect.height() - _zone_h:
                    _want_unparent = True

        if _want_unparent:
            # User wants to unparent — treat as root-level reorder
            self._drag.target_depth = 0
            target_indent = 0
            # Compute insertion index at root level
            row_tops = []
            row_heights = []
            for i in range(self.count()):
                it = self.item(i)
                if it:
                    rect = self.visualItemRect(it)
                    row_tops.append(float(rect.top()))
                    row_heights.append(float(rect.height()))
            idx = get_drop_index(py, row_tops, self.count(), row_heights)
            self._drag.drop_mode = DropMode.REORDER
            self._drag.drop_target_id = None
            self._drag.drop_target_row = -1
            self._drag.insert_index = idx
            self._manage_eject_timer(is_group, target_row)
            return

        if target_id is None:
            # Pointer is below all rows or on a separator — append at end.
            # If the source is nested, treat as unparent to root.
            if self._drag.source_parent_id and source_depth > 0:
                self._drag.target_depth = 0
                self._drag.drop_mode = DropMode.REORDER
                self._drag.drop_target_id = None
                self._drag.drop_target_row = -1
                self._drag.insert_index = self.count()
                return
            self._drag.drop_mode = DropMode.REORDER
            self._drag.drop_target_id = None
            self._drag.drop_target_row = -1
            self._drag.insert_index = self.count()
            return

        # --- Determine drop mode ---
        row_rect = QRectF(self.visualItemRect(target_item))
        thumb_rect = self._thumbnail_rect_for_row(target_row)
        mode = get_drop_mode(px, py, row_rect, thumb_rect, target_indent)

        # Groups should never receive CLIP drops — treat as NEST instead.
        if mode == DropMode.CLIP and is_group:
            mode = DropMode.NEST

        self._drag.drop_target_id = target_id
        self._drag.drop_target_row = target_row
        self._drag.drop_mode = mode
        # Use the greater of the X-inferred depth and the target row's indent
        # so reordering between nested rows correctly detects group membership.
        self._drag.target_depth = max(inferred_depth, target_indent) if mode == DropMode.REORDER else inferred_depth

        if mode == DropMode.REORDER:
            # Compute insertion index
            row_tops = []
            row_heights = []
            for i in range(self.count()):
                it = self.item(i)
                if it:
                    rect = self.visualItemRect(it)
                    row_tops.append(float(rect.top()))
                    row_heights.append(float(rect.height()))
            self._drag.insert_index = get_drop_index(py, row_tops, self.count(), row_heights)
        else:
            self._drag.insert_index = -1

        self._manage_eject_timer(is_group, target_row)

    def _manage_eject_timer(self, is_group: bool, target_row: int) -> None:
        """Show eject affordance if hovering over a group for 400ms."""
        if is_group and self._drag.source_parent_id and not self._drag.eject_shown:
            if not self._drag.eject_timer_active:
                self._drag.eject_timer_active = True
                self._eject_timer.start()
        else:
            self._eject_timer.stop()
            self._drag.eject_timer_active = False

    def _on_eject_timeout(self) -> None:
        self._drag.eject_shown = True
        self._drag.eject_timer_active = False
        self._schedule_overlay_repaint()

    # =====================================================================
    # Commit the drop
    # =====================================================================

    def _detect_group_at(self, insert_index: int) -> str | None:
        """Return the parent group id if *insert_index* falls inside a group's children zone."""
        # Check the row AT insert_index
        if 0 <= insert_index < self.count():
            item = self.item(insert_index)
            if item and not item.data(ROLE_IS_SEP):
                parent = item.data(ROLE_PARENT_ID)
                if parent:
                    return parent
        # Check the row BEFORE insert_index
        prev = insert_index - 1
        if 0 <= prev < self.count():
            item = self.item(prev)
            if item and not item.data(ROLE_IS_SEP):
                parent = item.data(ROLE_PARENT_ID)
                if parent:
                    return parent
                # If the previous row is an expanded group header,
                # the insert is at the first child position.
                if item.data(ROLE_IS_GROUP):
                    return item.data(ROLE_LAYER_ID)
        return None

    def _commit_drop(self) -> None:
        """Translate drag state into the appropriate signal emission."""
        drag = self._drag
        source_ids = list(drag.dragged_ids)
        if not source_ids:
            return

        mode = drag.drop_mode
        target_id = drag.drop_target_id

        # Determine source layer type from the first dragged item
        src_row = drag.source_indices[0] if drag.source_indices else -1
        src_item = self.item(src_row) if 0 <= src_row < self.count() else None
        source_is_mask = bool(src_item and src_item.data(ROLE_IS_MASK))
        source_is_adj = bool(src_item and src_item.data(ROLE_IS_ADJ_FILTER))

        # ----- REORDER -----
        if mode == DropMode.REORDER:
            source_parent = drag.source_parent_id
            idx = drag.insert_index

            # If source was nested and user dragged to root level
            if source_parent and drag.target_depth == 0:
                self.layers_unparented.emit(source_ids, idx)
                return

            # If the drop position falls between children of a group,
            # reparent into that group and reorder.
            if drag.target_depth > 0:
                group_id = self._detect_group_at(idx)
                if group_id:
                    self.layers_reordered_into_group.emit(
                        source_ids, group_id, idx,
                    )
                    return

            self.layers_reordered.emit(source_ids, idx)
            return

        # ----- NEST (middle of row — any target layer) -----
        if mode == DropMode.NEST and target_id:
            if target_id not in set(source_ids):
                if source_is_mask:
                    # Mask layers → attach to target's mask_layers
                    for sid in source_ids:
                        self.mask_dropped_on_layer.emit(sid, target_id)
                elif source_is_adj:
                    # Adj/filter → attach as adj/filter child
                    for sid in source_ids:
                        self.adj_filter_dropped_on_layer.emit(sid, target_id)
                else:
                    # Raster/group/other → reparent as child
                    self.layers_dropped_in_group.emit(source_ids, target_id)
            return

        # ----- CLIP (thumbnail — acts as mask) -----
        if mode == DropMode.CLIP and target_id:
            for sid in source_ids:
                if sid != target_id:
                    if source_is_mask:
                        self.mask_dropped_on_layer.emit(sid, target_id)
                    elif source_is_adj:
                        self.adj_filter_dropped_on_layer.emit(sid, target_id)
                    else:
                        # Convert regular layer to mask for the target
                        self.layer_dropped_as_mask.emit(sid, target_id)
            return

    # =====================================================================
    # Legacy compat for separator / mask highlight from old code
    # =====================================================================

    def _clear_mask_highlight(self) -> None:
        if self._mask_drop_target is not None:
            w = self.itemWidget(self._mask_drop_target)
            if w:
                w.setStyleSheet("background: transparent;")
            self._mask_drop_target = None

    # =====================================================================
    # Paint event — still used for selection highlight via delegate
    # =====================================================================

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        # The overlay handles all drag visuals; nothing else needed here.
