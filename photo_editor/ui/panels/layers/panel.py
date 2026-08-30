"""Main LayersPanel — dockable panel for managing the layer stack."""

from __future__ import annotations

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QLineEdit,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ....core.document import Document
from ....core.enums import BlendMode, LayerType
from ....core.layer_filter import LayerFilter, match_count, visible_layer_ids
from ...styles import render_qss, themed_value

from .base import (
    ROLE_INDENT,
    ROLE_IS_ADJ_FILTER,
    ROLE_IS_CLIPPED,
    ROLE_IS_GROUP,
    ROLE_IS_MASK,
    ROLE_IS_SEP,
    ROLE_LAYER_ID,
    ROLE_PARENT_ID,
    ROW_HEIGHT,
    SEP_HEIGHT,
    h_separator,
    toolbar_btn,
)
from ...theme import ThemeManager
from .blend_combo import BlendModeCombo
from .icons import icon_lock, ico_adjustment, ico_filter, ico_folder, ico_fx, ico_grid
from .icons import ico_mask, ico_new_layer, ico_settings, ico_trash, ico_duplicate
from .layer_item import LayerItemWidget
from .layer_list import LayerListWidget
from .thumbnails import make_group_thumbnail, make_thumbnail


class LayersPanel(QWidget):
    """Dockable panel for managing the layer stack."""

    layer_selected = Signal(int)
    visibility_toggled = Signal(str)
    lock_toggled = Signal(str)
    opacity_changed = Signal(float)
    blend_mode_changed = Signal(BlendMode)
    blend_mode_hovered = Signal(object)
    blend_mode_hover_ended = Signal()
    add_requested = Signal()
    delete_requested = Signal()
    duplicate_requested = Signal()
    group_requested = Signal()
    mask_requested = Signal()
    merge_down_requested = Signal()
    flatten_requested = Signal()
    rename_requested = Signal(str, str)
    styles_requested = Signal()
    adjustment_layer_requested = Signal(str)
    edit_adjustment_requested = Signal(str)
    filter_layer_requested = Signal(str)
    edit_filter_requested = Signal(str)
    layers_reordered = Signal(list, int)
    layers_reparented = Signal(list, str)
    layers_reordered_into_group = Signal(list, str, int)  # (ids, group_id, visual_row)
    layers_unparented = Signal(list, int)  # (layer_ids, target_visual_row)
    mask_dropped_on_layer = Signal(str, str)
    adj_filter_dropped_on_layer = Signal(str, str)
    clip_to_layer = Signal(str, str)  # (dragged_id, target_id)
    layer_dropped_as_mask = Signal(str, str)  # raster → mask conversion
    multi_selection_changed = Signal(list)  # list[str] of selected layer IDs

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._doc: Document | None = None
        self._refreshing = False
        self._row_layer_ids: list[str] = []
        self._row_structure: list[tuple] = []
        # Layers-panel search. A twenty-layer project is where scrolling
        # stops being a way to find anything.
        self._filter = LayerFilter()
        self._collapsed_groups: set[str] = set()
        self._collapsed_masks: set[str] = set()
        self._build_ui()
        ThemeManager.instance().theme_changed.connect(self._apply_theme)
        self._apply_theme(ThemeManager.instance().active_palette)

    def _apply_theme(self, palette: dict) -> None:
        self.setStyleSheet(render_qss("layers_panel_root.qss", palette))
        self._header.setStyleSheet(render_qss("layers_panel_header.qss", palette))
        self._opacity_lbl.setStyleSheet(render_qss("layers_panel_opacity_label.qss", palette))
        self._opacity_spin.setStyleSheet(render_qss("layers_panel_spin.qss", palette))
        self._blend_combo.setStyleSheet(render_qss("layers_panel_combo.qss", palette))
        self._opacity_slider.setStyleSheet(
            render_qss(
                "layers_panel_slider.qss",
                palette,
                slider_handle=themed_value(palette, "fg_accent", palette["fg"]),
                slider_hover="#ffffff",
            )
        )
        self._toolbar.setStyleSheet(render_qss("layers_panel_toolbar.qss", palette))
        menu_button_qss = render_qss("layers_panel_menu_button.qss", palette)
        menu_qss = render_qss("layers_panel_menu.qss", palette)
        self._adj_btn.setStyleSheet(menu_button_qss)
        self._adj_menu.setStyleSheet(menu_qss)
        self._filt_btn.setStyleSheet(menu_button_qss)
        self._filt_menu.setStyleSheet(menu_qss)
        btn_qss = render_qss("layers_panel_bottom_btn.qss", palette)
        for btn in [
            self._settings_btn, self._lock_btn,
            self._btn_new_layer, self._btn_styles, self._btn_mask,
            self._btn_group, self._btn_duplicate, self._btn_flatten, self._btn_delete
        ]:
            btn.setStyleSheet(btn_qss)

        sep_qss = render_qss("layers_panel_separator_line.qss", border=palette.get("border", "#444444"))
        self._sep1.setStyleSheet(sep_qss)
        self._sep2.setStyleSheet(sep_qss)

        self._settings_btn.setIcon(ico_settings())
        self._lock_btn.setIcon(icon_lock(False))  # will be overridden by _sync_active if doc is open
        self._adj_btn.setIcon(ico_adjustment())
        self._filt_btn.setIcon(ico_filter())
        self._btn_new_layer.setIcon(ico_new_layer())
        self._btn_styles.setIcon(ico_fx())
        self._btn_mask.setIcon(ico_mask())
        self._btn_group.setIcon(ico_folder())
        self._btn_duplicate.setIcon(ico_duplicate())
        self._btn_flatten.setIcon(ico_grid())
        self._btn_delete.setIcon(ico_trash())

        # Trigger item rebuild on theme change if document is valid
        if self._doc and hasattr(self, '_list'):
            self.refresh(self._doc, thumbnails=True, force=True)
            self._sync_active(self._doc)

    def _build_ui(self) -> None:

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._header = QWidget()
        header_layout = QVBoxLayout(self._header)
        header_layout.setContentsMargins(6, 4, 6, 4)
        header_layout.setSpacing(4)

        r1 = QHBoxLayout()
        r1.setSpacing(4)

        self._opacity_lbl = QLabel("Opacity:")
        r1.addWidget(self._opacity_lbl)

        self._opacity_spin = QSpinBox()
        self._opacity_spin.setRange(0, 100)
        self._opacity_spin.setValue(100)
        self._opacity_spin.setSuffix(" %")
        self._opacity_spin.setFixedWidth(52)
        self._opacity_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self._opacity_spin.valueChanged.connect(self._on_opacity_changed)
        r1.addWidget(self._opacity_spin)

        self._blend_combo = BlendModeCombo()
        for mode in BlendMode:
            self._blend_combo.addItem(mode.name.replace("_", " ").title(), mode)
        self._blend_combo.currentIndexChanged.connect(self._on_blend_changed)
        self._blend_combo.hover_preview.connect(self.blend_mode_hovered.emit)
        self._blend_combo.hover_ended.connect(self.blend_mode_hover_ended.emit)
        r1.addWidget(self._blend_combo, 1)

        self._settings_btn = QPushButton()
        self._settings_btn.setIcon(ico_settings())
        self._settings_btn.setIconSize(QSize(16, 16))
        self._settings_btn.setFixedSize(22, 22)
        self._settings_btn.setFlat(True)
        self._settings_btn.setToolTip("Layer options")
        self._settings_btn.setStyleSheet("background: transparent; border: none;")
        r1.addWidget(self._settings_btn)

        self._lock_btn = QPushButton()
        self._lock_btn.setIcon(icon_lock(False))
        self._lock_btn.setIconSize(QSize(16, 16))
        self._lock_btn.setFixedSize(22, 22)
        self._lock_btn.setFlat(True)
        self._lock_btn.setToolTip("Toggle lock")
        self._lock_btn.setStyleSheet("background: transparent; border: none;")
        self._lock_btn.clicked.connect(self._on_header_lock_clicked)
        r1.addWidget(self._lock_btn)

        header_layout.addLayout(r1)

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(100)
        self._opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        header_layout.addWidget(self._opacity_slider)

        # --- Search row -------------------------------------------------
        search_row = QHBoxLayout()
        search_row.setContentsMargins(0, 0, 0, 0)
        search_row.setSpacing(4)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search layers…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._on_filter_text)
        search_row.addWidget(self._search, 1)

        self._kind_combo = QComboBox()
        self._kind_combo.addItem("All kinds", "")
        for key, label in (
            ("raster", "Raster"), ("text", "Text"), ("shape", "Shape"),
            ("adjustment", "Adjustment"), ("filter", "Filter"),
            ("group", "Group"), ("mask", "Mask"),
        ):
            self._kind_combo.addItem(label, key)
        self._kind_combo.setFixedWidth(104)
        self._kind_combo.currentIndexChanged.connect(self._on_filter_kind)
        search_row.addWidget(self._kind_combo)

        header_layout.addLayout(search_row)

        self._filter_status = QLabel("")
        self._filter_status.setVisible(False)
        header_layout.addWidget(self._filter_status)

        root.addWidget(self._header)
        self._sep1 = h_separator()
        root.addWidget(self._sep1)

        self._list = LayerListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._list.layers_reordered.connect(self.layers_reordered.emit)
        self._list.layers_dropped_in_group.connect(self.layers_reparented.emit)
        self._list.layers_reordered_into_group.connect(self.layers_reordered_into_group.emit)
        self._list.layers_unparented.connect(self.layers_unparented.emit)
        self._list.mask_dropped_on_layer.connect(self.mask_dropped_on_layer.emit)
        self._list.adj_filter_dropped_on_layer.connect(self.adj_filter_dropped_on_layer.emit)
        self._list.clip_to_layer.connect(self.clip_to_layer.emit)
        self._list.layer_dropped_as_mask.connect(self.layer_dropped_as_mask.emit)
        self._list.delete_key_pressed.connect(self.delete_requested.emit)
        root.addWidget(self._list, 1)

        self._sep2 = h_separator()
        root.addWidget(self._sep2)

        self._toolbar = QWidget()
        tb_layout = QHBoxLayout(self._toolbar)
        tb_layout.setContentsMargins(4, 3, 4, 3)
        tb_layout.setSpacing(1)

        self._btn_new_layer = toolbar_btn(ico_new_layer(), "New layer", self.add_requested)
        tb_layout.addWidget(self._btn_new_layer)
        self._btn_styles = toolbar_btn(ico_fx(), "Layer styles", self.styles_requested)
        tb_layout.addWidget(self._btn_styles)
        self._btn_mask = toolbar_btn(ico_mask(), "Add mask", self.mask_requested)
        tb_layout.addWidget(self._btn_mask)

        self._adj_btn = QPushButton()
        self._adj_btn.setIcon(ico_adjustment())
        self._adj_btn.setIconSize(QSize(16, 16))
        self._adj_btn.setFixedSize(24, 24)
        self._adj_btn.setFlat(True)
        self._adj_btn.setToolTip("Add adjustment layer")
        
        self._adj_menu = QMenu(self)
        
        _ADJ_NAMES = [
            "Brightness/Contrast", "Levels", "Curves", "Exposure",
            "Vibrance", "Hue/Saturation", "White Balance", "Color Balance", "Recolor",
            "Split Toning", "Normals", "Black & White",
            "Photo Filter", "Gradient Map", "Selective Color", "Channel Mixer",
            "Invert", "Posterize", "Threshold",
        ]
        for adj_name in _ADJ_NAMES:
            action = self._adj_menu.addAction(adj_name)
            action.triggered.connect(
                lambda checked, n=adj_name: self.adjustment_layer_requested.emit(n),
            )
        self._adj_btn.clicked.connect(self._show_adj_menu)
        tb_layout.addWidget(self._adj_btn)

        self._filt_btn = QPushButton()
        self._filt_btn.setIcon(ico_filter())
        self._filt_btn.setIconSize(QSize(16, 16))
        self._filt_btn.setFixedSize(24, 24)
        self._filt_btn.setFlat(True)
        self._filt_btn.setToolTip("Add filter layer")
        
        self._filt_menu = QMenu(self)
        
        _FILTER_CATEGORIES = [
            ("Blur", ["Gaussian Blur", "Motion Blur", "Radial Blur", "Surface Blur", "Lens Blur"]),
            ("Sharpen", ["Sharpen", "Unsharp Mask", "Smart Sharpen"]),
            ("Noise", ["Add Noise", "Reduce Noise", "Dust & Scratches", "Median"]),
            ("Distort", ["Ripple", "Wave", "Twirl", "Pinch", "Perspective"]),
            ("Stylize", ["Emboss", "Find Edges", "Solarize", "Oil Paint"]),
            ("Render", ["Clouds", "Difference Clouds", "Lighting Effects"]),
        ]
        for cat_name, filters in _FILTER_CATEGORIES:
            sub = self._filt_menu.addMenu(cat_name)
            sub.setStyleSheet(self._filt_menu.styleSheet())
            for fname in filters:
                action = sub.addAction(fname)
                action.triggered.connect(
                    lambda checked, n=fname: self.filter_layer_requested.emit(n),
                )
        self._filt_btn.clicked.connect(self._show_filt_menu)
        tb_layout.addWidget(self._filt_btn)

        tb_layout.addStretch()

        self._btn_group = toolbar_btn(ico_folder(), "New group", self.group_requested)
        tb_layout.addWidget(self._btn_group)
        self._btn_duplicate = toolbar_btn(ico_duplicate(), "Duplicate layer", self.duplicate_requested)
        tb_layout.addWidget(self._btn_duplicate)
        self._btn_flatten = toolbar_btn(ico_grid(), "Flatten image", self.flatten_requested)
        tb_layout.addWidget(self._btn_flatten)
        self._btn_delete = toolbar_btn(ico_trash(), "Delete layer", self.delete_requested)
        tb_layout.addWidget(self._btn_delete)

        root.addWidget(self._toolbar)

    # ---- Filtering ---------------------------------------------------

    def _on_filter_text(self, text: str) -> None:
        self._filter.text = text
        self._apply_filter()

    def _on_filter_kind(self, _index: int) -> None:
        kind = self._kind_combo.currentData() or ""
        self._filter.kinds = {kind} if kind else set()
        self._apply_filter()

    def _apply_filter(self) -> None:
        """Rebuild the panel for the current query."""
        if self._doc is None:
            return
        self.refresh(self._doc, thumbnails=True, force=True)
        # Reordering against a filtered view is ambiguous -- the panel shows
        # a subset, so a drop position says nothing about where the hidden
        # layers should end up. Dragging is therefore disabled while a filter
        # is active; without this the reorder rebuilt the stack from the
        # visible rows alone and every non-matching layer was lost.
        active = self._filter.is_active
        self._list.reorder_enabled = not active
        if active:
            hits = match_count(self._doc, self._filter)
            total = len(self._doc.layers.layers)
            self._filter_status.setText(
                f"{hits} of {total} layers match — clear the filter to reorder"
                if hits else "No layers match")
            self._filter_status.setVisible(True)
        else:
            self._filter_status.setVisible(False)

    @property
    def filtering(self) -> bool:
        """True when the panel is showing a subset of the stack."""
        return self._filter.is_active

    @property
    def layer_filter(self) -> LayerFilter:
        return self._filter

    def clear_filter(self) -> None:
        self._search.clear()
        self._kind_combo.setCurrentIndex(0)
        self._filter.clear()
        self._apply_filter()

    def refresh(self, document: Document, *, thumbnails: bool = True,
                force: bool = False) -> None:
        """Sync the panel with *document*.

        *force* rebuilds the rows even when the structure is unchanged.
        Callers used to signal that by clearing ``_row_structure``, which
        silently failed when the *new* structure was also empty -- as it is
        when a filter matches nothing, so the panel kept showing every
        layer.
        """
        self._doc = document
        self._refreshing = True

        allowed = visible_layer_ids(document, self._filter)
        # While filtering, groups are shown expanded: collapsing away the
        # very rows the query selected would be perverse.
        collapsed = set() if allowed is not None else self._collapsed_groups
        display_order = self._build_display_order(
            document, collapsed, self._collapsed_masks, allowed=allowed)
        new_ids: list[str] = []
        new_structure: list[tuple] = []
        for entry in display_order:
            if len(entry) == 3:
                new_ids.append("__sep__")
                new_structure.append(("__sep__", entry[1]))
            else:
                new_ids.append(entry[0].id)
                new_structure.append((entry[0].id, entry[1], entry[0].parent_id))
        structure_changed = force or (new_structure != self._row_structure)

        if not structure_changed and not thumbnails:
            self._sync_row_states(document)
            self._sync_active(document)
            self._refreshing = False
            return

        if not structure_changed and thumbnails:
            # Structure intact — just update thumbnails + row state in place
            self._sync_row_states(document)
            self._sync_thumbnails(document)
            self._sync_active(document)
            self._refreshing = False
            return

        # Build children map for drag-drop circular detection
        children_map: dict[str, list[str]] = {}
        for layer in document.layers:
            kids = list(layer.children) + list(layer.mask_layers)
            if kids:
                children_map[layer.id] = kids
        self._list.set_children_map(children_map)

        self._list.clear()
        self._row_layer_ids = []
        self._row_structure = new_structure

        # Pre-compute which parents have children of each category (O(N) once)
        _parents_with_adj: set[str] = set()
        _parents_with_mask: set[str] = set()
        _parents_with_raster: set[str] = set()
        for cl in document.layers:
            if not cl.parent_id:
                continue
            if cl.layer_type in (LayerType.ADJUSTMENT, LayerType.FILTER):
                _parents_with_adj.add(cl.parent_id)
            elif cl.layer_type == LayerType.MASK:
                _parents_with_mask.add(cl.parent_id)
            else:
                _parents_with_raster.add(cl.parent_id)

        for entry in display_order:
            if len(entry) == 3:
                _, indent, _ = entry
                item = QListWidgetItem()
                item.setData(ROLE_LAYER_ID, "__sep__")
                item.setData(ROLE_IS_SEP, True)
                item.setSizeHint(QSize(0, SEP_HEIGHT))
                item.setFlags(Qt.ItemFlag.NoItemFlags)

                sep_widget = QWidget()
                sep_layout = QHBoxLayout(sep_widget)
                left_margin = 4 + indent * 16
                sep_layout.setContentsMargins(left_margin, 0, 4, 0)
                sep_layout.setSpacing(0)
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                theme_border = ThemeManager.instance().active_palette['border']
                line.setStyleSheet(render_qss("layers_panel_separator_line.qss", border=theme_border))
                line.setFixedHeight(1)
                sep_layout.addWidget(line)
                sep_widget.setStyleSheet(render_qss("layers_panel_separator_widget.qss"))

                self._list.addItem(item)
                self._list.setItemWidget(item, sep_widget)
                self._row_layer_ids.append("__sep__")
                continue

            layer, indent = entry
            is_group = layer.layer_type == LayerType.GROUP
            is_adjustment = layer.layer_type == LayerType.ADJUSTMENT
            is_filter = layer.layer_type == LayerType.FILTER
            is_text = layer.layer_type == LayerType.TEXT
            is_mask_layer = layer.layer_type == LayerType.MASK
            is_clipped = getattr(layer, 'clipping_mask', False)
            has_mask = layer.mask is not None or bool(layer.mask_layers)
            has_adj_children = layer.id in _parents_with_adj
            has_mask_children = layer.id in _parents_with_mask
            has_raster_children = layer.id in _parents_with_raster
            has_children = has_mask or has_mask_children or has_adj_children or has_raster_children
            is_collapsed = layer.id in self._collapsed_groups

            item = QListWidgetItem()
            item.setData(ROLE_LAYER_ID, layer.id)
            item.setData(ROLE_IS_GROUP, is_group)
            item.setData(ROLE_INDENT, indent)
            item.setData(ROLE_PARENT_ID, layer.parent_id or "")
            item.setData(ROLE_IS_MASK, is_mask_layer)
            item.setData(ROLE_IS_ADJ_FILTER, is_adjustment or is_filter)
            item.setData(ROLE_IS_CLIPPED, is_clipped)
            item.setSizeHint(QSize(0, ROW_HEIGHT))

            thumbnail = None
            if thumbnails and not is_adjustment and not is_filter and not is_text:
                if is_group:
                    thumbnail = make_group_thumbnail(document, layer)
                else:
                    thumbnail = make_thumbnail(layer)

            widget = LayerItemWidget(
                layer.id, layer.name, layer.visible, layer.locked,
                indent=indent, is_group=is_group, is_collapsed=is_collapsed,
                has_mask=has_mask, has_children=has_children,
                masks_collapsed=(layer.id in self._collapsed_masks),
                thumbnail=thumbnail,
                is_adjustment=is_adjustment, is_filter=is_filter,
                is_text=is_text, is_mask_layer=is_mask_layer,
                is_clipped=is_clipped,
            )
            widget.visibility_clicked.connect(self.visibility_toggled.emit)
            widget.lock_clicked.connect(self.lock_toggled.emit)
            widget.rename_finished.connect(self.rename_requested.emit)
            if is_group:
                widget.collapse_clicked.connect(self._on_collapse_toggled)
            elif has_children:
                widget.collapse_clicked.connect(self._on_mask_collapse_toggled)

            self._list.addItem(item)
            self._list.setItemWidget(item, widget)
            self._row_layer_ids.append(layer.id)

        self._sync_active(document)
        self._refreshing = False

    def _sync_row_states(self, document: Document) -> None:
        layers_by_id = {layer.id: layer for layer in document.layers}
        for row in range(self._list.count()):
            item = self._list.item(row)
            if not item:
                continue
            lid = item.data(ROLE_LAYER_ID)
            layer = layers_by_id.get(lid)
            if not layer:
                continue
            widget = self._list.itemWidget(item)
            if isinstance(widget, LayerItemWidget):
                widget.update_state(layer.visible, layer.locked)

    def _sync_thumbnails(self, document: Document) -> None:
        """Update thumbnails on existing row widgets without rebuilding."""
        layers_by_id = {layer.id: layer for layer in document.layers}
        for row in range(self._list.count()):
            item = self._list.item(row)
            if not item:
                continue
            lid = item.data(ROLE_LAYER_ID)
            layer = layers_by_id.get(lid)
            if not layer:
                continue
            if layer.layer_type in (LayerType.ADJUSTMENT, LayerType.FILTER, LayerType.TEXT):
                continue
            widget = self._list.itemWidget(item)
            if not isinstance(widget, LayerItemWidget):
                continue
            thumb_label = getattr(widget, '_thumb_label', None)
            if thumb_label is None:
                continue
            if layer.layer_type == LayerType.GROUP:
                pm = make_group_thumbnail(document, layer)
            else:
                pm = make_thumbnail(layer)
            thumb_label.setPixmap(pm)

    def refresh_controls_only(self, document: Document) -> None:
        self._doc = document
        self._refreshing = True
        self._sync_active(document)
        self._refreshing = False

    def _sync_active(self, document: Document) -> None:
        from .icons import icon_lock

        active = document.layers.active_layer
        if active is None:
            # No active layer — clear the visual selection
            self._list.blockSignals(True)
            self._list.clearSelection()
            self._list.setCurrentRow(-1)
            self._list.blockSignals(False)
            return
        if active:
            sel_indices = document.layers.selected_indices
            self._list.blockSignals(True)
            if len(sel_indices) > 1:
                # Multi-selection: set current row first (clears selection
                # in ExtendedSelection mode), then re-add all selected rows.
                for row in range(self._list.count()):
                    it = self._list.item(row)
                    if it and it.data(ROLE_LAYER_ID) == active.id:
                        self._list.setCurrentRow(row)
                        break
                # Now add all multi-selected rows back
                for si in sel_indices:
                    if 0 <= si < len(document.layers.layers):
                        lid = document.layers.layers[si].id
                        try:
                            r = self._row_layer_ids.index(lid)
                            item = self._list.item(r)
                            if item:
                                item.setSelected(True)
                        except ValueError:
                            pass
            else:
                # Single selection
                for row in range(self._list.count()):
                    it = self._list.item(row)
                    if it and it.data(ROLE_LAYER_ID) == active.id:
                        self._list.setCurrentRow(row)
                        break
            self._list.blockSignals(False)

            op_val = int(active.opacity * 100)
            self._opacity_spin.blockSignals(True)
            self._opacity_spin.setValue(op_val)
            self._opacity_spin.blockSignals(False)
            self._opacity_slider.blockSignals(True)
            self._opacity_slider.setValue(op_val)
            self._opacity_slider.blockSignals(False)
            blend_idx = self._blend_combo.findData(active.blend_mode)
            if blend_idx >= 0:
                self._blend_combo.blockSignals(True)
                self._blend_combo.setCurrentIndex(blend_idx)
                self._blend_combo.blockSignals(False)
            self._lock_btn.setIcon(icon_lock(active.locked))

    def selected_layer_ids(self) -> list[str]:
        ids: list[str] = []
        for item in self._list.selectedItems():
            lid = item.data(ROLE_LAYER_ID)
            if lid:
                ids.append(lid)
        return ids

    def row_layer_ids(self) -> list[str]:
        return list(self._row_layer_ids)

    @staticmethod
    def _build_display_order(
        document: Document, collapsed: set[str],
        masks_collapsed: set[str] | None = None,
        allowed: set[str] | None = None,
    ) -> list[tuple]:
        if masks_collapsed is None:
            masks_collapsed = set()
        layers = list(document.layers)
        if allowed is not None:
            layers = [l for l in layers if l.id in allowed]
        children_of: dict[str, list] = {}
        mask_children_of: dict[str, list] = {}
        adj_children_of: dict[str, list] = {}
        for layer in layers:
            if layer.parent_id:
                if layer.layer_type == LayerType.MASK:
                    mask_children_of.setdefault(layer.parent_id, []).append(layer)
                elif layer.layer_type in (LayerType.ADJUSTMENT, LayerType.FILTER):
                    adj_children_of.setdefault(layer.parent_id, []).append(layer)
                else:
                    children_of.setdefault(layer.parent_id, []).append(layer)

        def _emit_children(lid, indent, result, is_group, group_collapsed):
            has_masks = lid in mask_children_of
            has_adj = lid in adj_children_of
            has_raster = lid in children_of
            masks_hidden = lid in masks_collapsed

            if is_group and group_collapsed:
                return

            if has_masks and not masks_hidden:
                for child in reversed(mask_children_of[lid]):
                    result.append((child, indent))

            if has_masks and has_adj and not masks_hidden:
                result.append((None, indent, "sep"))

            if has_adj and not masks_hidden:
                for child in reversed(adj_children_of[lid]):
                    result.append((child, indent))

            if has_raster and not masks_hidden:
                if (has_masks or has_adj) and not masks_hidden:
                    result.append((None, indent, "sep"))
                for child in reversed(children_of[lid]):
                    result.append((child, indent))
                    child_is_group = child.layer_type == LayerType.GROUP
                    child_collapsed = child_is_group and child.id in collapsed
                    _emit_children(child.id, indent + 1, result,
                                   is_group=child_is_group, group_collapsed=child_collapsed)

        result: list[tuple] = []
        for layer in reversed(layers):
            if layer.parent_id is not None:
                continue
            is_group = layer.layer_type == LayerType.GROUP
            result.append((layer, 0))
            group_collapsed = is_group and layer.id in collapsed
            _emit_children(layer.id, 1, result, is_group, group_collapsed)
        return result

    def _show_adj_menu(self) -> None:
        pos = self._adj_btn.mapToGlobal(self._adj_btn.rect().topLeft())
        pos.setY(pos.y() - self._adj_menu.sizeHint().height())
        self._adj_menu.exec(pos)

    def _show_filt_menu(self) -> None:
        pos = self._filt_btn.mapToGlobal(self._filt_btn.rect().topLeft())
        pos.setY(pos.y() - self._filt_menu.sizeHint().height())
        self._filt_menu.exec(pos)

    def _on_collapse_toggled(self, group_id: str) -> None:
        if group_id in self._collapsed_groups:
            self._collapsed_groups.discard(group_id)
        else:
            self._collapsed_groups.add(group_id)
        if self._doc:
            self.refresh(self._doc)

    def _on_mask_collapse_toggled(self, layer_id: str) -> None:
        if layer_id in self._collapsed_masks:
            self._collapsed_masks.discard(layer_id)
        else:
            self._collapsed_masks.add(layer_id)
        if self._doc:
            self.refresh(self._doc)

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        layer_id = item.data(ROLE_LAYER_ID)
        if self._doc and layer_id:
            layer = self._doc.layers.get(layer_id)
            if layer and layer.layer_type == LayerType.ADJUSTMENT:
                self.edit_adjustment_requested.emit(layer_id)
                return
            if layer and layer.layer_type == LayerType.FILTER:
                self.edit_filter_requested.emit(layer_id)
                return
        widget = self._list.itemWidget(item)
        if isinstance(widget, LayerItemWidget):
            widget.start_rename()

    def _on_row_changed(self, row: int) -> None:
        if self._refreshing or row < 0 or not self._doc:
            return
        if 0 <= row < len(self._row_layer_ids):
            lid = self._row_layer_ids[row]
            for i, layer in enumerate(self._doc.layers):
                if layer.id == lid:
                    self.layer_selected.emit(i)
                    break

    def _on_selection_changed(self) -> None:
        """Fired when the user (de)selects rows in ExtendedSelection mode."""
        if self._refreshing or not self._doc:
            return
        ids = self.selected_layer_ids()
        self.multi_selection_changed.emit(ids)
        # Also update the current row's controls
        row = self._list.currentRow()
        layer = self._layer_for_row(row)
        if layer:
            op_val = int(layer.opacity * 100)
            self._opacity_spin.blockSignals(True)
            self._opacity_spin.setValue(op_val)
            self._opacity_spin.blockSignals(False)
            self._opacity_slider.blockSignals(True)
            self._opacity_slider.setValue(op_val)
            self._opacity_slider.blockSignals(False)
            blend_idx = self._blend_combo.findData(layer.blend_mode)
            if blend_idx >= 0:
                self._blend_combo.blockSignals(True)
                self._blend_combo.setCurrentIndex(blend_idx)
                self._blend_combo.blockSignals(False)
            from .icons import icon_lock
            self._lock_btn.setIcon(icon_lock(layer.locked))

    def _layer_for_row(self, row: int):
        if not self._doc or row < 0 or row >= len(self._row_layer_ids):
            return None
        lid = self._row_layer_ids[row]
        return self._doc.layers.get(lid)

    def _on_opacity_changed(self, value: int) -> None:
        if self._refreshing:
            return
        self._opacity_slider.blockSignals(True)
        self._opacity_slider.setValue(value)
        self._opacity_slider.blockSignals(False)
        self.opacity_changed.emit(value / 100.0)

    def _on_opacity_slider_changed(self, value: int) -> None:
        if self._refreshing:
            return
        self._opacity_spin.blockSignals(True)
        self._opacity_spin.setValue(value)
        self._opacity_spin.blockSignals(False)
        self.opacity_changed.emit(value / 100.0)

    def _on_header_lock_clicked(self) -> None:
        item = self._list.currentItem()
        if item:
            layer_id = item.data(ROLE_LAYER_ID)
            if layer_id:
                self.lock_toggled.emit(layer_id)

    def _on_blend_changed(self, idx: int) -> None:
        if self._refreshing:
            return
        mode = self._blend_combo.itemData(idx)
        if mode is not None:
            self.blend_mode_changed.emit(mode)

    def toggle_visibility_for_selected(self) -> None:
        item = self._list.currentItem()
        if item:
            layer_id = item.data(ROLE_LAYER_ID)
            if layer_id:
                self.visibility_toggled.emit(layer_id)
