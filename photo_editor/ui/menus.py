"""Application menu bar with all top-level menus.

Shortcuts are sourced from :class:`ShortcutManager` so they update
automatically when the user changes preset or edits a binding.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import QLabel, QMenuBar

from .shortcut_manager import ShortcutManager
from .styles import render_qss


class EditorMenuBar(QMenuBar):
    """Full menu bar mirroring Photoshop-style organisation."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.actions_map: dict[str, QAction] = {}
        self._mgr = ShortcutManager.instance()
        
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logo_path = os.path.join(base_dir, "assets", "app", "logo.svg")
        self._logo_label = QLabel()
        self._logo_label.setPixmap(QIcon(logo_path).pixmap(24, 24))
        self._logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._logo_label.setContentsMargins(12, 0, 8, 0)
        self.setCornerWidget(self._logo_label, Qt.Corner.TopLeftCorner)

        self._build()
        self._mgr.shortcuts_changed.connect(self._refresh_shortcuts)
        self.setNativeMenuBar(False)
        from .theme import ThemeManager
        ThemeManager.instance().theme_changed.connect(self._apply_theme)
        self._apply_theme(ThemeManager.instance().active_palette)

    def _apply_theme(self, palette: dict) -> None:
        self.setStyleSheet(render_qss("menu_bar.qss", palette))

    def _build(self) -> None:
        self._file_menu()
        self._edit_menu()
        self._image_menu()
        self._layer_menu()
        self._select_menu()
        self._filter_menu()
        self._view_menu()
        self._help_menu()

    # ---- File ---------------------------------------------------------------

    def _file_menu(self) -> None:
        m = self.addMenu("&File")
        self._add(m, "new", "&New Project…")
        self._add(m, "open", "&Open Image…")
        self._add(m, "import_svg", "Import &SVG…")
        self._add(m, "place_image", "&Place Image as Layer…")
        m.addSeparator()
        self._add(m, "save", "&Save Project")
        self._add(m, "save_as", "Save Project &As…")
        m.addSeparator()
        self._add(m, "export", "&Export…")
        self._add(m, "export_svg", "Export S&VG…")
        self._add(m, "export_pdf", "Export &PDF…")
        m.addSeparator()
        self._add(m, "quit", "&Quit")

    def _edit_menu(self) -> None:
        m = self.addMenu("&Edit")
        self._add(m, "undo", "&Undo")
        self._add(m, "redo", "&Redo")
        m.addSeparator()
        self._add(m, "cut", "Cu&t")
        self._add(m, "copy", "&Copy")
        self._add(m, "paste", "&Paste")
        m.addSeparator()
        self._add(m, "delete_sel", "&Delete")
        self._add(m, "fill_fg", "Fill with &Foreground Color")
        self._add(m, "fill_bg", "Fill with &Background Color")
        m.addSeparator()
        self._add(m, "edit_document_settings", "&Document Settings…")

    def _image_menu(self) -> None:
        m = self.addMenu("&Image")
        self._add(m, "resize_canvas", "Canvas &Size…")
        self._add(m, "resize_image", "Image &Size…")
        m.addSeparator()
        self._add(m, "rotate_cw", "Rotate 90° CW")
        self._add(m, "rotate_ccw", "Rotate 90° CCW")
        self._add(m, "flip_h", "Flip &Horizontal")
        self._add(m, "flip_v", "Flip &Vertical")

    def _layer_menu(self) -> None:
        m = self.addMenu("&Layer")
        self._add(m, "new_layer", "&New Layer")
        self._add(m, "new_vector_layer", "New &Vector Layer")
        self._add(m, "new_group", "New &Group")
        self._add(m, "dup_layer", "&Duplicate Layer")
        self._add(m, "del_layer", "De&lete Layer")
        m.addSeparator()
        self._add(m, "merge_down", "Merge &Down")
        self._add(m, "flatten", "&Flatten Image")
        m.addSeparator()
        mask_sub = m.addMenu("Mas&k")
        self._add(mask_sub, "add_mask", "Add Layer &Mask (White)")
        self._add(mask_sub, "add_mask_black", "Add Layer Mask (&Black)")
        self._add(mask_sub, "add_mask_standalone", "Add &Standalone Mask Layer")
        mask_sub.addSeparator()
        self._add(mask_sub, "remove_mask_layer", "&Remove Mask Layer")
        self._add(mask_sub, "apply_mask_layer", "Appl&y Mask Layer")
        mask_sub.addSeparator()
        self._add(mask_sub, "invert_mask_layer", "&Invert Mask Layer")
        self._add(mask_sub, "convert_to_mask", "&Convert Layer to Mask")
        m.addSeparator()
        self._add(m, "toggle_vis", "Toggle &Visibility")

    def _select_menu(self) -> None:
        m = self.addMenu("&Select")
        self._add(m, "select_all", "Select &All")
        self._add(m, "deselect", "&Deselect")
        self._add(m, "invert_sel", "&Invert Selection")
        m.addSeparator()
        self._add(m, "duplicate_sel", "Duplicate &Layer via Selection")
        self._add(m, "selection_to_mask", "Selection to &Mask Layer")
        m.addSeparator()
        self._add(m, "feather_sel", "&Feather…")
        self._add(m, "grow_sel", "&Grow…")
        self._add(m, "shrink_sel", "S&hrink…")

    def _filter_menu(self) -> None:
        m = self.addMenu("F&ilter")
        for cat, items in [
            ("Blur", ["Gaussian Blur", "Motion Blur", "Radial Blur", "Surface Blur", "Lens Blur"]),
            ("Sharpen", ["Sharpen", "Unsharp Mask", "Smart Sharpen"]),
            ("Noise", ["Add Noise", "Reduce Noise", "Dust & Scratches", "Median"]),
            ("Distort", ["Ripple", "Wave", "Twirl", "Pinch", "Perspective"]),
            ("Stylize", ["Emboss", "Find Edges", "Solarize", "Oil Paint"]),
            ("Render", ["Clouds", "Difference Clouds", "Lighting Effects"]),
        ]:
            sub = m.addMenu(cat)
            for name in items:
                key = f"filter_{name.lower().replace(' ', '_').replace('&', '')}"
                self._add(sub, key, name)

    def _view_menu(self) -> None:
        m = self.addMenu("&View")
        self._add(m, "zoom_in", "Zoom &In")
        self._add(m, "zoom_out", "Zoom &Out")
        self._add(m, "zoom_fit", "Zoom to &Fit")
        self._add(m, "zoom_100", "Actual &Pixels")
        m.addSeparator()
        self._add(m, "toggle_grid", "Show &Grid")
        self._add(m, "toggle_rulers", "Show &Rulers")
        self._add(m, "toggle_guides", "Show G&uides")
        self._add(m, "toggle_snapping", "S&nap to Objects")
        m.addSeparator()
        
        from .theme import THEMES
        theme_sub = m.addMenu("&Theme")
        for key in THEMES.keys():
            action_key = f"theme_{key.lower().replace(' ', '_')}"
            self._add(theme_sub, action_key, key)

    def _help_menu(self) -> None:
        m = self.addMenu("&Help")
        self._add(m, "keyboard_shortcuts", "&Keyboard Shortcuts…")
        m.addSeparator()
        self._add(m, "about", "&About")

    # ---- Helper -------------------------------------------------------------

    def _add(self, menu, key: str, text: str) -> QAction:
        action = QAction(text, self)
        shortcut = self._mgr.binding(key)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        menu.addAction(action)
        self.actions_map[key] = action
        return action

    def _refresh_shortcuts(self) -> None:
        """Re-apply all shortcut key-sequences from the manager."""
        for key, action in self.actions_map.items():
            shortcut = self._mgr.binding(key)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            else:
                action.setShortcut(QKeySequence())
