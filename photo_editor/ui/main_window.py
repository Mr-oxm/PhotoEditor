"""Main application window — assembles all panels, menus, and canvas."""

from __future__ import annotations

import math
import sys
from typing import Callable

from PySide6.QtCore import Qt, QPointF, QSignalBlocker, QTimer
from PySide6.QtWidgets import (
    QDockWidget, QMainWindow, QMessageBox, QScrollBar, QWidget, QStackedWidget,
)

from ..commands.base import Command
from ..core.brush_engine import BrushManager
from ..core.document import Document
from ..core.enums import BlendMode, ToolType
from ..engine.render_pipeline import RenderPipeline
from ..engine.renderer import RenderScheduler
from .canvas_view import CanvasView
from .app_signals import AppSignals
from .document_session import DocumentSession
from .file_tab_bar import FileTabBar
from .menus import EditorMenuBar
from .panels.brushes_panel import BrushesPanel
from .panels.color_panel import ColorPanel
from .panels.history_panel import HistoryPanel
from .panels.layers_panel import LayersPanel
from .panels.properties_panel import PropertiesPanel
from .panels.transform_panel import TransformPanel
from .panels.channels_panel import ChannelsPanel
from .controllers import DocumentController
from .shortcut_manager import ShortcutManager
from .status_bar import EditorStatusBar
from .theme import ThemeManager, THEMES
from .tool_manager import ToolManager
from .toolbar import EditorToolbar
from .icons import app_icon
from .styles import render_qss

_IMG_FLT = "Images (*.png *.jpg *.jpeg *.webp *.tiff *.tif *.bmp)"


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self, dev_mode: bool = False) -> None:
        super().__init__()
        self._dev_mode = dev_mode or ("--dev" in sys.argv)
        self.setWindowTitle("Basera")
        self.resize(1440, 900)
        self.setWindowIcon(app_icon())
        
        ThemeManager.instance().theme_changed.connect(
            lambda _: self.setStyleSheet(THEMES[ThemeManager.instance().active_theme_name])
        )
        self.setStyleSheet(THEMES["Dark"])
        self.setAcceptDrops(True)

        self._doc: Document | None = None
        self._app_signals = AppSignals()
        self._pipeline = RenderPipeline()
        self._render_scheduler = RenderScheduler(
            self._pipeline,
            interval_ms=16,          # ~60 fps ceiling
            preview_max_size=2048,   # replaced per-frame by _sync_preview_level
        )
        self._tools = ToolManager()

        # Brush manager — load ABR files from assets
        self._brush_mgr = BrushManager.instance()
        import os
        _assets_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets", "brushes"
        )
        self._brush_mgr.load_brushes_dir(_assets_dir)

        # Multi-document tracking
        self._document_session: DocumentSession | None = None

        # Render throttle — handled by RenderScheduler (~30 fps, off UI thread)
        # Deferred panel refresh — coalesced to avoid rebuilding panels
        # dozens of times per second during interactive operations.
        self._panel_refresh_timer = QTimer(self)
        self._panel_refresh_timer.setInterval(200)  # 5 fps panel updates
        self._panel_refresh_timer.setSingleShot(True)
        self._panel_refresh_timer.timeout.connect(self._do_deferred_panel_refresh)
        self._panel_refresh_pending = False

        # Autosave. Only viable now that saving a large project takes
        # seconds on a worker rather than half a minute on the UI thread.
        from ..utils.autosave import AutosaveManager, purge_stale_markers
        purge_stale_markers()
        self._autosave = AutosaveManager()
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(60_000)   # check once a minute
        self._autosave_timer.timeout.connect(self._run_autosave)
        self._autosave_timer.start()
        
        # Track whether text editing is active to manage conflicting shortcuts
        self._text_editing_active = False

        self._shortcut_mgr = ShortcutManager.instance()

        self._build_ui()
        self._document_ctrl = DocumentController()
        self._document_ctrl.wire(self)
        from .controllers.selection_ctrl import SelectionController
        self._selection_ctrl = SelectionController()
        self._selection_ctrl.wire(self)
        from .controllers.filter_ctrl import FilterController
        self._filter_ctrl = FilterController()
        self._filter_ctrl.wire(self)
        from .controllers.transform_ctrl import TransformController
        self._transform_ctrl = TransformController()
        self._transform_ctrl.wire(self)
        from .controllers.crop_ctrl import CropController
        self._crop_ctrl = CropController()
        self._crop_ctrl.wire(self)
        from .controllers.vector_ctrl import VectorController
        self._vector_ctrl = VectorController()
        self._vector_ctrl.wire(self)
        from .controllers.gradient_ctrl import GradientController
        self._gradient_ctrl = GradientController()
        self._gradient_ctrl.wire(self)
        from .controllers.text_ctrl import TextController
        self._text_ctrl = TextController()
        self._text_ctrl.wire(self)
        from .controllers.layer_ctrl import LayerController
        self._layer_ctrl = LayerController()
        self._layer_ctrl.wire(self)
        from .controllers.view_ctrl import ViewController
        self._view_ctrl = ViewController()
        self._view_ctrl.wire(self)
        from .controllers.tool_ctrl import ToolController
        self._tool_ctrl = ToolController()
        self._tool_ctrl.wire(self)
        from .controllers.color_ctrl import ColorController
        self._color_ctrl = ColorController()
        self._color_ctrl.wire(self)
        from .controllers.canvas_ctrl import CanvasController
        self._canvas_ctrl = CanvasController()
        self._canvas_ctrl.wire(self)
        from .controllers.shortcut_ctrl import ShortcutController
        self._shortcut_ctrl = ShortcutController()
        self._shortcut_ctrl.wire(self)
        from .controllers.drop_ctrl import DropController
        self._drop_ctrl = DropController()
        self._drop_ctrl.wire(self)
        self._wire_menus()
        self._wire_panels()
        self._wire_canvas()
        self._wire_file_tabs()
        self._render_scheduler.render_ready.connect(self._on_render_ready)
        self._render_scheduler.render_error.connect(self._on_render_error)
        self._wire_app_signals()
        # Shortcuts wired by ShortcutController

        # Wire brush system
        self._brushes_panel.set_brush_manager(self._brush_mgr)
        self._props_panel.brush_bar.set_brush_manager(self._brush_mgr)
        self._brush_mgr.brush_changed.connect(self._on_brush_preset_changed)

        # Dev mode: auto-create 1080p project, skip welcome screen
        if self._dev_mode:
            self._document_ctrl.new_document(1920, 1080)
        else:
            # Start with welcome screen
            self._show_welcome_screen()
            self._set_editor_visible(False)

    # ---- UI assembly --------------------------------------------------------

    def _build_ui(self) -> None:
        from PySide6.QtWidgets import QVBoxLayout, QGridLayout, QToolBar
        from .widgets.rulers import HorizontalRuler, VerticalRuler, RulerCorner, Guide, RULER_SIZE
        from .welcome_screen import WelcomeScreen
        
        self._menu = EditorMenuBar(self)
        self.setMenuBar(self._menu)

        # Properties panel as a toolbar directly under the menu bar
        self._props_panel = PropertiesPanel()
        self._props_toolbar = QToolBar("Properties", self)
        self._props_toolbar.setMovable(False)
        self._props_toolbar.setFloatable(False)
        self._props_toolbar.addWidget(self._props_panel)
        from .theme import ThemeManager
        ThemeManager.instance().theme_changed.connect(self._on_theme_changed)
        self._on_theme_changed(ThemeManager.instance().active_palette)
        
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._props_toolbar)

        self._toolbar = EditorToolbar(self)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self._toolbar)
        
        # Central widget: file tabs + rulers + canvas
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        
        # File tab bar at the top of the central area
        self._file_tabs = FileTabBar()
        self._document_session = DocumentSession(self._file_tabs)
        central_layout.addWidget(self._file_tabs)
        
        # Grid: rulers + canvas
        ruler_grid = QWidget()
        grid = QGridLayout(ruler_grid)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self._ruler_corner = RulerCorner()
        self._h_ruler = HorizontalRuler()
        self._v_ruler = VerticalRuler()
        self._canvas = CanvasView(self)
        self._canvas.set_tool_manager_ref(self._tools)
        self._h_scrollbar = QScrollBar(Qt.Orientation.Horizontal, self)
        self._v_scrollbar = QScrollBar(Qt.Orientation.Vertical, self)
        self._scroll_corner = QWidget(self)

        grid.addWidget(self._ruler_corner, 0, 0)
        grid.addWidget(self._h_ruler, 0, 1)
        grid.addWidget(self._v_ruler, 1, 0)
        grid.addWidget(self._canvas, 1, 1)
        grid.addWidget(self._v_scrollbar, 1, 2)
        grid.addWidget(self._h_scrollbar, 2, 1)
        grid.addWidget(self._scroll_corner, 2, 2)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(1, 1)

        # Rulers default visible
        self._rulers_visible = True
        self._guides: list = []
        # Smart snapping while dragging layers (View > Snap to Objects).
        self._snap_enabled = True

        # Ruler/guide signals wired by ViewController

        central_layout.addWidget(ruler_grid, 1)

        # Stacked widget: 0 = welcome screen, 1 = editor
        self._stacked = QStackedWidget()

        self._welcome = WelcomeScreen()
        self._welcome.new_project_requested.connect(self._on_welcome_new)
        self._welcome.open_image_requested.connect(self._on_welcome_open_image)
        self._welcome.open_basera_requested.connect(self._on_welcome_open_basera)
        self._welcome.recent_project_selected.connect(self._on_welcome_recent_selected)

        self._stacked.addWidget(self._welcome)   # index 0
        self._stacked.addWidget(central)          # index 1

        self.setCentralWidget(self._stacked)
        
        # Dock panels on the sides
        # Ensure all dock tabs are at the top, not bottom
        from PySide6.QtWidgets import QTabWidget
        self.setTabPosition(Qt.DockWidgetArea.AllDockWidgetAreas, QTabWidget.TabPosition.North)
        
        # Force single tabs to be visible and allow grouped dragging by their tabs
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks |
            QMainWindow.DockOption.AllowNestedDocks |
            QMainWindow.DockOption.AllowTabbedDocks |
            QMainWindow.DockOption.GroupedDragging |
            QMainWindow.DockOption.ForceTabbedDocks
        )

        self._layers_panel = LayersPanel()
        self._layers_dock = self._dock(self._layers_panel, "Layers", Qt.DockWidgetArea.RightDockWidgetArea)
        
        self._color_panel = ColorPanel()
        self._color_dock = self._dock(self._color_panel, "Color", Qt.DockWidgetArea.RightDockWidgetArea)
        
        self._transform_panel = TransformPanel()
        self._transform_dock = self._dock(self._transform_panel, "Transform", Qt.DockWidgetArea.RightDockWidgetArea)

        self._history_panel = HistoryPanel()
        self._history_dock = self._dock(self._history_panel, "History", Qt.DockWidgetArea.RightDockWidgetArea)

        self._brushes_panel = BrushesPanel()
        self._brushes_dock = self._dock(self._brushes_panel, "Brushes", Qt.DockWidgetArea.RightDockWidgetArea)
        
        self._channels_panel = ChannelsPanel()
        self._channels_dock = self._dock(self._channels_panel, "Channels", Qt.DockWidgetArea.RightDockWidgetArea)

        # Tabify top group
        self.tabifyDockWidget(self._layers_dock, self._color_dock)
        self.tabifyDockWidget(self._layers_dock, self._transform_dock)

        # Tabify bottom group
        self.tabifyDockWidget(self._history_dock, self._brushes_dock)
        self.tabifyDockWidget(self._history_dock, self._channels_dock)

        # Ensure correct active tabs
        self._layers_dock.raise_()
        self._history_dock.raise_()

        # Set vertical height of top and bottom panel groups to 50/50 by default
        self.resizeDocks([self._layers_dock, self._history_dock], [500, 500], Qt.Orientation.Vertical)

        self._status = EditorStatusBar(self)
        self.setStatusBar(self._status)

    def _dock(self, widget, title: str, area) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setWidget(widget)
        # Remove the panel header (title bar) since it's already in the tab
        from PySide6.QtWidgets import QWidget
        d.setTitleBarWidget(QWidget())
        self.addDockWidget(area, d)
        return d

    # ---- Wiring: menus ------------------------------------------------------

    def _wire_menus(self) -> None:
        a = self._menu.actions_map
        # Flip/rotate wired by TransformController, filter_* by FilterController
        if "keyboard_shortcuts" in a:
            a["keyboard_shortcuts"].triggered.connect(
                self._shortcut_ctrl._on_keyboard_shortcuts
            )
        if "about" in a:
            a["about"].triggered.connect(self._on_about)

    # ---- Wiring: panels -----------------------------------------------------

    def _wire_panels(self) -> None:
        # Toolbar and props panel wired by ToolController, color panel by ColorController
        # text/gradient/align/vector wired by TextController, GradientController,
        # TransformController, VectorController
        self._transform_panel.value_changed.connect(lambda: self._refresh(invalidate=True))
        self._channels_panel.value_changed.connect(lambda: self._refresh(invalidate=True))

    # ---- Wiring: file tabs --------------------------------------------------

    def _wire_file_tabs(self) -> None:
        pass  # Tab signals wired by DocumentController

    def _wire_app_signals(self) -> None:
        self._app_signals.selection_overlay_requested.connect(
            self._selection_ctrl.update_selection_overlay
        )
        self._app_signals.transform_box_requested.connect(
            self._transform_ctrl.update_transform_box
        )
        self._app_signals.properties_panel_requested.connect(
            self._tool_ctrl.update_properties_panel
        )
        self._app_signals.vector_bool_state_requested.connect(
            self._vector_ctrl.refresh_bool_state
        )
        self._app_signals.rulers_update_requested.connect(
            self._view_ctrl.update_rulers
        )
        self._app_signals.history_refresh_requested.connect(self._refresh_history_panel)
        self._app_signals.canvas_update_requested.connect(self._canvas.update)
        self._app_signals.brush_cursor_requested.connect(
            self._tool_ctrl.update_brush_cursor
        )
        self._app_signals.transform_panel_refresh_requested.connect(
            self._refresh_transform_panel
        )
        self._app_signals.channels_panel_refresh_requested.connect(
            self._refresh_channels_panel
        )
        self._app_signals.duplicate_selection_requested.connect(
            self._selection_ctrl.on_duplicate_selection
        )
        self._app_signals.clone_preview_requested.connect(
            self._tool_ctrl.update_clone_preview
        )
        self._app_signals.adjustment_layer_requested.connect(
            self._filter_ctrl.on_add_adjustment_layer
        )
        self._app_signals.edit_adjustment_requested.connect(
            self._filter_ctrl.on_edit_adjustment_layer
        )
        self._app_signals.filter_layer_requested.connect(
            self._filter_ctrl.on_add_filter_layer
        )
        self._app_signals.edit_filter_requested.connect(
            self._filter_ctrl.on_edit_filter_layer
        )
        self._app_signals.text_overlay_requested.connect(
            self._text_ctrl.update_overlay
        )
        self._app_signals.text_hover_cursor_requested.connect(
            self._text_ctrl.update_hover_cursor
        )
        self._app_signals.tool_selection_requested.connect(
            self._toolbar.select_tool
        )
        self._app_signals.text_editing_shortcuts_requested.connect(
            self._shortcut_ctrl.update_text_editing_shortcuts
        )

    # ---- Wiring: canvas -----------------------------------------------------

    def _wire_canvas(self) -> None:
        self._canvas.cursor_moved.connect(self._status.set_cursor_pos)
        # Canvas input wired by CanvasController
        self._canvas.view_changed.connect(self._on_canvas_view_changed)
        self._h_scrollbar.valueChanged.connect(self._on_h_scrollbar_changed)
        self._v_scrollbar.valueChanged.connect(self._on_v_scrollbar_changed)
        self._status.zoom_to_mouse_changed.connect(self._canvas.set_zoom_to_mouse)
        self._status.zoom_to_mouse = self._canvas.zoom_to_mouse
        # Guide drag wired by ViewController

    def _sync_preview_level(self) -> None:
        """Match the render resolution to what the screen actually shows.

        Compositing a 4K document at 4K while displaying it at 40% zoom does
        six times the necessary work. Conversely, rendering a fixed-size
        preview would go soft the moment the user zooms in. So the target is
        simply the document's on-screen size: never fewer pixels than are
        displayed, never many more.
        """
        if not self._doc:
            return
        longest = max(self._doc.width, self._doc.height)
        target = int(longest * self._canvas.zoom)
        # Clamp up to a floor so a very zoomed-out view still has enough
        # detail for the layer panel and for zooming back in smoothly.
        target = max(512, min(target, longest))
        self._render_scheduler.set_preview_max_size(target)
        # Only composite what is on screen. At 100% zoom a 4K document
        # shows a fraction of its pixels; rendering the rest is waste.
        self._render_scheduler.set_roi(self._canvas.visible_doc_rect())

    def _on_canvas_view_changed(self) -> None:
        self._view_ctrl.update_rulers()
        self._status.set_zoom(self._canvas.zoom)
        self._sync_canvas_scrollbars()
        self._sync_preview_level()

    def _sync_canvas_scrollbars(self) -> None:
        span_x, span_y = self._canvas.scrollable_span()
        limit_x = span_x / 2.0
        limit_y = span_y / 2.0
        value_x = int(round(limit_x - self._canvas.pan.x())) if span_x > 0 else 0
        value_y = int(round(limit_y - self._canvas.pan.y())) if span_y > 0 else 0

        with QSignalBlocker(self._h_scrollbar):
            self._h_scrollbar.setRange(0, max(0, int(math.ceil(span_x))))
            self._h_scrollbar.setPageStep(max(1, self._canvas.width()))
            self._h_scrollbar.setSingleStep(24)
            self._h_scrollbar.setValue(value_x)

        with QSignalBlocker(self._v_scrollbar):
            self._v_scrollbar.setRange(0, max(0, int(math.ceil(span_y))))
            self._v_scrollbar.setPageStep(max(1, self._canvas.height()))
            self._v_scrollbar.setSingleStep(24)
            self._v_scrollbar.setValue(value_y)

    def _on_h_scrollbar_changed(self, value: int) -> None:
        span_x, _ = self._canvas.scrollable_span()
        limit_x = span_x / 2.0
        self._canvas.set_pan(QPointF(limit_x - value, self._canvas.pan.y()))

    def _on_v_scrollbar_changed(self, value: int) -> None:
        _, span_y = self._canvas.scrollable_span()
        limit_y = span_y / 2.0
        self._canvas.set_pan(QPointF(self._canvas.pan.x(), limit_y - value))

    def _run_autosave(self) -> None:
        """Autosave every open document that has unsaved changes.

        Runs on a worker so a large project does not stall the UI, and only
        touches documents that actually changed, so an idle session costs
        nothing.
        """
        session = getattr(self, "_document_session", None)
        if session is None or getattr(self, "_autosave_busy", False):
            return
        pending = []
        for i in range(len(session)):
            entry = session.entry_at(i)
            if entry is None:
                continue
            doc = entry.document
            key = f"{i}-{id(doc):x}"
            if self._autosave.should_save(key, doc):
                pending.append((key, doc))
        if not pending:
            return

        from ..utils.worker import Worker
        self._autosave_busy = True

        def run() -> int:
            for key, doc in pending:
                self._autosave.save(key, doc)
            return len(pending)

        def done(count) -> None:
            self._autosave_busy = False
            if count:
                self._status.show_activity("Autosaved", 1500)

        def failed(message: str) -> None:
            self._autosave_busy = False
            self._status.show_activity(f"Autosave failed: {message}", 4000)

        Worker.run_async(run, on_result=done, on_error=failed)

    def offer_recovery(self) -> None:
        """On startup, offer any work left behind by a crashed session."""
        from ..utils.autosave import find_recoverable
        try:
            entries = find_recoverable(
                exclude_session=self._autosave._session)
        except Exception:
            return
        if not entries:
            return
        from .dialogs.recovery_dialog import RecoveryDialog
        dlg = RecoveryDialog(entries, self)
        if not dlg.exec():
            return
        for entry in dlg.selected_entries():
            try:
                self._document_ctrl.on_open_basera(str(entry.doc_path))
                if self._doc is not None:
                    # A recovered document is an unsaved copy: it must not
                    # silently overwrite the file it was recovered from.
                    self._doc.file_path = None
                    self._doc.name = entry.name
                    self._doc.mark_dirty()
                entry.discard()
            except Exception:
                continue
        if self._doc is not None:
            self._activate_project()

    def _refresh_history_panel(self) -> None:
        if self._doc:
            self._history_panel.refresh(self._doc.history)

    def _refresh_transform_panel(self) -> None:
        if self._doc:
            self._transform_panel.refresh(self._doc)

    def _refresh_channels_panel(self) -> None:
        if self._doc:
            self._channels_panel.refresh(self._doc)

    # ---- Document lifecycle -------------------------------------------------

    def _refresh(self, invalidate: bool = True, layer_id: str | None = None) -> None:
        """Full UI refresh.

        Parameters
        ----------
        invalidate : bool
            If *True* (default), the render cache is marked stale so the
            composite is recomputed.  Pass *False* for operations that
            only change non-pixel state (layer selection, panel sync)
            to skip expensive recompositing.
        layer_id : str | None
            Optional layer that changed.  When set the engine can use
            its incremental cache instead of a full rebuild.
        """
        if not self._doc:
            return
        self._canvas.set_document_ref(self._doc)
        if invalidate:
            # Immediately refresh the layers panel structure (no thumbnails)
            # so the user gets instant visual feedback while the expensive
            # async render runs.  The render callback will then refresh
            # again with thumbnails once the composite is ready.
            self._layers_panel.refresh(self._doc, thumbnails=False)
            self._sync_preview_level()
            self._pipeline.invalidate(layer_id)
            self._render_scheduler.enqueue_render(self._doc, full_refresh=True)
        else:
            # Cache hit — sync path is fast
            result = self._pipeline.execute_to_uint8(self._doc)
            self._canvas.set_image(
                result, force=False,
                doc_size=(self._doc.width, self._doc.height), src_rect=None)
            self._layers_panel.refresh(self._doc)
            self._history_panel.refresh(self._doc.history)
            self._transform_panel.refresh(self._doc)
            self._channels_panel.refresh(self._doc)
            self._selection_ctrl.update_selection_overlay()
            self._transform_ctrl.update_transform_box()
            self._view_ctrl.update_rulers()

    def _refresh_canvas_only(self) -> None:
        """Re-render and update canvas only (skip panel updates). Async."""
        if not self._doc:
            return
        self._sync_preview_level()
        active = self._doc.layers.active_layer
        self._pipeline.invalidate(active.id if active else None)
        self._render_scheduler.enqueue_render(self._doc, full_refresh=False)

    def _refresh_lightweight(self) -> None:
        """Re-render canvas + lightweight panel sync (no thumbnails). Async."""
        if not self._doc:
            return
        active = self._doc.layers.active_layer
        self._pipeline.invalidate(active.id if active else None)
        self._render_scheduler.enqueue_render(self._doc, full_refresh=False)

    def _schedule_render(self) -> None:
        """Request a deferred render (throttled, off the UI thread)."""
        if not self._doc:
            return
        self._sync_preview_level()
        active = self._doc.layers.active_layer
        self._pipeline.invalidate(active.id if active else None)
        self._render_scheduler.enqueue_render(self._doc, full_refresh=False)

    def _on_render_ready(self, rgba, _gen_id: int, full_refresh: bool,
                         doc_w: int = 0, doc_h: int = 0,
                         src_rect=None) -> None:
        """Render worker completed — update canvas and optionally panels."""
        if not self._doc:
            return
        # The frame may be a downscaled preview covering only the visible
        # region; the canvas still needs the document's logical size for all
        # coordinate mapping, plus the rect the buffer actually covers.
        self._canvas.set_image(
            rgba, force=True,
            doc_size=(doc_w, doc_h) if doc_w and doc_h
            else (self._doc.width, self._doc.height),
            src_rect=src_rect,
        )
        self._sync_canvas_scrollbars()
        # Overlays track the frame and are cheap now that the selection
        # caches its contours and bounds.
        self._selection_ctrl.update_selection_overlay()
        self._transform_ctrl.update_transform_box()
        # The transform, channels and history panels depend on document
        # state, not on the rendered pixels, so refreshing them once per
        # frame was pure waste -- the channels panel even re-composited the
        # active group each time. They are coalesced to ~5 fps instead.
        self._schedule_panel_refresh()
        if full_refresh:
            self._layers_panel.refresh(self._doc)

    def _on_render_error(self, message: str) -> None:
        """Render worker failed — fallback to sync render."""
        if self._doc:
            result = self._pipeline.execute_to_uint8(self._doc)
            self._canvas.set_image(
                result, force=True,
                doc_size=(self._doc.width, self._doc.height), src_rect=None)
            self._sync_canvas_scrollbars()
        self._status.show_activity(f"Render error: {message}", 3000)

    def execute_command(self, command: Command):
        """Execute a command and refresh. Decouples UI from engine.

        Returns the result of command.execute() (e.g. bool for MergeDownCommand).
        """
        if not self._doc:
            return None
        result = command.execute(self._doc)
        self._pipeline.invalidate()
        self._refresh()
        return result

    def execute_command_async(
        self,
        command: Command,
        on_success: Callable[[object], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        """Run a heavy command off the UI thread. Callbacks run on main thread."""
        if not self._doc:
            return
        from ..utils.worker import Worker

        doc = self._doc

        def run() -> object:
            return command.execute(doc)

        def on_result(result: object) -> None:
            try:
                if on_success:
                    on_success(result)
                else:
                    self._pipeline.invalidate()
                    self._refresh()
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self._status.show_activity(f"Callback error: {exc}", 5000)

        def on_err(msg: str) -> None:
            try:
                if on_error:
                    on_error(msg)
                else:
                    self._status.show_activity(f"Error: {msg}", 5000)
            except Exception:
                import traceback
                traceback.print_exc()
                self._status.show_activity(f"Error: {msg}", 5000)

        Worker.run_async(run, on_result=on_result, on_error=on_err)

    def _schedule_panel_refresh(self) -> None:
        """Request a deferred panel refresh (throttled to ~5fps)."""
        if not self._panel_refresh_pending:
            self._panel_refresh_pending = True
            self._panel_refresh_timer.start()

    def _do_deferred_panel_refresh(self) -> None:
        """Timer callback — perform the actual panel refresh."""
        self._panel_refresh_pending = False
        if self._doc:
            self._layers_panel.refresh(self._doc, thumbnails=False)
            self._transform_panel.refresh(self._doc)
            self._channels_panel.refresh(self._doc)
            self._history_panel.refresh(self._doc.history)
            self._view_ctrl.update_rulers()

    # ---- Key event handling -------------------------------------------------

    def keyPressEvent(self, event) -> None:
        if (self._tools.active_type == ToolType.TEXT
                and self._canvas._text_editing):
            return super().keyPressEvent(event)

        key = event.key()
        mods = event.modifiers()

        if self._vector_ctrl.handle_key_press(key, event):
            return
        if not mods and self._layer_ctrl.handle_numpad_opacity(key):
            return

        super().keyPressEvent(event)

    def dragEnterEvent(self, event) -> None:
        self._drop_ctrl.on_drag_enter(event)

    def dropEvent(self, event) -> None:
        self._drop_ctrl.on_drop(event)

    def _on_about(self) -> None:
        """Show the About dialog."""
        about_text = (
            "<h3>Basera (v0.4-alpha)</h3>"
            "<p><b>A professional-grade, Photoshop-style photo editor</b> built in Python "
            "with a modular, extensible architecture.</p>"
            "<p><b>Key Features:</b><ul>"
            "<li>Full Vector Support (SVG, shapes, pen/node tools)</li>"
            "<li>Advanced Selection & Masking workflows</li>"
            "<li>Retouching Power (Healing Brush, Clone Stamp)</li>"
            "<li>Layer System with 28 Blending Modes</li>"
            "<li>15 Non-Destructive Adjustments & 24 Filters</li>"
            "<li>Professional UI with dockable panels, rulers, and guides</li>"
            "</ul></p>"
            "<hr>"
            "<p>Developed by <b>Omar Ahmed Emara</b><br>"
            "GitHub: <a href='https://github.com/Mr-oxm'>https://github.com/Mr-oxm</a></p>"
        )
        QMessageBox.about(self, "About Basera", about_text)

    def _on_theme_changed(self, palette: dict) -> None:
        self._props_toolbar.setStyleSheet(render_qss("props_toolbar.qss", palette))

    def _on_brush_preset_changed(self, preset) -> None:
        """Apply the selected brush preset to the active brush-type tool."""
        if preset is not None and hasattr(self, "_tool_ctrl"):
            self._tool_ctrl.apply_brush_preset(preset)

    # ---- Welcome screen integration ----------------------------------------

    def _show_welcome_screen(self) -> None:
        """Switch the stacked widget to the welcome screen."""
        self._stacked.setCurrentIndex(0)
        self._set_editor_visible(False)
        self._refresh_recent_projects()

    def _show_editor(self) -> None:
        """Switch the stacked widget to the editor view."""
        self._stacked.setCurrentIndex(1)
        self._set_editor_visible(True)

    def _set_editor_visible(self, visible: bool) -> None:
        """Hide or show all panels, toolbars, and status bar."""
        self._toolbar.setVisible(visible)
        self._props_toolbar.setVisible(visible)
        self._layers_dock.setVisible(visible)
        self._history_dock.setVisible(visible)
        self._color_dock.setVisible(visible)
        self._transform_dock.setVisible(visible)
        self._brushes_dock.setVisible(visible)
        self._channels_dock.setVisible(visible)
        self._status.setVisible(visible)
        self._file_tabs.setVisible(visible)

    def _activate_project(self) -> None:
        """Transition from welcome screen to the editor with active project."""
        self._show_editor()

    def _on_welcome_new(self) -> None:
        """Handle 'New Project' from welcome screen."""
        from .dialogs.new_project_dialog import NewProjectDialog
        dlg = NewProjectDialog(self)
        if dlg.exec():
            w, h, dpi = dlg.get_values()
            self._document_ctrl.new_document(
                w, h, dpi,
                color_mode=dlg.get_color_mode(),
                color_profile=dlg.get_color_profile(),
                unit=dlg.get_unit(),
            )
            self._activate_project()

    def _on_welcome_open_image(self) -> None:
        """Handle 'Open Image' from welcome screen."""
        self._document_ctrl.on_open()
        if self._doc is not None:
            self._activate_project()

    def _on_welcome_open_basera(self) -> None:
        """Handle 'Open .basera' from welcome screen."""
        self._document_ctrl.on_open_basera()

    def _on_welcome_recent_selected(self, path: str) -> None:
        """Handle a click on a recent project entry."""
        from pathlib import Path
        if not Path(path).exists():
            from PySide6.QtWidgets import QMessageBox
            from ..utils.recent_projects import remove_recent_project
            QMessageBox.warning(
                self, "File Not Found",
                f"The project file could not be found:\n{path}\n\n"
                "It will be removed from your recent projects list.",
            )
            remove_recent_project(path)
            self._refresh_recent_projects()
            return
        self._document_ctrl.on_open_basera(path)

    def _refresh_recent_projects(self) -> None:
        """Reload recent projects from disk and push them to the welcome screen."""
        from ..utils.recent_projects import load_recent_projects, format_file_size
        entries = load_recent_projects()
        projects = [
            {
                "name": e["name"],
                "path": e["path"],
                "size": format_file_size(e["path"]),
            }
            for e in entries
        ]
        self._welcome.set_recent_projects(projects)

    def _on_last_tab_closed(self) -> None:
        """Called when the last tab is closed — return to welcome screen."""
        self._doc = None
        if not self._dev_mode:
            self._show_welcome_screen()

    # ---- Application close --------------------------------------------------

    def closeEvent(self, event) -> None:
        """Check every open document for unsaved changes before quitting.

        Wrapped in a try/except so that any unexpected error never silently
        swallows the dialog and closes the window without asking the user.
        """
        try:
            ctrl = getattr(self, "_document_ctrl", None)
            session = getattr(self, "_document_session", None)

            if ctrl is not None and session is not None:
                for i in range(len(session)):
                    entry = session.entry_at(i)
                    if entry is None:
                        continue
                    doc = entry.document
                    if not doc.dirty:
                        continue

                    result = ctrl._confirm_unsaved(doc)
                    if result == "cancel":
                        event.ignore()
                        return
                    if result == "save":
                        saved = ctrl._save_basera_sync(doc)
                        if not saved:
                            # User cancelled the Save As dialog — stay open.
                            event.ignore()
                            return

        except Exception:
            import traceback
            traceback.print_exc()
            # Do NOT silently close when an error occurs — ask the user.
            from PySide6.QtWidgets import QMessageBox
            resp = QMessageBox.question(
                self,
                "Close Basera",
                "An error occurred while checking for unsaved changes.\n"
                "Close anyway and risk losing unsaved work?",
            )
            if resp != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        # Clean exit: drop this session's autosaves so we do not offer to
        # "recover" work the user already dealt with.
        try:
            self._autosave_timer.stop()
            self._autosave.release()
        except Exception:
            pass
        try:
            self._render_scheduler.wait_for_idle(2000)
            self._pipeline.shutdown()
        except Exception:
            pass
        event.accept()
