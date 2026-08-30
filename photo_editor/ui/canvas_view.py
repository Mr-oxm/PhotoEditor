"""Zoomable, pannable canvas with selection overlay, transform box, tool cursors,
and in-canvas text editing overlay (cursor, selection highlight, text box).

Uses ``QOpenGLWidget`` when available so all QPainter operations are
GPU-accelerated.  Falls back to the software ``QWidget`` path
transparently.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QTimer, Signal
from PySide6.QtGui import (
    QColor, QCursor, QImage, QKeyEvent, QMouseEvent,
    QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QWheelEvent,
)
from PySide6.QtWidgets import QApplication, QWidget

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    _BASE_CLASS = QOpenGLWidget
except ImportError:
    _BASE_CLASS = QWidget

from ..core.enums import ToolType
from .canvas.canvas_cursors import (
    CURSORS,
    HANDLE_CURSORS,
    HANDLE_HIT,
    build_rotate_cursor,
    checker_tile,
    gradient_cursor,
)
from .canvas.canvas_overlays import CanvasOverlays
from .canvas.canvas_input import CanvasInputHandler


class CanvasView(_BASE_CLASS):
    """Interactive canvas that displays the composited document.

    Inherits from QOpenGLWidget when available (GPU-accelerated QPainter),
    otherwise falls back to QWidget (software rasterizer).
    """

    cursor_moved = Signal(int, int)
    tool_pressed = Signal(int, int, float)
    tool_moved = Signal(int, int, float)
    tool_released = Signal(int, int)
    # Widget-coordinate signals (for pan tool / raw screen deltas)
    widget_pressed = Signal(float, float)
    widget_moved = Signal(float, float)
    widget_released = Signal()
    # Emitted whenever zoom or pan changes (for ruler sync)
    view_changed = Signal()
    # Guide interaction on canvas
    guide_grabbed = Signal(object)          # Guide
    guide_drag_moved = Signal(object, float)  # Guide, new doc position
    guide_drag_released = Signal(object, float, bool)  # Guide, pos, delete?
    tool_double_clicked = Signal(int, int)  # doc x, doc y

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._zoom_to_mouse = True
        self._last_mouse = QPointF()
        self._panning = False
        self._doc_w = 0
        self._doc_h = 0
        # Selection overlay — marching ants
        self._sel_mask: np.ndarray | None = None
        self._sel_contours: list | None = None  # list of QPolygonF for marching ants
        self._sel_version: int | None = None    # content id of _sel_contours
        self._march_offset: int = 0               # animated dash offset
        self._march_timer = QTimer(self)
        self._march_timer.setInterval(100)         # ~10 fps animation
        self._march_timer.timeout.connect(self._march_tick)
        # Drag rect feedback for selection tools
        self._drag_rect: QRectF | None = None
        self._drag_is_ellipse: bool = False  # True → draw ellipse instead of rect
        # Lasso path preview (list of (x, y) doc coords)
        self._lasso_points: list[tuple[int, int]] | None = None
        # Transform bounding box (x, y, w, h) in document coordinates
        self._transform_box: tuple[int, int, int, int] | None = None
        self._transform_angle: float = 0.0  # rotation angle in degrees
        # Brush cursor / live dab preview
        self._brush_size: int = 0       # diameter in document pixels (0 = hidden)
        self._brush_cursor_pos: QPointF = QPointF()   # last mouse widget pos
        self._brush_cursor_visible: bool = False
        self._dab_pixmap: QPixmap | None = None       # pre-computed dab (brush or eraser)
        self._dab_is_eraser: bool = False
        # Cache identity of last rgba buffer to skip redundant QPixmap builds
        self._last_rgba: np.ndarray | None = None
        # Document-space rect the current frame covers (None = whole document).
        self._src_rect: tuple[int, int, int, int] | None = None

        # Text editing overlay state
        self._text_cursor_pos: tuple[int, int] | None = None   # (x, y) in doc coords
        self._text_cursor_height: int = 20
        self._text_cursor_visible: bool = False     # blink state
        self._text_editing: bool = False
        self._text_box: tuple[int, int, int, int] | None = None  # (x, y, w, h)
        self._text_box_angle: float = 0.0
        self._text_draw_rect: tuple[int, int, int, int] | None = None  # drawing preview
        self._text_selection_rects: list[tuple[int, int, int, int]] = []  # selection highlight
        # Cursor blink timer
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(530)
        self._blink_timer.timeout.connect(self._blink_cursor)
        # Key event forwarding for text editing
        self._key_handler = None  # callable(key, text, modifiers) -> bool

        # Gradient handle overlay state
        self._grad_start: tuple[int, int] | None = None   # doc coords
        self._grad_end: tuple[int, int] | None = None
        self._grad_stops: list = []                        # GradientStop list
        self._grad_handles_visible: bool = False

        # Clone / Heal source overlay state
        self._current_tool_type: ToolType | None = None
        self._source_pos: tuple[int, int] | None = None    # doc coords of source
        self._source_offset: tuple[int, int] | None = None  # (ox, oy) offset locked
        self._source_drawing: bool = False                  # True while painting
        self._clone_preview_pixmap: QPixmap | None = None   # live preview of source patch
        self._alt_held: bool = False                         # Alt modifier is held

        # Crop bounding box overlay state
        self._crop_box: tuple[int, int, int, int] | None = None

        # Smart-snapping alignment lines (document coords), shown only while
        # a drag is actually snapped to something.
        self._snap_lines: list = []  # (x, y, w, h) doc coords

        # Guide lines (list of guide objects with .orientation and .position)
        self._guide_lines: list = []
        # Preview guide (shown while dragging from ruler before release)
        self._preview_guide = None   # Guide or None
        # Guide dragging on canvas
        self._dragging_canvas_guide = None  # Guide being dragged on canvas
        self._guide_snap_px = 6  # pixel hit-test distance for guides

        # References set by MainWindow for vector overlay drawing
        self._doc_ref = None          # Document | None
        self._tool_manager_ref = None  # ToolManager | None

        # Boolean preview overlay state (set by VectorController)
        self._bool_preview_path = None   # VectorPath | None
        self._bool_source_ids: set = set()

        # Pick-segments mode overlay state (set by VectorController)
        self._pick_segments_state = None  # PickSegmentsState | None

        self._overlays = CanvasOverlays(self)
        self._input_handler = CanvasInputHandler(self)

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(200, 200)

    # ---- Public API ---------------------------------------------------------

    def visible_doc_rect(self, margin: float = 0.15) -> tuple[int, int, int, int] | None:
        """Document-space rectangle currently on screen, plus a margin.

        Returns None when the whole document is visible, so the renderer
        can skip ROI bookkeeping entirely. The margin gives the next frame
        a little slack during a pan, so small movements do not immediately
        expose un-rendered edges.
        """
        if not self._doc_w or not self._doc_h or self._zoom <= 0:
            return None
        dr = self._doc_rect()
        if (dr.left() >= 0 and dr.top() >= 0
                and dr.right() <= self.width() and dr.bottom() <= self.height()):
            return None                      # entire document fits on screen
        # Widget rect -> document coords
        x0 = (0 - dr.left()) / self._zoom
        y0 = (0 - dr.top()) / self._zoom
        x1 = (self.width() - dr.left()) / self._zoom
        y1 = (self.height() - dr.top()) / self._zoom
        mx = (x1 - x0) * margin
        my = (y1 - y0) * margin
        x0 = max(0, int(x0 - mx))
        y0 = max(0, int(y0 - my))
        x1 = min(self._doc_w, int(x1 + mx) + 1)
        y1 = min(self._doc_h, int(y1 + my) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        if x0 == 0 and y0 == 0 and x1 >= self._doc_w and y1 >= self._doc_h:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    def set_image(self, rgba: np.ndarray, force: bool = False,
                  doc_size: tuple[int, int] | None = None,
                  src_rect: tuple[int, int, int, int] | None = None) -> None:
        """Display a composited frame.

        *doc_size* is the document's logical size in document pixels. The
        frame itself may be a lower-resolution preview, and everything that
        maps between document and widget coordinates -- selection contours,
        transform boxes, guides, rulers, tool hit-testing -- works in
        document space. Deriving the document size from the buffer instead
        would silently rescale all of it whenever the preview level changed.
        """
        # Skip redundant QPixmap creation when the buffer is identical
        if not force and rgba is self._last_rgba and src_rect == self._src_rect:
            return
        self._last_rgba = rgba
        self._src_rect = src_rect
        h, w = rgba.shape[:2]
        if doc_size is not None:
            self._doc_w, self._doc_h = doc_size
        else:
            self._doc_w, self._doc_h = w, h
        # Use the buffer directly without .copy() — QPixmap.fromImage
        # copies the data internally so we don't need a second copy.
        qimg = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_selection_mask(self, mask: np.ndarray | None,
                           version: int | None = None) -> None:
        """Set the selection mask for marching-ants overlay rendering.

        *version* identifies the selection's content. When it is unchanged
        the contours are reused: tracing them costs a full-document uint8
        conversion, a cv2.findContours pass and one Python-level QPointF per
        contour point, and it was being redone on every rendered frame.
        """
        if mask is None:
            if self._sel_mask is None:
                return  # already cleared
            self._sel_mask = None
            self._sel_contours = None
            self._sel_version = None
            self._march_timer.stop()
            self.update()
            return
        if version is not None and version == self._sel_version:
            return  # unchanged since the contours were built
        self._sel_version = version
        self._sel_mask = mask
        # Extract contours from the binary mask for marching ants
        mask_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in contours:
            if len(c) < 2:
                continue
            pts = [QPointF(float(pt[0][0]), float(pt[0][1])) for pt in c]
            pts.append(pts[0])  # close the polygon
            polys.append(QPolygonF(pts))
        self._sel_contours = polys
        if polys:
            self._march_timer.start()
        else:
            self._march_timer.stop()
        self.update()

    def _march_tick(self) -> None:
        """Advance marching ants animation by one step."""
        self._march_offset = (self._march_offset + 1) % 16
        self.update()

    def set_drag_rect(self, rect: QRectF | None, ellipse: bool = False) -> None:
        self._drag_rect = rect
        self._drag_is_ellipse = ellipse
        self.update()

    def set_lasso_points(self, points: list[tuple[int, int]] | None) -> None:
        """Set lasso path preview points (doc coords) or None to clear."""
        self._lasso_points = points
        self.update()

    def set_transform_box(self, box: tuple[int, int, int, int] | None,
                          angle: float = 0.0) -> None:
        """Set the bounding box for the active layer (doc coords: x, y, w, h)."""
        self._transform_box = box
        self._transform_angle = angle
        self.update()

    def set_tool_cursor(self, tool_type: ToolType) -> None:
        self._current_tool_type = tool_type
        if tool_type == ToolType.GRADIENT:
            self.setCursor(gradient_cursor())
            return
        shape = CURSORS.get(tool_type, Qt.CursorShape.ArrowCursor)
        self.setCursor(QCursor(shape))

    # ---- Crop bounding box overlay API ------------------------------------

    def set_crop_box(self, box: tuple[int, int, int, int] | None) -> None:
        """Set / clear the crop bounding box overlay (doc coords: x, y, w, h)."""
        self._crop_box = box
        self.update()

    def set_snap_lines(self, lines: list) -> None:
        """Set the alignment lines to draw during a snapped drag."""
        if not lines and not self._snap_lines:
            return
        self._snap_lines = list(lines)
        self.update()

    def set_guides(self, guides: list) -> None:
        """Set the guide lines to draw (objects with .orientation and .position)."""
        self._guide_lines = list(guides)
        self.update()

    def set_preview_guide(self, guide) -> None:
        """Set a temporary preview guide (or None to clear)."""
        self._preview_guide = guide
        self.update()

    # ---- Clone / Heal source overlay API ------------------------------------

    def set_source_position(self, pos: tuple[int, int] | None) -> None:
        """Set/clear the clone/heal source position (doc coords)."""
        self._source_pos = pos
        self.update()

    def set_source_offset(self, offset: tuple[int, int] | None) -> None:
        """Set the locked source offset (ox, oy) during a paint stroke."""
        self._source_offset = offset

    def set_source_drawing(self, drawing: bool) -> None:
        """Toggle whether a clone/heal stroke is in progress."""
        self._source_drawing = drawing
        if not drawing:
            self._clone_preview_pixmap = None

    def set_clone_preview(self, preview_rgba: np.ndarray | None) -> None:
        """Set the live clone/heal preview patch (RGBA uint8)."""
        if preview_rgba is None or preview_rgba.size == 0:
            self._clone_preview_pixmap = None
            self.update()
            return
        h, w = preview_rgba.shape[:2]
        qimg = QImage(preview_rgba.data, w, h, w * 4,
                      QImage.Format.Format_RGBA8888)
        self._clone_preview_pixmap = QPixmap.fromImage(qimg.copy())
        self.update()

    # ---- Text editing overlay API -------------------------------------------

    def set_text_editing(self, editing: bool) -> None:
        """Enable / disable text editing overlay."""
        self._text_editing = editing
        if editing:
            self._text_cursor_visible = True
            self._blink_timer.start()
        else:
            self._blink_timer.stop()
            self._text_cursor_visible = False
            self._text_cursor_pos = None
            self._text_box = None
            self._text_draw_rect = None
            self._text_selection_rects = []
        self.update()

    def set_text_cursor(self, x: int, y: int, height: int) -> None:
        """Set the text cursor position in document coordinates."""
        self._text_cursor_pos = (x, y)
        self._text_cursor_height = height
        self._text_cursor_visible = True
        self._blink_timer.start()
        self.update()

    def set_text_box(self, box: tuple[int, int, int, int] | None,
                     angle: float = 0.0) -> None:
        """Set the text bounding box (doc coords) for overlay drawing."""
        self._text_box = box
        self._text_box_angle = angle
        self.update()

    def set_text_draw_rect(self, rect: tuple[int, int, int, int] | None) -> None:
        """Set the text box drawing preview rectangle (doc coords)."""
        self._text_draw_rect = rect
        self.update()

    def set_text_selection_rects(self, rects: list[tuple[int, int, int, int]]) -> None:
        """Set selection highlight rectangles (doc coords relative to text box)."""
        self._text_selection_rects = rects
        self.update()

    def set_key_handler(self, handler) -> None:
        """Set the key event handler for text editing.
        handler(key: int, text: str, modifiers: Qt.KeyboardModifier) -> bool"""
        self._key_handler = handler

    def _blink_cursor(self) -> None:
        self._text_cursor_visible = not self._text_cursor_visible
        self.update()

    # ---- Gradient handle overlay API ----------------------------------------

    def set_gradient_handles(
        self,
        start: tuple[int, int] | None,
        end: tuple[int, int] | None,
        stops: list,
        visible: bool,
    ) -> None:
        """Set / clear the gradient control-line overlay (doc coords)."""
        self._grad_start = start
        self._grad_end = end
        self._grad_stops = list(stops) if stops else []
        self._grad_handles_visible = visible
        self.update()

    # ---- Brush cursor / live dab preview ------------------------------------

    def set_brush_dab(self, dab_rgba: np.ndarray | None,
                      is_eraser: bool = False) -> None:
        """Set the pre-computed dab preview (RGBA uint8).

        For a **brush** the dab is drawn directly (paint colour + hardness).
        For an **eraser** the dab's alpha channel is used to shape a
        checkerboard pattern that previews the transparency to be created.
        """
        if dab_rgba is None or dab_rgba.size == 0:
            self._dab_pixmap = None
            self._brush_size = 0
            self._dab_is_eraser = False
            self.update()
            return

        h, w = dab_rgba.shape[:2]
        self._brush_size = w  # diameter in doc pixels
        self._dab_is_eraser = is_eraser

        if is_eraser:
            # Build a checkerboard image shaped by the eraser's alpha mask.
            # This shows exactly the transparency pattern the eraser will
            # create, following the tool's hardness / opacity / future shapes.
            alpha = dab_rgba[..., 3]  # uint8, 0-255
            cs = max(2, w // 8)       # checker square size
            yy, xx = np.mgrid[0:h, 0:w]
            parity = ((yy // cs) + (xx // cs)) % 2
            checker = np.zeros((h, w, 4), dtype=np.uint8)
            checker[..., :3] = np.where(parity[..., None] == 0, 230, 180)
            checker[..., 3] = alpha   # shaped by the eraser dab
            qimg = QImage(checker.data, w, h, w * 4,
                          QImage.Format.Format_RGBA8888)
            self._dab_pixmap = QPixmap.fromImage(qimg.copy())
        else:
            # Convert the RGBA dab straight into a QPixmap
            qimg = QImage(dab_rgba.data, w, h, w * 4,
                          QImage.Format.Format_RGBA8888)
            self._dab_pixmap = QPixmap.fromImage(qimg.copy())

        self.update()

    def hide_brush_preview(self) -> None:
        self._dab_pixmap = None
        self._brush_size = 0
        self._brush_cursor_visible = False
        self.update()

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def pan(self) -> QPointF:
        return QPointF(self._pan)

    @property
    def zoom_to_mouse(self) -> bool:
        return self._zoom_to_mouse

    def set_zoom_to_mouse(self, enabled: bool) -> None:
        self._zoom_to_mouse = bool(enabled)

    def fit_zoom(self) -> float:
        if not self._doc_w or not self._doc_h or self.width() <= 0 or self.height() <= 0:
            return 1.0
        sx = self.width() / self._doc_w
        sy = self.height() / self._doc_h
        return max(0.01, min(min(sx, sy) * 0.9, 32.0))

    def scrollable_span(self) -> tuple[float, float]:
        if not self._doc_w or not self._doc_h:
            return 0.0, 0.0
        return (
            max(0.0, self._doc_w * self._zoom - self.width()),
            max(0.0, self._doc_h * self._zoom - self.height()),
        )

    def _pan_limits(self) -> tuple[float, float]:
        span_x, span_y = self.scrollable_span()
        return span_x / 2.0, span_y / 2.0

    def _clamp_pan(self, pan: QPointF) -> QPointF:
        limit_x, limit_y = self._pan_limits()
        return QPointF(
            max(-limit_x, min(pan.x(), limit_x)),
            max(-limit_y, min(pan.y(), limit_y)),
        )

    def set_pan(self, pan: QPointF) -> None:
        self._pan = self._clamp_pan(pan)
        self.update()
        self.view_changed.emit()

    def scroll_by(self, dx: float = 0.0, dy: float = 0.0) -> None:
        self.set_pan(self._pan + QPointF(dx, dy))

    def _canvas_to_doc_float(self, pos: QPointF) -> QPointF:
        cx = self.width() / 2 + self._pan.x()
        cy = self.height() / 2 + self._pan.y()
        dx = (pos.x() - cx) / self._zoom + self._doc_w / 2
        dy = (pos.y() - cy) / self._zoom + self._doc_h / 2
        return QPointF(dx, dy)

    def _pan_for_anchor(self, doc_pos: QPointF, widget_pos: QPointF, zoom: float) -> QPointF:
        return self._clamp_pan(QPointF(
            widget_pos.x() - self.width() / 2 - (doc_pos.x() - self._doc_w / 2) * zoom,
            widget_pos.y() - self.height() / 2 - (doc_pos.y() - self._doc_h / 2) * zoom,
        ))

    def set_zoom(self, z: float, anchor: QPointF | None = None) -> None:
        target_zoom = max(0.01, min(z, 32.0))
        doc_anchor = None
        if anchor is not None and self._doc_w and self._doc_h:
            doc_anchor = self._canvas_to_doc_float(anchor)
        self._zoom = target_zoom
        if doc_anchor is not None:
            self._pan = self._pan_for_anchor(doc_anchor, anchor, target_zoom)
        else:
            self._pan = self._clamp_pan(self._pan)
        self.update()
        self.view_changed.emit()

    def zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        self.set_zoom(self._zoom * factor, anchor=anchor)

    def zoom_to_fit(self) -> None:
        if self._doc_w and self._doc_h:
            self._zoom = self.fit_zoom()
            self._pan = QPointF(0, 0)
            self.update()
            self.view_changed.emit()

    def _canvas_to_doc(self, pos: QPointF) -> tuple[int, int]:
        doc_pos = self._canvas_to_doc_float(pos)
        dx = doc_pos.x()
        dy = doc_pos.y()
        return int(dx), int(dy)

    def _doc_rect(self) -> QRectF:
        cx = self.width() / 2 + self._pan.x()
        cy = self.height() / 2 + self._pan.y()
        sw = self._doc_w * self._zoom
        sh = self._doc_h * self._zoom
        return QRectF(cx - sw / 2, cy - sh / 2, sw, sh)

    # ---- Paint --------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        from .theme import ThemeManager
        palette = ThemeManager.instance().active_palette
        p.fillRect(self.rect(), QColor(palette['bg1']))

        if self._pixmap is None:
            p.end()
            return

        dr = self._doc_rect()

        # Checkerboard — single hardware-accelerated tiled draw
        p.save()
        p.setClipRect(dr.toAlignedRect())
        p.drawTiledPixmap(dr.toAlignedRect(), checker_tile())
        p.restore()

        # Document image. A partial frame (viewport ROI) covers only part
        # of the document, so it is drawn into the matching sub-rectangle.
        if self._src_rect is None:
            p.drawPixmap(dr.toAlignedRect(), self._pixmap)
        else:
            sx, sy, sw, sh = self._src_rect
            fx = dr.width() / self._doc_w if self._doc_w else 1.0
            fy = dr.height() / self._doc_h if self._doc_h else 1.0
            target = QRectF(dr.left() + sx * fx, dr.top() + sy * fy,
                            sw * fx, sh * fy)
            p.drawPixmap(target.toAlignedRect(), self._pixmap)

        # Selection overlay — marching ants
        if self._sel_contours:
            self._overlays.draw_marching_ants(p, dr)

        # Selection drag rectangle / ellipse
        if self._drag_rect is not None:
            pen = QPen(QColor(100, 180, 255), 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(QColor(100, 180, 255, 30))
            if self._drag_is_ellipse:
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                p.drawEllipse(self._drag_rect)
            else:
                p.drawRect(self._drag_rect)

        # Lasso path preview
        if self._lasso_points and len(self._lasso_points) > 1 and self._doc_w > 0:
            from PySide6.QtGui import QPolygonF as _QPolyF
            pen = QPen(QColor(100, 180, 255), 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            poly = _QPolyF()
            for lx, ly in self._lasso_points:
                sx = dr.left() + (lx / self._doc_w) * dr.width()
                sy = dr.top() + (ly / self._doc_h) * dr.height()
                poly.append(QPointF(sx, sy))
            # Close back to start
            lx0, ly0 = self._lasso_points[0]
            poly.append(QPointF(
                dr.left() + (lx0 / self._doc_w) * dr.width(),
                dr.top() + (ly0 / self._doc_h) * dr.height(),
            ))
            p.drawPolyline(poly)

        # Transform bounding box with handles
        if self._transform_box is not None:
            self._overlays.draw_transform_box(p, dr)

        # Move tool marquee selection box
        if self._current_tool_type == ToolType.MOVE and self._tool_manager_ref is not None:
            tool = self._tool_manager_ref.active_tool
            mrect = getattr(tool, 'marquee_rect', None)
            if callable(getattr(type(tool), 'marquee_rect', None)):
                # It's a property, access the value
                mrect = tool.marquee_rect
            if mrect is not None:
                s, e = mrect
                ss = self._doc_to_widget(dr, float(s[0]), float(s[1]))
                se = self._doc_to_widget(dr, float(e[0]), float(e[1]))
                marquee_pen = QPen(QColor(100, 180, 255), 1.0, Qt.PenStyle.DashLine)
                marquee_pen.setCosmetic(True)
                p.setPen(marquee_pen)
                p.setBrush(QColor(100, 180, 255, 30))
                p.drawRect(QRectF(ss, se).normalized())

        # Text editing overlays
        if self._text_draw_rect is not None:
            self._overlays.draw_text_draw_rect(p, dr)
        if self._text_box is not None:
            self._overlays.draw_text_box(p, dr)
        if self._text_selection_rects:
            self._overlays.draw_text_selection(p, dr)
        if self._text_cursor_pos is not None and self._text_cursor_visible and self._text_editing:
            self._overlays.draw_text_cursor(p, dr)

        # Brush cursor preview circle
        if self._brush_size > 0 and self._brush_cursor_visible:
            self._overlays.draw_brush_cursor(p)

        # Clone / Heal source crosshair and live preview
        if (self._source_pos is not None
                and self._current_tool_type in (ToolType.CLONE_STAMP, ToolType.HEALING_BRUSH)):
            self._overlays.draw_source_overlay(p, dr)

        # Crop bounding box with dimmed surround
        if self._crop_box is not None:
            self._overlays.draw_crop_box(p, dr)

        # Gradient control line + stop handles
        if self._grad_handles_visible:
            self._overlays.draw_gradient_handles(p, dr)

        # Guide lines overlay (including preview guide)
        if self._guide_lines or self._preview_guide is not None:
            self._overlays.draw_guides(p, dr)

        # Smart-snapping alignment lines
        if self._snap_lines:
            self._overlays.draw_snap_lines(p, dr)

        # Vector object overlay (node handles, path outlines)
        if self._current_tool_type in (ToolType.PEN, ToolType.NODE, ToolType.VECTOR_SHAPE, ToolType.MOVE):
            self._overlays.draw_vector_overlay(p, dr)

        # Boolean preview overlay (hover over toolbar buttons)
        if getattr(self, "_bool_preview_path", None) is not None:
            self._overlays.draw_boolean_preview(p, dr)

        # Pick-segments overlay
        if getattr(self, "_pick_segments_state", None) is not None:
            ps = self._pick_segments_state
            if ps.active:
                self._overlays.draw_pick_segments(p, dr)

        p.end()

    def _update_transform_cursor(self, pos: QPointF) -> None:
        """Adjust cursor shape when hovering over the transform box / handles."""
        dr = self._doc_rect()
        br = self._overlays.box_widget_rect(dr)
        if br is None:
            return

        cx = br.x() + br.width() / 2
        cy = br.y() + br.height() / 2
        hw, hh = br.width() / 2, br.height() / 2

        # Inverse-rotate the mouse position into the box's local frame
        px, py = pos.x() - cx, pos.y() - cy
        if self._transform_angle != 0.0:
            rad = math.radians(self._transform_angle)
            rx = px * math.cos(rad) - py * math.sin(rad)
            ry = px * math.sin(rad) + py * math.cos(rad)
            px, py = rx, ry

        # Rotation handle node (above top-center)
        rh_offset = 20.0
        rh_x, rh_y = 0, -hh - rh_offset
        if abs(px - rh_x) <= HANDLE_HIT and abs(py - rh_y) <= HANDLE_HIT:
            self.setCursor(QCursor(build_rotate_cursor()))
            return

        # Hit-test against centered handle positions (expanded hit area)
        local_handles = [
            ("TL", -hw, -hh), ("T", 0, -hh), ("TR", hw, -hh),
            ("L", -hw, 0), ("R", hw, 0),
            ("BL", -hw, hh), ("B", 0, hh), ("BR", hw, hh),
        ]
        for name, hx, hy in local_handles:
            if abs(px - hx) <= HANDLE_HIT and abs(py - hy) <= HANDLE_HIT:
                self.setCursor(QCursor(HANDLE_CURSORS[name]))
                return

        if -hw <= px <= hw and -hh <= py <= hh:
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
            return

        # Rotate cursor only near corners (within ROTATE_PROX pixels)
        ROTATE_PROX = 50.0
        corners = [
            (-hw, -hh), (hw, -hh), (-hw, hh), (hw, hh),
            (rh_x, rh_y),
        ]
        for (ccx, ccy) in corners:
            dist_sq = (px - ccx) ** 2 + (py - ccy) ** 2
            if dist_sq <= ROTATE_PROX * ROTATE_PROX:
                self.setCursor(build_rotate_cursor())
                return

        # Outside all zones — revert to default cursor
        self.set_tool_cursor(self._current_tool_type)

    def _hit_test_guide(self, pos: QPointF):
        """Return the Guide object near *pos* (widget coords), or None."""
        if not self._guide_lines or self._doc_w == 0 or self._doc_h == 0:
            return None
        dr = self._doc_rect()
        sx = dr.width() / self._doc_w
        sy = dr.height() / self._doc_h
        snap = self._guide_snap_px
        for g in self._guide_lines:
            if g.orientation == Qt.Orientation.Horizontal:
                wy = dr.top() + g.position * sy
                if abs(pos.y() - wy) <= snap:
                    return g
            else:  # Vertical
                wx = dr.left() + g.position * sx
                if abs(pos.x() - wx) <= snap:
                    return g
        return None

    def set_document_ref(self, doc) -> None:
        """Store a reference to the active Document for overlay drawing."""
        self._doc_ref = doc

    def set_tool_manager_ref(self, tm) -> None:
        """Store a reference to the ToolManager for overlay state."""
        self._tool_manager_ref = tm

    def _doc_to_widget(self, dr: QRectF, dx: float, dy: float) -> QPointF:
        """Convert document coords to widget coords."""
        sx = dr.width() / self._doc_w if self._doc_w else 1
        sy = dr.height() / self._doc_h if self._doc_h else 1
        return QPointF(dr.left() + dx * sx, dr.top() + dy * sy)

    # ---- Key events (text editing + Alt source mode) -----------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._input_handler.handle_key_press(event):
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if self._input_handler.handle_key_release(event):
            return
        super().keyReleaseEvent(event)

    # ---- Mouse events -------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:
        self._input_handler.handle_wheel(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._input_handler.handle_mouse_press(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._input_handler.handle_mouse_move(event)

    def resizeEvent(self, event) -> None:
        self._pan = self._clamp_pan(self._pan)
        super().resizeEvent(event)
        self.view_changed.emit()

    def leaveEvent(self, event) -> None:
        self._input_handler.handle_leave()

    def enterEvent(self, event) -> None:
        self._input_handler.handle_enter()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._input_handler.handle_mouse_release(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self._input_handler.handle_mouse_double_click(event)
