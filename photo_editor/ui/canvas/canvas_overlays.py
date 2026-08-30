"""Canvas overlay drawing — marching ants, transform box, guides, etc."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF

from ...core.enums import ToolType

if TYPE_CHECKING:
    from ..canvas_view import CanvasView  # noqa: F401


def _map_polygon_to_screen(poly: QPolygonF, dr: QRectF, sx: float, sy: float) -> QPolygonF:
    """Map a QPolygonF from document coords to screen coords."""
    pts = []
    for i in range(poly.count()):
        pt = poly.at(i)
        pts.append(QPointF(dr.left() + pt.x() * sx, dr.top() + pt.y() * sy))
    return QPolygonF(pts)


class CanvasOverlays:
    """Draws all overlay elements (marching ants, transform box, guides, etc.)."""

    def __init__(self, canvas: CanvasView) -> None:
        self._canvas = canvas

    def draw_marching_ants(self, p: QPainter, dr: QRectF) -> None:
        """Draw animated marching-ants contour from the selection mask."""
        c = self._canvas
        if not c._sel_contours or c._doc_w == 0 or c._doc_h == 0:
            return
        sx = dr.width() / c._doc_w
        sy = dr.height() / c._doc_h

        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        bg_pen = QPen(QColor(255, 255, 255), 1.0, Qt.PenStyle.SolidLine)
        bg_pen.setCosmetic(True)
        p.setPen(bg_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for poly in c._sel_contours:
            mapped = _map_polygon_to_screen(poly, dr, sx, sy)
            p.drawPolyline(mapped)

        fg_pen = QPen(QColor(0, 0, 0), 1.0, Qt.PenStyle.DashLine)
        fg_pen.setCosmetic(True)
        fg_pen.setDashOffset(float(c._march_offset))
        p.setPen(fg_pen)
        for poly in c._sel_contours:
            mapped = _map_polygon_to_screen(poly, dr, sx, sy)
            p.drawPolyline(mapped)

        p.restore()

    def draw_crop_box(self, p: QPainter, dr: QRectF) -> None:
        """Draw the crop bounding box with dimmed outside area and handles."""
        c = self._canvas
        if c._crop_box is None or c._doc_w == 0 or c._doc_h == 0:
            return
        x, y, w, h = c._crop_box
        sx = dr.width() / c._doc_w
        sy = dr.height() / c._doc_h
        crop_rect = QRectF(dr.left() + x * sx, dr.top() + y * sy, w * sx, h * sy)

        p.save()

        dim_path = QPainterPath()
        dim_path.addRect(dr)
        inner_path = QPainterPath()
        inner_path.addRect(crop_rect)
        dim_path = dim_path.subtracted(inner_path)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 140))
        p.drawPath(dim_path)

        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(220, 50, 50, 230), 1.5)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(crop_rect)

        pen_grid = QPen(QColor(220, 50, 50, 80), 1.0)
        pen_grid.setCosmetic(True)
        pen_grid.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen_grid)
        for i in (1, 2):
            gx = crop_rect.left() + crop_rect.width() * i / 3
            p.drawLine(QPointF(gx, crop_rect.top()), QPointF(gx, crop_rect.bottom()))
            gy = crop_rect.top() + crop_rect.height() * i / 3
            p.drawLine(QPointF(crop_rect.left(), gy), QPointF(crop_rect.right(), gy))

        hs = 7
        handle_pts = [
            (crop_rect.left(), crop_rect.top()),
            (crop_rect.left() + crop_rect.width() / 2, crop_rect.top()),
            (crop_rect.right(), crop_rect.top()),
            (crop_rect.left(), crop_rect.top() + crop_rect.height() / 2),
            (crop_rect.right(), crop_rect.top() + crop_rect.height() / 2),
            (crop_rect.left(), crop_rect.bottom()),
            (crop_rect.left() + crop_rect.width() / 2, crop_rect.bottom()),
            (crop_rect.right(), crop_rect.bottom()),
        ]
        p.setPen(QPen(QColor(220, 50, 50, 230), 1))
        p.setBrush(QColor(255, 255, 255))
        for hx, hy in handle_pts:
            p.drawRect(QRectF(hx - hs / 2, hy - hs / 2, hs, hs))

        p.restore()

    def draw_transform_box(self, p: QPainter, dr: QRectF) -> None:
        """Draw the transform bounding box with resize/rotate handles."""
        c = self._canvas
        br = self.box_widget_rect(dr)
        if br is None:
            return

        cx = br.x() + br.width() / 2
        cy = br.y() + br.height() / 2
        hw, hh = br.width() / 2, br.height() / 2

        p.save()
        p.translate(cx, cy)
        if c._transform_angle != 0.0:
            p.rotate(-c._transform_angle)

        p.setPen(QPen(QColor(0, 150, 255), 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(-hw, -hh, br.width(), br.height()))

        rh_offset = 20.0
        rh_x, rh_y = 0, -hh - rh_offset
        p.setPen(QPen(QColor(0, 150, 255), 1.0))
        p.drawLine(QPointF(0, -hh), QPointF(rh_x, rh_y))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(QPen(QColor(0, 150, 255), 1.5))
        p.setBrush(QColor(255, 255, 255))
        p.drawEllipse(QPointF(rh_x, rh_y), 5.0, 5.0)

        hs = 7
        handle_pts = [
            (-hw, -hh), (0, -hh), (hw, -hh),
            (-hw, 0), (hw, 0),
            (-hw, hh), (0, hh), (hw, hh),
        ]
        p.setPen(QPen(QColor(0, 150, 255), 1))
        p.setBrush(QColor(255, 255, 255))
        for hx, hy in handle_pts:
            p.drawRect(QRectF(hx - hs / 2, hy - hs / 2, hs, hs))

        p.restore()

    def box_widget_rect(self, dr: QRectF) -> QRectF | None:
        """Return the transform box rectangle in widget coordinates."""
        c = self._canvas
        if c._transform_box is None or c._doc_w == 0 or c._doc_h == 0:
            return None
        x, y, w, h = c._transform_box
        sx = dr.width() / c._doc_w
        sy = dr.height() / c._doc_h
        return QRectF(dr.left() + x * sx, dr.top() + y * sy, w * sx, h * sy)

    def draw_brush_cursor(self, p: QPainter) -> None:
        """Draw a live preview of the brush dab."""
        c = self._canvas
        pos = c._brush_cursor_pos
        radius_screen = (c._brush_size / 2) * c._zoom

        if radius_screen < 1.5:
            return

        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        dab_screen = radius_screen * 2
        target = QRectF(pos.x() - radius_screen, pos.y() - radius_screen,
                        dab_screen, dab_screen)

        if c._dab_pixmap is not None:
            p.drawPixmap(target.toAlignedRect(), c._dab_pixmap)

        pen_outer = QPen(QColor(0, 0, 0, 160), 1.0)
        pen_outer.setCosmetic(True)
        p.setPen(pen_outer)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(pos, radius_screen, radius_screen)

        pen_inner = QPen(QColor(255, 255, 255, 180), 1.0)
        pen_inner.setCosmetic(True)
        pen_inner.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen_inner)
        p.drawEllipse(pos, radius_screen, radius_screen)

        ch = max(3.0, min(radius_screen * 0.12, 6.0))
        p.setPen(QPen(QColor(255, 255, 255, 200), 1.0))
        p.drawLine(QPointF(pos.x() - ch, pos.y()), QPointF(pos.x() + ch, pos.y()))
        p.drawLine(QPointF(pos.x(), pos.y() - ch), QPointF(pos.x(), pos.y() + ch))

        if c._clone_preview_pixmap is not None:
            p.setOpacity(1.0)
            p.drawPixmap(target.toAlignedRect(), c._clone_preview_pixmap)
            p.setOpacity(1.0)

        p.restore()

    def draw_source_overlay(self, p: QPainter, dr: QRectF) -> None:
        """Draw crosshair at clone/heal source position."""
        c = self._canvas
        if c._source_pos is None or c._doc_w == 0 or c._doc_h == 0:
            return

        sx_scale = dr.width() / c._doc_w
        sy_scale = dr.height() / c._doc_h

        if c._source_offset is not None and c._brush_cursor_visible:
            doc_pos = c._canvas_to_doc(c._brush_cursor_pos)
            src_x = doc_pos[0] + c._source_offset[0]
            src_y = doc_pos[1] + c._source_offset[1]
        else:
            src_x, src_y = c._source_pos

        wx = dr.left() + src_x * sx_scale
        wy = dr.top() + src_y * sy_scale

        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        arm = 12.0
        gap = 4.0

        pen_outer = QPen(QColor(0, 0, 0, 200), 1.6)
        pen_outer.setCosmetic(True)
        p.setPen(pen_outer)
        p.drawLine(QPointF(wx, wy - arm), QPointF(wx, wy - gap))
        p.drawLine(QPointF(wx, wy + gap), QPointF(wx, wy + arm))
        p.drawLine(QPointF(wx - arm, wy), QPointF(wx - gap, wy))
        p.drawLine(QPointF(wx + arm, wy), QPointF(wx + gap, wy))

        pen_inner = QPen(QColor(0, 220, 255, 230), 1.0)
        pen_inner.setCosmetic(True)
        p.setPen(pen_inner)
        p.drawLine(QPointF(wx, wy - arm), QPointF(wx, wy - gap))
        p.drawLine(QPointF(wx, wy + gap), QPointF(wx, wy + arm))
        p.drawLine(QPointF(wx - arm, wy), QPointF(wx - gap, wy))
        p.drawLine(QPointF(wx + arm, wy), QPointF(wx + gap, wy))

        p.setPen(QPen(QColor(0, 220, 255, 200), 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(wx, wy), 3.0, 3.0)

        if c._brush_size > 0:
            src_radius = (c._brush_size / 2) * c._zoom
            pen_src = QPen(QColor(0, 220, 255, 120), 1.0)
            pen_src.setCosmetic(True)
            pen_src.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen_src)
            p.drawEllipse(QPointF(wx, wy), src_radius, src_radius)

        p.restore()

    def draw_gradient_handles(self, p: QPainter, dr: QRectF) -> None:
        """Draw the gradient control line with coloured stop circles."""
        c = self._canvas
        if not c._grad_start or not c._grad_end:
            return
        if c._doc_w == 0 or c._doc_h == 0:
            return

        sx = dr.width() / c._doc_w
        sy = dr.height() / c._doc_h

        s = QPointF(dr.left() + c._grad_start[0] * sx,
                    dr.top() + c._grad_start[1] * sy)
        e = QPointF(dr.left() + c._grad_end[0] * sx,
                    dr.top() + c._grad_end[1] * sy)

        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen_shadow = QPen(QColor(0, 0, 0, 100), 3.0)
        pen_shadow.setCosmetic(True)
        p.setPen(pen_shadow)
        p.drawLine(s, e)

        pen_line = QPen(QColor(255, 255, 255, 200), 1.4)
        pen_line.setCosmetic(True)
        p.setPen(pen_line)
        p.drawLine(s, e)

        for stop in c._grad_stops:
            if stop.position <= 0.0 or stop.position >= 1.0:
                continue
            cx = s.x() + (e.x() - s.x()) * stop.position
            cy = s.y() + (e.y() - s.y()) * stop.position
            r, g, b, a = stop.color.to_rgb8()
            p.setPen(QPen(QColor(255, 255, 255, 220), 1.4))
            p.setBrush(QColor(r, g, b, a))
            p.drawEllipse(QPointF(cx, cy), 5.0, 5.0)

        if c._grad_stops:
            r0, g0, b0, a0 = c._grad_stops[0].color.to_rgb8()
        else:
            r0, g0, b0, a0 = 0, 0, 0, 255
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 50))
        p.drawEllipse(s, 10.0, 10.0)
        p.setPen(QPen(QColor(255, 255, 255), 2.0))
        p.setBrush(QColor(r0, g0, b0, a0))
        p.drawEllipse(s, 8.0, 8.0)

        if c._grad_stops:
            r1, g1, b1, a1 = c._grad_stops[-1].color.to_rgb8()
        else:
            r1, g1, b1, a1 = 255, 255, 255, 255
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 50))
        p.drawEllipse(e, 10.0, 10.0)
        p.setPen(QPen(QColor(255, 255, 255), 2.0))
        p.setBrush(QColor(r1, g1, b1, a1))
        p.drawEllipse(e, 8.0, 8.0)

        p.restore()

    def draw_guides(self, p: QPainter, dr: QRectF) -> None:
        """Draw horizontal and vertical guide lines."""
        c = self._canvas
        all_guides = list(c._guide_lines)
        preview = c._preview_guide
        if not all_guides and preview is None:
            return
        if c._doc_w == 0 or c._doc_h == 0:
            return

        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        guide_pen = QPen(QColor(74, 179, 255, 180), 1.0, Qt.PenStyle.DashLine)
        guide_pen.setCosmetic(True)
        preview_pen = QPen(QColor(74, 179, 255, 100), 1.0, Qt.PenStyle.DashLine)
        preview_pen.setCosmetic(True)
        p.setBrush(Qt.BrushStyle.NoBrush)

        sx = dr.width() / c._doc_w
        sy = dr.height() / c._doc_h

        def _draw_one(g, pen):
            p.setPen(pen)
            if g.orientation == Qt.Orientation.Vertical:
                wx = dr.left() + g.position * sx
                p.drawLine(QPointF(wx, 0), QPointF(wx, c.height()))
            else:
                wy = dr.top() + g.position * sy
                p.drawLine(QPointF(0, wy), QPointF(c.width(), wy))

        for g in all_guides:
            _draw_one(g, guide_pen)
        if preview is not None:
            _draw_one(preview, preview_pen)

        p.restore()

    def draw_vector_overlay(self, p: QPainter, dr: QRectF) -> None:
        """Draw path outlines, node handles, pen preview for vector layers."""
        c = self._canvas
        doc = c._doc_ref
        if doc is None or c._doc_w == 0 or c._doc_h == 0:
            return
            
        try:
            from ...vector.path import SegmentType, HandleMode
        except ImportError:
            return

        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        sx = dr.width() / c._doc_w
        sy = dr.height() / c._doc_h
        
        accent = QColor(100, 180, 255)
        accent_dim = QColor(100, 180, 255, 100)
        white = QColor(255, 255, 255)

        path_pen = QPen(accent, 1.0, Qt.PenStyle.SolidLine)
        path_pen.setCosmetic(True)
        path_pen_sel = QPen(accent, 1.5, Qt.PenStyle.SolidLine)
        path_pen_sel.setCosmetic(True)
        handle_pen = QPen(accent_dim, 0.8, Qt.PenStyle.SolidLine)
        handle_pen.setCosmetic(True)
        preview_pen = QPen(QColor(100, 180, 255, 120), 1.0, Qt.PenStyle.DashLine)
        preview_pen.setCosmetic(True)

        is_pen_tool = c._current_tool_type == ToolType.PEN
        is_node_tool = c._current_tool_type == ToolType.NODE
        show_nodes = c._current_tool_type in (ToolType.PEN, ToolType.NODE)

        selected_indices = doc.layers.selected_indices
        num_layers = len(doc.layers.layers)
        shape_layers = []
        for idx in selected_indices:
            if 0 <= idx < num_layers:
                l = doc.layers.layers[idx]
                if getattr(l, "_vector_data", None) is not None:
                    shape_layers.append(l)

        active_layer = doc.layers.active_layer
        if not shape_layers and active_layer and getattr(active_layer, "_vector_data", None) is not None:
            shape_layers.append(active_layer)

        if not shape_layers:
            p.restore()
            return
            
        def _get_to_screen_func(layer, dr, sx, sy):
            nd_angle = layer.transform_angle
            has_nd_rotation = (
                nd_angle != 0.0
                and getattr(layer, "_source_pixels", None) is not None
            )

            _live_dx: float = 0.0
            _live_dy: float = 0.0
            _live_sx: float = 1.0
            _live_sy: float = 1.0
            _live_old_cx: float = 0.0
            _live_old_cy: float = 0.0
            _live_new_cx: float = 0.0
            _live_new_cy: float = 0.0
            _apply_live_scale: bool = False

            if c._current_tool_type is not None:
                from ...core.enums import ToolType as _TT
                if c._current_tool_type == _TT.MOVE and c._tool_manager_ref is not None:
                    from ...tools.move.move_tool import MoveTool as _MT
                    from ...tools.move._enums import _Mode as _MMode
                    _mt = c._tool_manager_ref.active_tool
                    if (
                        isinstance(_mt, _MT)
                        and _mt._dragging
                        and layer.layer_type.name == "SHAPE"
                        and (layer is getattr(_mt, "_active_layer", None) or layer in getattr(_mt, "_multi_layers", []))
                    ):
                        is_multi = layer in getattr(_mt, "_multi_layers", [])
                        if is_multi:
                            # Direct affine mapping handling multi-selection
                            from ...vector.geometry import AffineTransform
                            import math as _math
                            
                            _multi_positions = getattr(_mt, "_multi_positions", {})
                            _multi_dims = getattr(_mt, "_multi_dims", {})
                            _base_sx = getattr(_mt, "_multi_base_sx", {}).get(layer.id, 1.0)
                            _base_sy = getattr(_mt, "_multi_base_sy", {}).get(layer.id, 1.0)
                            _base_angle = getattr(_mt, "_multi_base_angle", {}).get(layer.id, 0.0)
                            
                            _ox, _oy = _multi_positions.get(layer.id, layer.position)
                            _ow, _oh = _multi_dims.get(layer.id, (layer.width, layer.height))
                            
                            _orig_cx = _ox + _ow / 2.0
                            _orig_cy = _oy + _oh / 2.0
                            _new_cx = layer.position[0] + layer.width / 2.0
                            _new_cy = layer.position[1] + layer.height / 2.0
                            
                            xf = AffineTransform.translation(_new_cx, _new_cy)
                            angle_deg = layer.transform_angle - _base_angle
                            if angle_deg != 0.0:
                                xf = xf.rotate(-_math.radians(angle_deg))
                                
                            scale_x = layer.transform_scale_x / max(abs(_base_sx), 1e-6)
                            scale_y = layer.transform_scale_y / max(abs(_base_sy), 1e-6)
                            if _base_sx < 0: scale_x = -scale_x
                            if _base_sy < 0: scale_y = -scale_y
                            if scale_x != 1.0 or scale_y != 1.0:
                                xf = xf.scale(scale_x, scale_y)
                                
                            xf = xf.translate(-_orig_cx, -_orig_cy)
                            
                            def f_multi(dx: float, dy: float) -> QPointF:
                                rx = xf.a * dx + xf.b * dy + xf.tx
                                ry = xf.c * dx + xf.d * dy + xf.ty
                                return QPointF(dr.left() + rx * sx, dr.top() + ry * sy)
                            return f_multi

                        # Single layer handling
                        if _mt._mode == _MMode.MOVE:
                            _ox, _oy = getattr(_mt, "_orig_position", layer.position)
                            _live_dx = layer.position[0] - _ox
                            _live_dy = layer.position[1] - _oy
                        elif _mt._mode == _MMode.RESIZE:
                            _live_old_cx, _live_old_cy = getattr(_mt, "_orig_center", (0, 0))
                            _ow = getattr(_mt, "_orig_width", layer.width)
                            _oh = getattr(_mt, "_orig_height", layer.height)
                            _live_sx = layer.width / max(_ow, 1.0)
                            _live_sy = layer.height / max(_oh, 1.0)
                            _live_new_cx = layer.position[0] + layer.width / 2.0
                            _live_new_cy = layer.position[1] + layer.height / 2.0
                            _apply_live_scale = True

            if has_nd_rotation:
                import math as _math
                _rad = _math.radians(nd_angle)
                _cos_a = _math.cos(_rad)
                _sin_a = _math.sin(_rad)
                _rcx = layer.position[0] + layer.width / 2.0
                _rcy = layer.position[1] + layer.height / 2.0

                def f(dx: float, dy: float) -> QPointF:
                    rx = _rcx + _cos_a * (dx - _rcx) + _sin_a * (dy - _rcy)
                    ry = _rcy - _sin_a * (dx - _rcx) + _cos_a * (dy - _rcy)
                    return QPointF(dr.left() + rx * sx, dr.top() + ry * sy)
                return f
            elif _apply_live_scale:
                def f(dx: float, dy: float) -> QPointF:
                    lx = dx - _live_old_cx
                    ly = dy - _live_old_cy
                    lx *= _live_sx
                    ly *= _live_sy
                    rx = lx + _live_new_cx
                    ry = ly + _live_new_cy
                    return QPointF(dr.left() + rx * sx, dr.top() + ry * sy)
                return f
            elif _live_dx != 0.0 or _live_dy != 0.0:
                def f(dx: float, dy: float) -> QPointF:
                    return QPointF(dr.left() + (dx + _live_dx) * sx,
                                   dr.top() + (dy + _live_dy) * sy)
                return f
            else:
                def f(dx: float, dy: float) -> QPointF:
                    return QPointF(dr.left() + dx * sx, dr.top() + dy * sy)
                return f

        for layer in shape_layers:
            vl = getattr(layer, "_vector_data", None)
            if not vl:
                continue
                
            _to_screen = _get_to_screen_func(layer, dr, sx, sy)

            for obj in vl.objects:
                if not obj.visible:
                    continue
                is_obj_selected = obj.selected
                path = obj.transformed_path()

                for sp in path.sub_paths:
                    if not sp.nodes:
                        continue

                    pp = QPainterPath()
                    origin = sp.nodes[0].position
                    pp.moveTo(_to_screen(origin.x, origin.y))
                    for seg in sp.segments:
                        if seg.seg_type == SegmentType.LINE:
                            pp.lineTo(_to_screen(seg.end.x, seg.end.y))
                        elif seg.seg_type == SegmentType.CUBIC:
                            pp.cubicTo(
                                _to_screen(seg.cp1.x, seg.cp1.y),
                                _to_screen(seg.cp2.x, seg.cp2.y),
                                _to_screen(seg.end.x, seg.end.y),
                            )
                        elif seg.seg_type == SegmentType.CLOSE:
                            pp.closeSubpath()
                    p.setPen(path_pen_sel if is_obj_selected else path_pen)
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawPath(pp)

                    if show_nodes and (is_obj_selected or is_pen_tool):
                        for node in sp.nodes:
                            np_ = _to_screen(node.position.x, node.position.y)
                            node_sel = getattr(node, "selected", False)

                            if node_sel:
                                if node.in_handle and not node.in_handle.approx_eq(node.position):
                                    hp = _to_screen(node.in_handle.x, node.in_handle.y)
                                    p.setPen(handle_pen)
                                    p.drawLine(np_, hp)
                                    p.setPen(QPen(accent, 0.8))
                                    p.setBrush(QColor(100, 180, 255, 150))
                                    p.drawEllipse(hp, 3, 3)
                                if node.out_handle and not node.out_handle.approx_eq(node.position):
                                    hp = _to_screen(node.out_handle.x, node.out_handle.y)
                                    p.setPen(handle_pen)
                                    p.drawLine(np_, hp)
                                    p.setPen(QPen(accent, 0.8))
                                    p.setBrush(QColor(100, 180, 255, 150))
                                    p.drawEllipse(hp, 3, 3)

                            p.setPen(QPen(accent, 1.2))
                            mode = getattr(node, "mode", None)
                            if node_sel:
                                p.setBrush(accent)
                            else:
                                p.setBrush(white)

                            if mode == HandleMode.SHARP:
                                p.drawRect(QRectF(np_.x() - 3.5, np_.y() - 3.5, 7, 7))
                            elif mode == HandleMode.SMOOTH:
                                p.drawEllipse(np_, 3.5, 3.5)
                            elif mode == HandleMode.SYMMETRIC:
                                diamond = QPainterPath()
                                diamond.moveTo(np_.x(), np_.y() - 4)
                                diamond.lineTo(np_.x() + 4, np_.y())
                                diamond.lineTo(np_.x(), np_.y() + 4)
                                diamond.lineTo(np_.x() - 4, np_.y())
                                diamond.closeSubpath()
                                p.drawPath(diamond)
                            else:
                                p.drawRect(QRectF(np_.x() - 3, np_.y() - 3, 6, 6))

        if is_pen_tool and c._tool_manager_ref is not None and active_layer:
            _to_screen = _get_to_screen_func(active_layer, dr, sx, sy)
            from ...vector.pen_tool import PenTool
            tool = c._tool_manager_ref.active_tool
            if isinstance(tool, PenTool) and tool.is_drawing and tool.preview_point:
                nodes = tool.current_nodes
                if nodes:
                    last = nodes[-1]
                    last_screen = _to_screen(last.position.x, last.position.y)
                    preview_screen = _to_screen(tool.preview_point.x, tool.preview_point.y)
                    p.setPen(preview_pen)
                    if last.out_handle and not last.out_handle.approx_eq(last.position):
                        cp1 = _to_screen(last.out_handle.x, last.out_handle.y)
                        preview_pp = QPainterPath()
                        preview_pp.moveTo(last_screen)
                        preview_pp.cubicTo(cp1, preview_screen, preview_screen)
                        p.drawPath(preview_pp)
                    else:
                        p.drawLine(last_screen, preview_screen)

                    if len(nodes) >= 3:
                        from ...vector.geometry import Vec2
                        first = nodes[0]
                        first_screen = _to_screen(first.position.x, first.position.y)
                        dist = Vec2(tool.preview_point.x, tool.preview_point.y).distance_to(first.position)
                        if dist < 8.0:
                            p.setPen(QPen(QColor(50, 200, 100), 1.5))
                            p.setBrush(Qt.BrushStyle.NoBrush)
                            p.drawEllipse(first_screen, 6, 6)

        if is_node_tool and c._tool_manager_ref is not None and active_layer:
            _to_screen = _get_to_screen_func(active_layer, dr, sx, sy)
            from ...vector.node_tool import NodeTool
            tool = c._tool_manager_ref.active_tool
            if isinstance(tool, NodeTool):
                mrect = tool.marquee_rect
                if mrect is not None:
                    s, e = mrect
                    ss = _to_screen(s.x, s.y)
                    se = _to_screen(e.x, e.y)
                    marquee_pen = QPen(accent, 1.0, Qt.PenStyle.DashLine)
                    marquee_pen.setCosmetic(True)
                    p.setPen(marquee_pen)
                    p.setBrush(QColor(100, 180, 255, 30))
                    p.drawRect(QRectF(ss, se).normalized())

                # Node-add hover indicator (plus sign on the segment)
                hover_pt = getattr(tool, "_hover_seg_pos", None)
                if hover_pt is not None:
                    hp = _to_screen(hover_pt.x, hover_pt.y)
                    # Outer ring
                    p.setPen(QPen(accent, 1.5))
                    p.setPen(QPen(accent, 1.5))
                    p.setBrush(white)
                    p.drawEllipse(hp, 5, 5)
                    # Small plus sign inside the circle
                    p.setPen(QPen(accent, 1.4))
                    cross = 2.8
                    p.drawLine(QPointF(hp.x() - cross, hp.y()),
                               QPointF(hp.x() + cross, hp.y()))
                    p.drawLine(QPointF(hp.x(), hp.y() - cross),
                               QPointF(hp.x(), hp.y() + cross))

        p.restore()

    def draw_text_cursor(self, p: QPainter, dr: QRectF) -> None:
        """Draw the blinking text cursor."""
        c = self._canvas
        if c._text_cursor_pos is None:
            return
        cx, cy = c._text_cursor_pos
        h = c._text_cursor_height

        if c._text_box is not None and c._text_box_angle != 0.0:
            bx, by, bw, bh = c._text_box
            box_cx = bx + bw / 2.0
            box_cy = by + bh / 2.0
            wc = c._doc_to_widget(dr, box_cx, box_cy)
            p.save()
            p.translate(wc.x(), wc.y())
            p.rotate(-c._text_box_angle)
            sx = dr.width() / c._doc_w if c._doc_w else 1
            sy = dr.height() / c._doc_h if c._doc_h else 1
            lcx = (bx + cx - box_cx) * sx
            lcy = (by + cy - box_cy) * sy
            lh = h * sy
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(QPointF(lcx, lcy), QPointF(lcx, lcy + lh))
            p.setPen(QPen(QColor(0, 0, 0), 1))
            p.drawLine(QPointF(lcx, lcy), QPointF(lcx, lcy + lh))
            p.restore()
        else:
            top = c._doc_to_widget(dr, cx + (c._text_box[0] if c._text_box else 0),
                                  cy + (c._text_box[1] if c._text_box else 0))
            bot = c._doc_to_widget(dr, cx + (c._text_box[0] if c._text_box else 0),
                                  cy + h + (c._text_box[1] if c._text_box else 0))
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(top, bot)
            p.setPen(QPen(QColor(0, 0, 0), 1))
            p.drawLine(top, bot)

    def draw_text_box(self, p: QPainter, dr: QRectF) -> None:
        """Draw the text bounding box with resize handles."""
        c = self._canvas
        if c._text_box is None or c._doc_w == 0:
            return
        x, y, w, h = c._text_box
        sx = dr.width() / c._doc_w
        sy = dr.height() / c._doc_h

        p.save()
        if c._text_box_angle != 0.0:
            cx = x + w / 2.0
            cy = y + h / 2.0
            wc = c._doc_to_widget(dr, cx, cy)
            p.translate(wc.x(), wc.y())
            p.rotate(-c._text_box_angle)
            hw, hh = w * sx / 2, h * sy / 2
        else:
            tl = c._doc_to_widget(dr, x, y)
            p.translate(tl.x() + w * sx / 2, tl.y() + h * sy / 2)
            hw, hh = w * sx / 2, h * sy / 2

        pen = QPen(QColor(0, 150, 255), 1.5, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(-hw, -hh, hw * 2, hh * 2))

        hs = 6
        handle_pts = [
            (-hw, -hh), (0, -hh), (hw, -hh),
            (-hw, 0), (hw, 0),
            (-hw, hh), (0, hh), (hw, hh),
        ]
        p.setPen(QPen(QColor(0, 150, 255), 1))
        p.setBrush(QColor(255, 255, 255))
        for hx, hy in handle_pts:
            p.drawRect(QRectF(hx - hs / 2, hy - hs / 2, hs, hs))

        p.restore()

    def draw_text_draw_rect(self, p: QPainter, dr: QRectF) -> None:
        """Draw the text box creation preview rectangle."""
        c = self._canvas
        if c._text_draw_rect is None or c._doc_w == 0:
            return
        x, y, w, h = c._text_draw_rect
        tl = c._doc_to_widget(dr, x, y)
        br = c._doc_to_widget(dr, x + w, y + h)
        pen = QPen(QColor(0, 150, 255), 2.0, Qt.PenStyle.SolidLine)
        p.setPen(pen)
        p.setBrush(QColor(0, 150, 255, 40))
        p.drawRect(QRectF(tl, br))

    def draw_text_selection(self, p: QPainter, dr: QRectF) -> None:
        """Draw text selection highlight rectangles."""
        c = self._canvas
        if not c._text_selection_rects or not c._text_box:
            return
        bx, by = c._text_box[0], c._text_box[1]
        sx = dr.width() / c._doc_w if c._doc_w else 1
        sy = dr.height() / c._doc_h if c._doc_h else 1

        p.save()
        if c._text_box_angle != 0.0:
            bw, bh = c._text_box[2], c._text_box[3]
            cx = bx + bw / 2.0
            cy = by + bh / 2.0
            wc = c._doc_to_widget(dr, cx, cy)
            p.translate(wc.x(), wc.y())
            p.rotate(-c._text_box_angle)
            offset_x = -bw / 2.0 * sx
            offset_y = -bh / 2.0 * sy
        else:
            tl = c._doc_to_widget(dr, bx, by)
            offset_x = tl.x()
            offset_y = tl.y()

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(60, 130, 220, 80))
        for rx, ry, rw, rh in c._text_selection_rects:
            p.drawRect(QRectF(offset_x + rx * sx, offset_y + ry * sy, rw * sx, rh * sy))
        p.restore()

    # ---- Boolean preview overlay ------------------------------------------

    def draw_boolean_preview(self, p: QPainter, dr: QRectF) -> None:
        """Draw semi-transparent preview of a hovered boolean operation.

        The canvas stores the preview state in:
        * ``_bool_preview_path``  — ``VectorPath`` result of the operation
        * ``_bool_source_ids``    — set of layer IDs to outline in red dashes
        """
        c = self._canvas
        preview_path = getattr(c, "_bool_preview_path", None)
        source_ids = getattr(c, "_bool_source_ids", None)
        if preview_path is None or c._doc_w == 0 or c._doc_h == 0:
            return

        sx = dr.width() / c._doc_w
        sy = dr.height() / c._doc_h

        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Draw the result shape as blue semi-transparent fill
        from ...vector.path import SegmentType
        for sp in preview_path.sub_paths:
            if not sp.nodes:
                continue
            pp = QPainterPath()
            origin = sp.nodes[0].position
            pp.moveTo(QPointF(dr.left() + origin.x * sx, dr.top() + origin.y * sy))
            for seg in sp.segments:
                ex = dr.left() + seg.end.x * sx
                ey = dr.top() + seg.end.y * sy
                if seg.seg_type == SegmentType.LINE:
                    pp.lineTo(QPointF(ex, ey))
                elif seg.seg_type == SegmentType.CUBIC:
                    c1x = dr.left() + seg.cp1.x * sx
                    c1y = dr.top() + seg.cp1.y * sy
                    c2x = dr.left() + seg.cp2.x * sx
                    c2y = dr.top() + seg.cp2.y * sy
                    pp.cubicTo(QPointF(c1x, c1y), QPointF(c2x, c2y), QPointF(ex, ey))
                elif seg.seg_type == SegmentType.CLOSE:
                    pp.closeSubpath()
            p.setPen(QPen(QColor(60, 130, 220, 140), 1.5))
            p.setBrush(QColor(60, 130, 220, 50))
            p.drawPath(pp)

        # Draw red dashed outlines for source layers that will be removed
        if source_ids:
            doc = c._doc_ref
            if doc:
                dash_pen = QPen(QColor(220, 60, 60, 180), 1.5, Qt.PenStyle.DashLine)
                dash_pen.setCosmetic(True)
                p.setPen(dash_pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                for layer in doc.layers.layers:
                    if layer.id not in source_ids:
                        continue
                    vl = getattr(layer, "_vector_data", None)
                    if vl is None:
                        continue
                    for obj in vl.objects:
                        path = obj.transformed_path()
                        for sp2 in path.sub_paths:
                            if not sp2.nodes:
                                continue
                            pp2 = QPainterPath()
                            o2 = sp2.nodes[0].position
                            pp2.moveTo(QPointF(dr.left() + o2.x * sx,
                                               dr.top() + o2.y * sy))
                            for seg2 in sp2.segments:
                                ex2 = dr.left() + seg2.end.x * sx
                                ey2 = dr.top() + seg2.end.y * sy
                                if seg2.seg_type == SegmentType.LINE:
                                    pp2.lineTo(QPointF(ex2, ey2))
                                elif seg2.seg_type == SegmentType.CUBIC:
                                    c1 = QPointF(dr.left() + seg2.cp1.x * sx,
                                                 dr.top() + seg2.cp1.y * sy)
                                    c2 = QPointF(dr.left() + seg2.cp2.x * sx,
                                                 dr.top() + seg2.cp2.y * sy)
                                    pp2.cubicTo(c1, c2, QPointF(ex2, ey2))
                                elif seg2.seg_type == SegmentType.CLOSE:
                                    pp2.closeSubpath()
                            p.drawPath(pp2)
        p.restore()

    # ---- Pick-segments overlay --------------------------------------------

    def draw_pick_segments(self, p: QPainter, dr: QRectF) -> None:
        """Draw pick-segments mode overlay: segments, hover glow, chip."""
        c = self._canvas
        pick_state = getattr(c, "_pick_segments_state", None)
        if pick_state is None or not pick_state.active:
            return
        if c._doc_w == 0 or c._doc_h == 0:
            return

        sx = dr.width() / c._doc_w
        sy = dr.height() / c._doc_h

        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        for seg in pick_state.segments:
            if len(seg.points) < 2:
                continue

            is_hovered = seg.id == pick_state.hovered_id

            if seg.included:
                r, g, b, a = seg.color
                pen_w = 3.5 if is_hovered else 2.0
                pen = QPen(QColor(r, g, b, a), pen_w)
            else:
                pen = QPen(QColor(120, 120, 120, 150), 1.5, Qt.PenStyle.DashLine)
                if is_hovered:
                    pen.setWidthF(2.5)

            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)

            pp = QPainterPath()
            p0 = seg.points[0]
            pp.moveTo(QPointF(dr.left() + p0.x * sx, dr.top() + p0.y * sy))
            for pt in seg.points[1:]:
                pp.lineTo(QPointF(dr.left() + pt.x * sx, dr.top() + pt.y * sy))
            p.drawPath(pp)

        # Live result preview: build included segments into a filled shape
        included = [s for s in pick_state.segments if s.included]
        if included:
            preview_pen = QPen(QColor(60, 180, 120, 100), 1.0)
            preview_pen.setCosmetic(True)
            p.setPen(preview_pen)
            p.setBrush(QColor(60, 180, 120, 30))
            for seg in included:
                if len(seg.points) < 2:
                    continue
                pp = QPainterPath()
                p0 = seg.points[0]
                pp.moveTo(QPointF(dr.left() + p0.x * sx, dr.top() + p0.y * sy))
                for pt in seg.points[1:]:
                    pp.lineTo(QPointF(dr.left() + pt.x * sx, dr.top() + pt.y * sy))
                p.drawPath(pp)

        # Status chip
        closed_n = pick_state.closed_count()
        has_open = pick_state.has_open()
        if closed_n > 0 and not has_open:
            chip_text = f"Ready \u2014 {closed_n} shape(s) found"
            chip_color = QColor(40, 160, 80, 200)
        else:
            chip_text = "Open path \u2014 keep selecting"
            chip_color = QColor(200, 140, 40, 200)

        font = p.font()
        font.setPixelSize(11)
        p.setFont(font)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(chip_text) + 16
        th = fm.height() + 8
        chip_rect = QRectF(dr.left() + 10, dr.top() + 10, tw, th)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(chip_color)
        p.drawRoundedRect(chip_rect, 4, 4)
        p.setPen(QColor(255, 255, 255))
        p.drawText(chip_rect, Qt.AlignmentFlag.AlignCenter, chip_text)

        p.restore()

    def draw_snap_lines(self, p, dr) -> None:
        """Draw the alignment lines produced by a snapped drag.

        Each line spans only the two objects whose relationship it
        represents, so the overlay reads as "these two edges line up"
        rather than as a full-canvas grid.
        """
        c = self.canvas
        if not c._snap_lines or not c._doc_w or not c._doc_h:
            return
        from ...core.snapping import SnapSource

        sx = dr.width() / c._doc_w
        sy = dr.height() / c._doc_h

        colours = {
            SnapSource.CANVAS: QColor(255, 90, 160),
            SnapSource.LAYER: QColor(255, 90, 160),
            SnapSource.GUIDE: QColor(90, 200, 255),
        }
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for line in c._snap_lines:
            pen = QPen(colours.get(line.source, QColor(255, 90, 160)), 1.0)
            pen.setCosmetic(True)
            p.setPen(pen)
            if line.vertical:
                x = dr.left() + line.position * sx
                p.drawLine(QPointF(x, dr.top() + line.start * sy),
                           QPointF(x, dr.top() + line.end * sy))
            else:
                y = dr.top() + line.position * sy
                p.drawLine(QPointF(dr.left() + line.start * sx, y),
                           QPointF(dr.left() + line.end * sx, y))
        p.restore()
