"""Fast vector rasteriser using QPainter.
"""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np
from PySide6.QtGui import (
    QPainter, QImage, QColor, QBrush, QPen, QLinearGradient, QRadialGradient,
    QPainterPath, QTransform
)
from PySide6.QtCore import Qt, QPointF, QRectF

from .geometry import Vec2, BBox, AffineTransform
from .path import VectorPath, SubPath, FillRule
from .style import (
    VectorStyle, VectorFill, VectorStroke,
    SolidPaint, GradientPaint, GradientType, PatternPaint,
    StrokeCap, StrokeJoin, StrokeAlign,
)
from .scene import VectorObject, VectorLayer

__all__ = ["VectorRasterizer"]


class VectorRasterizer:
    """Rasterises a ``VectorLayer`` using QPainter execution."""

    def __init__(self, tile_size: int = 256) -> None:
        self.tile_size = tile_size
        self._cache: dict[str, tuple[str, tuple[float, float], QImage]] = {}
        self._max_cache_entries: int = 128

    def rasterize_layer(
        self,
        vector_layer: VectorLayer,
        width: int,
        height: int,
        viewport: BBox | None = None,
        zoom: float = 1.0,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Render entire layer to a float32 RGBA buffer (0..1)."""
        
        # Create a master QImage to render into
        # Format_RGBA8888_Premultiplied is standard for QPainter
        master_img = QImage(width, height, QImage.Format.Format_RGBA8888_Premultiplied)
        master_img.fill(0) # Clear
        
        painter = QPainter(master_img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # We need to handle the origin offset.
        # origin is the world coordinate at (0, 0) of the image.
        # So we translate by -origin.
        painter.translate(-origin[0], -origin[1])
        
        for obj in vector_layer.objects:
            if not obj.visible:
                continue
                
            # Check cache
            state_hash = self._object_state_hash(obj)
            cached = self._cache.get(obj.id)
            
            # For now, we don't use the per-object image cache for composition because 
            # we want to render strictly back-to-front with correct blending in one pass if possible.
            # However, if we want to cache expensive renders (like complex paths), 
            # we can render them to small QImages.
            # Since QPainter is fast, let's just render directly.
            # If we really need caching, we'd draw the cached pixmap.
            
            self._render_object(painter, obj)

        painter.end()
        
        # Convert QImage to numpy float32
        ptr = master_img.bits()
        # QImage bits are aligned differently? RGBA8888 is usually packed.
        # We rely on numpy's frombuffer.
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
        
        # Convert to float 0..1
        return arr.astype(np.float32) / 255.0

    def rasterize_object(
        self,
        obj: VectorObject,
        width: int,
        height: int,
        zoom: float = 1.0,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        img = QImage(width, height, QImage.Format.Format_RGBA8888_Premultiplied)
        img.fill(0)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.translate(-origin[0], -origin[1])
        self._render_object(p, obj)
        p.end()
        
        arr = np.frombuffer(img.bits(), dtype=np.uint8).reshape((height, width, 4))
        return arr.astype(np.float32) / 255.0

    def invalidate(self, obj_id: str | None = None) -> None:
        if obj_id is None:
            self._cache.clear()
        else:
            self._cache.pop(obj_id, None)
            
    def clear_cache(self) -> None:
        self._cache.clear()

    @staticmethod
    def _object_state_hash(obj: VectorObject) -> str:
        """Compute a hash of the object's visual state for cache comparison."""
        h = hashlib.md5(usedforsecurity=False)
        # Transform
        xf = obj.transform
        h.update(f"T{xf.a:.8f},{xf.b:.8f},{xf.c:.8f},{xf.d:.8f},{xf.tx:.8f},{xf.ty:.8f}".encode())
        # Style summary
        s = obj.style
        for i, f in enumerate(s.fills):
            h.update(f"F{i}{f.visible}{f.opacity:.4f}".encode())
            if isinstance(f.paint, SolidPaint):
                h.update(f"S{f.paint.color}".encode())
        for i, st in enumerate(s.strokes):
            h.update(f"K{i}{st.visible}{st.opacity:.4f}{st.width:.4f}{st.cap.value}{st.join.value}".encode())
            if isinstance(st.paint, SolidPaint):
                h.update(f"S{st.paint.color}".encode())
        # Path geometry (hash node positions)
        path = obj.effective_path()
        for sp in path.sub_paths:
            for n in sp.nodes:
                h.update(f"N{n.position.x:.6f},{n.position.y:.6f}".encode())
                if n.in_handle:
                    h.update(f"I{n.in_handle.x:.6f},{n.in_handle.y:.6f}".encode())
                if n.out_handle:
                    h.update(f"O{n.out_handle.x:.6f},{n.out_handle.y:.6f}".encode())
            h.update(f"C{sp.closed}".encode())
        return h.hexdigest()

    def _render_object(self, painter: QPainter, obj: VectorObject) -> None:
        painter.save()
        
        # Apply object transform
        # obj.transform is our AffineTransform. Convert to QTransform.
        qt = obj.transform.to_qtransform()
        painter.setTransform(qt, combine=True)
        
        path = obj.effective_path().qpath
        style = obj.style
        
        # Render fills
        for fill in style.fills:
            if not fill.visible or fill.opacity <= 0:
                continue
                
            brush = self._create_brush(fill.paint, fill.opacity, obj)
            painter.setBrush(brush)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(path)
            
        # Render strokes
        for stroke in style.strokes:
            if not stroke.visible or stroke.opacity <= 0 or stroke.width <= 0:
                continue
                
            pen = self._create_pen(stroke, obj)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            
        painter.restore()

    def _create_brush(
        self,
        paint: object,
        opacity: float,
        obj: "VectorObject | None" = None,
    ) -> QBrush:
        if isinstance(paint, SolidPaint):
            c = paint.color
            color = QColor.fromRgbF(c[0], c[1], c[2], opacity)
            return QBrush(color)
            
        elif isinstance(paint, GradientPaint):
            # Use the path's local bbox for gradient coords so the gradient
            # always aligns with and covers the path (fixes offset + single-color).
            bb = obj.local_bbox() if obj else None
            if bb is not None and not bb.is_empty:
                if paint.gradient_type == GradientType.LINEAR:
                    start_pt = bb.min_pt
                    end_pt = bb.max_pt
                else:
                    start_pt = bb.center
                    end_pt = bb.center
                    radius = max(bb.width, bb.height) / 2 * 1.5
            else:
                start_pt = paint.start
                end_pt = paint.end
                radius = paint.radius
            
            if paint.gradient_type == GradientType.LINEAR:
                grad = QLinearGradient(start_pt.to_qpoint(), end_pt.to_qpoint())
            else:
                grad = QRadialGradient(start_pt.to_qpoint(), radius)
                grad.setCenter(start_pt.to_qpoint())
                grad.setFocalPoint(end_pt.to_qpoint())
                
            for stop in paint.stops:
                qc = QColor.fromRgbF(
                    stop.color[0], stop.color[1], stop.color[2],
                    opacity * stop.color[3],
                )
                grad.setColorAt(stop.offset, qc)
                
            return QBrush(grad)
            
        return QBrush(Qt.BrushStyle.SolidPattern)

    def _create_pen(
        self,
        stroke: VectorStroke,
        obj: "VectorObject | None" = None,
    ) -> QPen:
        brush = self._create_brush(stroke.paint, stroke.opacity, obj)
        pen = QPen(brush, stroke.width)
        
        # Cap
        if stroke.cap == StrokeCap.BUTT:
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        elif stroke.cap == StrokeCap.ROUND:
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        elif stroke.cap == StrokeCap.SQUARE:
            pen.setCapStyle(Qt.PenCapStyle.SquareCap)
            
        # Join
        if stroke.join == StrokeJoin.MITER:
            pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            pen.setMiterLimit(stroke.miter_limit)
        elif stroke.join == StrokeJoin.ROUND:
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        elif stroke.join == StrokeJoin.BEVEL:
            pen.setJoinStyle(Qt.PenJoinStyle.BevelJoin)
            
        return pen


# Keep the global helpers for compatibility
_shared_rasterizer = VectorRasterizer()
_auto_rasterize_enabled: bool = True

def set_auto_rasterize(enabled: bool) -> None:
    global _auto_rasterize_enabled
    _auto_rasterize_enabled = enabled

def get_auto_rasterize() -> bool:
    return _auto_rasterize_enabled

def rasterize_vector_layer_tight(
    doc: object, *, layer: object | None = None, force: bool = False,
) -> None:
    # Retain the tight bounding box logic using QPainter
    if not force and not _auto_rasterize_enabled:
        return

    if layer is None:
        try:
            stack = getattr(doc, "layers", None)
            if stack:
                layer = stack.active_layer
        except AttributeError:
            pass
            
    if layer is None:
        return
        
    vl = getattr(layer, "_vector_data", None)
    if vl is None:
        return

    union = BBox.empty()
    for obj in vl.objects:
        if obj.visible:
            union = union.union(obj.bbox())

    if union.is_empty:
        # Through the property: it moves content_version (history shares
        # buffers by version) and keeps layer.width/height in step.
        layer.pixels = np.zeros((1, 1, 4), dtype=np.float32)
        layer.position = (0, 0)
        layer._pixels_dirty = False
        return

    x0 = int(math.floor(union.min_pt.x)) - 2
    y0 = int(math.floor(union.min_pt.y)) - 2
    x1 = int(math.ceil(union.max_pt.x)) + 3
    y1 = int(math.ceil(union.max_pt.y)) + 3
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    pixels = _shared_rasterizer.rasterize_layer(
        vl, bw, bh, origin=(float(x0), float(y0))
    )
    # Assign via property to update width/height automatically
    layer.pixels = pixels
    layer.position = (x0, y0)
    layer._pixels_dirty = False

    # Update parent group bbox when vector layer content changes
    if getattr(layer, "parent_id", None):
        stack = getattr(doc, "layers", None)
        if stack is not None:
            from ..core.enums import LayerType
            parent = stack.get(layer.parent_id)
            if parent is not None and getattr(parent, "layer_type", None) == LayerType.GROUP:
                stack.update_group_bbox(parent)