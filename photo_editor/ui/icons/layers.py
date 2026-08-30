"""Central layer-panel icon builders."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap

from ..theme import ThemeManager


# Icons are rebuilt from scratch on every layer row of every panel refresh --
# a QPixmap, a QPainter and an antialiased QPainterPath each time, twenty
# times per refresh for a twenty-layer document. They depend only on their
# boolean state and the active palette, so they are cached on both.
_ICON_CACHE: dict[tuple, QIcon] = {}


def _cached_icon(kind: str, state) -> QIcon | None:
    return _ICON_CACHE.get(
        (kind, state, ThemeManager.instance().active_theme_name))


def _store_icon(kind: str, state, icon: QIcon) -> QIcon:
    _ICON_CACHE[(kind, state,
                 ThemeManager.instance().active_theme_name)] = icon
    return icon


def clear_icon_cache() -> None:
    """Drop cached icons (the palette they were drawn with has changed)."""
    _ICON_CACHE.clear()


def _draw_icon(size: int, draw_fn) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    draw_fn(painter, size)
    painter.end()
    return QIcon(pixmap)


def _tb_icon(draw_fn, size: int = 16) -> QIcon:
    return _draw_icon(size, draw_fn)


def icon_eye(visible: bool) -> QIcon:
    cached = _cached_icon("eye", visible)
    if cached is not None:
        return cached

    def _draw(p: QPainter, s: int):
        cx, cy = s / 2, s / 2
        palette = ThemeManager.instance().active_palette
        col_active = palette["fg"]
        col_inactive = palette["fg_dim"]
        col = QColor(col_active) if visible else QColor(col_inactive)
        p.setPen(QPen(col, 1.4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        eye = QPainterPath()
        eye.moveTo(2, cy)
        eye.quadTo(cx, 3, s - 2, cy)
        eye.quadTo(cx, s - 3, 2, cy)
        p.drawPath(eye)
        if visible:
            p.setBrush(col)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(cx, cy), 2.8, 2.8)
        else:
            p.setPen(QPen(QColor(170, 70, 70), 1.5))
            p.drawLine(QPointF(4, s - 4), QPointF(s - 4, 4))

    return _store_icon("eye", visible, _draw_icon(18, _draw))


def icon_lock(locked: bool) -> QIcon:
    cached = _cached_icon("lock", locked)
    if cached is not None:
        return cached

    def _draw(p: QPainter, s: int):
        palette = ThemeManager.instance().active_palette
        col_active = palette["fg"]
        col_inactive = palette["fg_dim"]
        col = QColor(col_active) if locked else QColor(col_inactive)
        cx = s / 2
        p.setPen(QPen(col, 1.4))
        bw, bh = 10.0, 6.0
        bx, by = (s - bw) / 2, s - bh - 2
        p.setBrush(col if locked else Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(bx, by, bw, bh), 1.5, 1.5)
        p.setBrush(Qt.BrushStyle.NoBrush)
        sw, sx = 6.0, (s - 6.0) / 2
        shackle = QPainterPath()
        shackle.moveTo(sx, by)
        shackle.lineTo(sx, by - 3)
        shackle.quadTo(sx, by - 6, cx, by - 6)
        shackle.quadTo(sx + sw, by - 6, sx + sw, by - 3)
        shackle.lineTo(sx + sw, by if locked else by - 5)
        p.drawPath(shackle)

    return _store_icon("lock", locked, _draw_icon(18, _draw))


def icon_mask(has_mask: bool) -> QIcon:
    cached = _cached_icon("mask", has_mask)
    if cached is not None:
        return cached
    if not has_mask:
        return _store_icon("mask", False, QIcon(QPixmap(18, 18)))

    def _draw(p: QPainter, s: int):
        p.setPen(QPen(QColor(180, 180, 180), 1.2))
        p.setBrush(QColor(180, 180, 180, 50))
        p.drawEllipse(QRectF(2, 2, s - 4, s - 4))

    return _store_icon("mask", True, _draw_icon(18, _draw))


def ico_new_layer() -> QIcon:
    def _d(p, s):
        palette = ThemeManager.instance().active_palette
        col = QColor(palette["fg"])
        p.setPen(QPen(col, 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(2, 4, s - 4, s - 6), 1, 1)
        p.drawLine(QPointF(s - 5, 4), QPointF(s - 5, 7))
        p.drawLine(QPointF(s - 5, 7), QPointF(s - 2, 7))

    return _tb_icon(_d)


def ico_fx() -> QIcon:
    def _d(p, s):
        palette = ThemeManager.instance().active_palette
        p.setPen(QPen(QColor(palette["fg"]), 1.4))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.drawText(QRectF(0, 0, s, s), Qt.AlignmentFlag.AlignCenter, "fx")

    return _tb_icon(_d)


def ico_mask() -> QIcon:
    def _d(p, s):
        palette = ThemeManager.instance().active_palette
        p.setPen(QPen(QColor(palette["fg"]), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(2, 2, s - 4, s - 4))

    return _tb_icon(_d)


def ico_mask_layer() -> QIcon:
    def _d(p, s):
        cx, cy, r = s / 2, s / 2, s / 2 - 2
        palette = ThemeManager.instance().active_palette
        p.setPen(QPen(QColor(palette["fg"]), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)
        clip = QPainterPath()
        clip.addRect(QRectF(cx, 0, cx, s))
        p.setClipPath(clip)
        p.setBrush(QColor(palette["fg"]))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), r, r)

    return _tb_icon(_d)


def ico_adjustment() -> QIcon:
    def _d(p, s):
        cx, cy, r = s / 2, s / 2, s / 2 - 2
        palette = ThemeManager.instance().active_palette
        p.setPen(QPen(QColor(palette["fg"]), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)
        clip = QPainterPath()
        clip.addRect(QRectF(0, 0, cx, s))
        p.setClipPath(clip)
        p.setBrush(QColor(palette["fg"]))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), r, r)

    return _tb_icon(_d)


def ico_filter() -> QIcon:
    def _d(p, s):
        palette = ThemeManager.instance().active_palette
        col = QColor(palette["fg"])
        p.setPen(QPen(col, 1.4))
        cx, cy = s / 2, s / 2
        r = s / 2 - 2
        pts = [
            QPointF(cx, cy - r),
            QPointF(cx + r * 0.35, cy - r * 0.35),
            QPointF(cx + r, cy),
            QPointF(cx + r * 0.35, cy + r * 0.35),
            QPointF(cx, cy + r),
            QPointF(cx - r * 0.35, cy + r * 0.35),
            QPointF(cx - r, cy),
            QPointF(cx - r * 0.35, cy - r * 0.35),
        ]
        path = QPainterPath()
        path.moveTo(pts[0])
        for pt in pts[1:]:
            path.lineTo(pt)
        path.closeSubpath()
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    return _tb_icon(_d)


def ico_text() -> QIcon:
    def _d(p, s):
        palette = ThemeManager.instance().active_palette
        col = QColor(palette["fg"])
        p.setPen(QPen(col, 1.6))
        p.setFont(QFont("Segoe UI", int(s * 0.6), QFont.Weight.Bold))
        p.drawText(QRectF(0, 0, s, s), Qt.AlignmentFlag.AlignCenter, "T")

    return _tb_icon(_d)


def ico_chain() -> QIcon:
    def _d(p, s):
        palette = ThemeManager.instance().active_palette
        col = QColor(palette["fg"])
        p.setPen(QPen(col, 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(3, 2, s - 6, 5), 2, 2)
        p.drawRoundedRect(QRectF(3, s - 7, s - 6, 5), 2, 2)
        p.drawLine(QPointF(s / 2, 7), QPointF(s / 2, s - 7))

    return _tb_icon(_d)


def ico_eraser() -> QIcon:
    def _d(p, s):
        palette = ThemeManager.instance().active_palette
        p.setPen(QPen(QColor(palette["fg"]), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(QPointF(4, s - 3), QPointF(s - 4, 3))
        p.drawLine(QPointF(s - 6, s - 3), QPointF(s - 2, s - 3))

    return _tb_icon(_d)


def ico_folder() -> QIcon:
    def _d(p, s):
        palette = ThemeManager.instance().active_palette
        col = QColor(palette["fg"])
        p.setPen(QPen(col, 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath()
        path.moveTo(2, 5)
        path.lineTo(2, s - 3)
        path.lineTo(s - 2, s - 3)
        path.lineTo(s - 2, 6)
        path.lineTo(s / 2 + 1, 6)
        path.lineTo(s / 2 - 1, 4)
        path.lineTo(2, 4)
        path.closeSubpath()
        p.drawPath(path)

    return _tb_icon(_d)


def ico_duplicate() -> QIcon:
    def _d(p, s):
        palette = ThemeManager.instance().active_palette
        col = QColor(palette["fg"])
        p.setPen(QPen(col, 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(1, 3, s - 5, s - 5), 1, 1)
        p.drawRoundedRect(QRectF(4, 1, s - 5, s - 5), 1, 1)

    return _tb_icon(_d)


def ico_move() -> QIcon:
    def _d(p, s):
        palette = ThemeManager.instance().active_palette
        col = QColor(palette["fg"])
        p.setPen(QPen(col, 1.4))
        cx, cy = s / 2, s / 2
        p.drawLine(QPointF(cx, 2), QPointF(cx, s - 2))
        p.drawLine(QPointF(2, cy), QPointF(s - 2, cy))
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            tip = QPointF(cx + dx * (cx - 2), cy + dy * (cy - 2))
            p.drawLine(tip, QPointF(tip.x() - dx * 3 + dy * 2, tip.y() - dy * 3 + dx * 2))
            p.drawLine(tip, QPointF(tip.x() - dx * 3 - dy * 2, tip.y() - dy * 3 - dx * 2))

    return _tb_icon(_d)


def ico_grid() -> QIcon:
    def _d(p, s):
        palette = ThemeManager.instance().active_palette
        col = QColor(palette["fg"])
        p.setPen(QPen(col, 1.0))
        t = 3
        for r in range(3):
            for c in range(3):
                x = t + c * (s - 2 * t) / 2
                y = t + r * (s - 2 * t) / 2
                w = (s - 2 * t) / 2 - 1
                p.drawRect(QRectF(x, y, w, w))

    return _tb_icon(_d)


def ico_trash() -> QIcon:
    def _d(p, s):
        palette = ThemeManager.instance().active_palette
        col = QColor(palette["fg"])
        p.setPen(QPen(col, 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(QPointF(3, 5), QPointF(s - 3, 5))
        p.drawLine(QPointF(s / 2 - 2, 5), QPointF(s / 2 - 2, 3))
        p.drawLine(QPointF(s / 2 - 2, 3), QPointF(s / 2 + 2, 3))
        p.drawLine(QPointF(s / 2 + 2, 3), QPointF(s / 2 + 2, 5))
        p.drawLine(QPointF(4, 5), QPointF(5, s - 2))
        p.drawLine(QPointF(5, s - 2), QPointF(s - 5, s - 2))
        p.drawLine(QPointF(s - 5, s - 2), QPointF(s - 4, 5))

    return _tb_icon(_d)


def ico_settings() -> QIcon:
    def _d(p, s):
        palette = ThemeManager.instance().active_palette
        col = QColor(palette["fg"])
        p.setPen(QPen(col, 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        cx, cy = s / 2, s / 2
        p.drawEllipse(QPointF(cx, cy), 3, 3)
        for i in range(8):
            a = math.radians(i * 45)
            inner, outer = 4.5, 6.5
            p.drawLine(
                QPointF(cx + inner * math.cos(a), cy + inner * math.sin(a)),
                QPointF(cx + outer * math.cos(a), cy + outer * math.sin(a)),
            )

    return _tb_icon(_d)


__all__ = [
    "icon_eye",
    "icon_lock",
    "icon_mask",
    "ico_adjustment",
    "ico_chain",
    "ico_duplicate",
    "ico_eraser",
    "ico_filter",
    "ico_folder",
    "ico_fx",
    "ico_grid",
    "ico_mask",
    "ico_mask_layer",
    "ico_move",
    "ico_new_layer",
    "ico_settings",
    "ico_text",
    "ico_trash",
]
