"""Thumbnail generation for layer previews — lazy + cached."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap

from ....core.document import Document
from .base import THUMB_SIZE


_THUMB_CHECKER: QPixmap | None = None

# ---- LRU thumbnail cache ---------------------------------------------------
# Key: (layer id, content version), Value: QPixmap
#
# The version is part of the key deliberately. Keying on layer.id alone meant
# a thumbnail was generated once and never again: invalidate_thumbnail()
# existed but had no callers anywhere, so every layer's thumbnail silently
# went stale the moment it was painted on. Including the content version
# makes invalidation automatic and impossible to forget.
_THUMB_CACHE_MAX = 256
_thumb_cache: OrderedDict[tuple, QPixmap] = OrderedDict()


def _layer_key(layer) -> tuple:
    return (layer.id, getattr(layer, "content_version", 0))


def _group_key(document, group) -> tuple:
    """Groups depend on their children, so fold their versions in too."""
    versions = tuple(
        (l.id, getattr(l, "content_version", 0), l.visible, l.opacity)
        for l in document.layers if l.parent_id == group.id
    )
    return (group.id, getattr(group, "content_version", 0), versions)


def invalidate_thumbnail(layer_id: str) -> None:
    """Drop every cached thumbnail for *layer_id*, at any version."""
    for key in [k for k in _thumb_cache if k[0] == layer_id]:
        _thumb_cache.pop(key, None)


def invalidate_all_thumbnails() -> None:
    """Clear the entire thumbnail cache (e.g. after theme change)."""
    _thumb_cache.clear()


def _cache_put(key: tuple, pm: QPixmap) -> None:
    _thumb_cache[key] = pm
    _thumb_cache.move_to_end(key)
    while len(_thumb_cache) > _THUMB_CACHE_MAX:
        _thumb_cache.popitem(last=False)


def thumb_checker(size: int = THUMB_SIZE) -> QPixmap:
    global _THUMB_CHECKER
    if _THUMB_CHECKER is None or _THUMB_CHECKER.width() != size:
        _THUMB_CHECKER = QPixmap(size, size)
        _THUMB_CHECKER.fill(QColor(42, 42, 42))
        tp = QPainter(_THUMB_CHECKER)
        tp.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        cs = 4
        light, dark = QColor(70, 70, 70), QColor(50, 50, 50)
        for r in range(0, size, cs):
            for c in range(0, size, cs):
                tp.fillRect(c, r, cs, cs, light if (r // cs + c // cs) % 2 == 0 else dark)
        tp.end()
    return _THUMB_CHECKER


def pixels_to_thumbnail_pixmap(px: np.ndarray, size: int = THUMB_SIZE) -> QPixmap:
    """Convert float32 RGBA pixels to a centered thumbnail QPixmap on checkerboard."""
    pm = QPixmap(thumb_checker(size))
    if px is None or px.size == 0:
        return pm
    h, w = px.shape[:2]
    if h > size * 4 or w > size * 4:
        step_h = max(1, h // (size * 2))
        step_w = max(1, w // (size * 2))
        px = px[::step_h, ::step_w]
        h, w = px.shape[:2]
    buf = np.empty((h, w, 4), dtype=np.uint8)
    np.multiply(px[:, :, 2:3], 255, out=buf[:, :, 0:1], casting='unsafe')
    np.multiply(px[:, :, 1:2], 255, out=buf[:, :, 1:2], casting='unsafe')
    np.multiply(px[:, :, 0:1], 255, out=buf[:, :, 2:3], casting='unsafe')
    np.multiply(px[:, :, 3:4], 255, out=buf[:, :, 3:4], casting='unsafe')
    np.clip(buf, 0, 255, out=buf)
    img = QImage(buf.data, w, h, w * 4, QImage.Format.Format_ARGB32)
    scaled = img.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.FastTransformation)
    tp = QPainter(pm)
    ox = (size - scaled.width()) // 2
    oy = (size - scaled.height()) // 2
    tp.drawImage(ox, oy, scaled)
    tp.end()
    return pm


def make_thumbnail(layer, size: int = THUMB_SIZE) -> QPixmap:
    """Generate (or return cached) a small QPixmap thumbnail for *layer*."""
    key = _layer_key(layer)
    cached = _thumb_cache.get(key)
    if cached is not None:
        _thumb_cache.move_to_end(key)
        return cached
    pm = QPixmap(thumb_checker(size))
    try:
        px = layer.pixels
        if px is not None and px.size > 0:
            pm = pixels_to_thumbnail_pixmap(px, size)
    except Exception:
        pass
    _cache_put(key, pm)
    return pm


def make_group_thumbnail(document: Document, group, size: int = THUMB_SIZE) -> QPixmap:
    """Generate (or return cached) a thumbnail for a group layer."""
    key = _group_key(document, group)
    cached = _thumb_cache.get(key)
    if cached is not None:
        _thumb_cache.move_to_end(key)
        return cached
    pm = QPixmap(thumb_checker(size))
    try:
        from ....engine.compositor import Compositor
        px = Compositor().composite_group_tight(group, document.layers)
        if px is not None and px.size > 0:
            pm = pixels_to_thumbnail_pixmap(px, size)
    except Exception:
        pass
    _cache_put(key, pm)
    return pm
