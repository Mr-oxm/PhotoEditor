from __future__ import annotations

import numpy as np

from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QColor, QPixmap, QPainter, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
)

from ...core.document import Document
from ..styles import render_qss
from .layers.icons import icon_eye
from ..theme import ThemeManager
from .layers.thumbnails import thumb_checker

def channel_pixels_to_pixmap(px: np.ndarray, channel_idx: int, size: int = 32) -> QPixmap:
    pm = QPixmap(thumb_checker(size))
    if px is None or px.size == 0:
        return pm
    h, w = px.shape[:2]
    if h > size * 4 or w > size * 4:
        step_h = max(1, h // (size * 2))
        step_w = max(1, w // (size * 2))
        px = px[::step_h, ::step_w]
        h, w = px.shape[:2]

    ch_data = px[:, :, channel_idx]
    ch_data_u8 = np.empty((h, w), dtype=np.uint8)
    np.multiply(ch_data, 255, out=ch_data_u8, casting='unsafe')
    np.clip(ch_data_u8, 0, 255, out=ch_data_u8)
    
    buf = np.zeros((h, w, 4), dtype=np.uint8)
    # ARGB32 format in Qt is BGRA in memory
    if channel_idx == 0: # Red
        buf[:, :, 2] = ch_data_u8
    elif channel_idx == 1: # Green
        buf[:, :, 1] = ch_data_u8
    elif channel_idx == 2: # Blue
        buf[:, :, 0] = ch_data_u8
    elif channel_idx == 3: # Alpha (Grayscale)
        buf[:, :, 0] = ch_data_u8
        buf[:, :, 1] = ch_data_u8
        buf[:, :, 2] = ch_data_u8
        
    buf[:, :, 3] = 255 # Fully opaque to make channel visible
    
    # Must copy the QImage to prevent garbage collection of buffer
    img = QImage(buf.data, w, h, w * 4, QImage.Format.Format_ARGB32).copy()

    scaled = img.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.FastTransformation)
    tp = QPainter(pm)
    ox = (size - scaled.width()) // 2
    oy = (size - scaled.height()) // 2
    tp.drawImage(ox, oy, scaled)
    tp.end()
    return pm


class ChannelItemWidget(QFrame):
    visibility_clicked = Signal(str, bool)

    def __init__(self, channel_id: str, channel_idx: int, name: str, parent=None):
        super().__init__(parent)
        self.channel_id = channel_id
        self.channel_idx = channel_idx
        self.visible_state = True
        
        self.setObjectName("channelRow")

        palette = ThemeManager.instance().active_palette
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # Thumbnail
        self._thumb_label = QLabel()
        self._thumb_label.setFixedSize(32, 32)
        self._thumb_label.setStyleSheet(f"border: 1px solid {palette['border']}; background: transparent; border-radius: 2px;")
        layout.addWidget(self._thumb_label)

        # Name
        self._name_label = QLabel(name)
        self._name_label.setStyleSheet(f"color: {palette['fg']}; background: transparent; padding: 0 2px;")
        layout.addWidget(self._name_label, 1)

        # Eye icon
        self._vis_btn = QPushButton()
        self._vis_btn.setIcon(icon_eye(True))
        self._vis_btn.setIconSize(QSize(18, 18))
        self._vis_btn.setFixedSize(24, 24)
        self._vis_btn.setFlat(True)
        self._vis_btn.setToolTip("Toggle visibility")
        self._vis_btn.setStyleSheet("background: transparent; border: none;")
        self._vis_btn.clicked.connect(self._toggle_vis)
        layout.addWidget(self._vis_btn)

    def _toggle_vis(self):
        self.visible_state = not self.visible_state
        self._vis_btn.setIcon(icon_eye(self.visible_state))
        self.visibility_clicked.emit(self.channel_id, self.visible_state)

    def update_state(self, visible: bool, pixels: np.ndarray | None):
        if self.visible_state != visible:
            self.visible_state = visible
            self._vis_btn.setIcon(icon_eye(visible))
        
        # update thumbnail preview
        if pixels is not None:
            pm = channel_pixels_to_pixmap(pixels, self.channel_idx, 32)
            self._thumb_label.setPixmap(pm)
        else:
            self._thumb_label.setPixmap(QPixmap(thumb_checker(32)))


class ChannelsPanel(QWidget):
    """Panel for toggling R, G, B, A channels of the active layer."""

    value_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            render_qss(
                "panel_surface.qss",
                selector="QWidget",
                row_selector="#channelRow",
                bg="#2a2a2a",
                fg="#ddd",
                font_size=11,
                label_fg="#ddd",
                row_radius=3,
                row_hover="#353535",
            )
        )

        self._doc: Document | None = None
        self._block_signals = False
        # Content key of what is currently displayed, so a
        # refresh with unchanged state can return immediately.
        self._shown_key = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        self.item_r = ChannelItemWidget("channel_r", 0, "Red")
        self.item_g = ChannelItemWidget("channel_g", 1, "Green")
        self.item_b = ChannelItemWidget("channel_b", 2, "Blue")
        self.item_a = ChannelItemWidget("channel_a", 3, "Alpha")

        self.item_r.visibility_clicked.connect(self._on_toggled)
        self.item_g.visibility_clicked.connect(self._on_toggled)
        self.item_b.visibility_clicked.connect(self._on_toggled)
        self.item_a.visibility_clicked.connect(self._on_toggled)

        layout.addWidget(self.item_r)
        layout.addWidget(self.item_g)
        layout.addWidget(self.item_b)
        layout.addWidget(self.item_a)
        layout.addStretch()

    def refresh(self, doc: Document | None):
        self._doc = doc
        if not doc or not doc.layers.active_layer:
            self.setEnabled(False)
            self._shown_key = None
            return

        self.setEnabled(True)
        # Skip entirely when nothing this panel displays has changed. It used
        # to re-derive four channel previews -- and, for a group, re-composite
        # the whole group -- on every rendered frame.
        layer = doc.layers.active_layer
        key = (layer.id, layer.content_version,
               layer.channel_r, layer.channel_g, layer.channel_b, layer.channel_a,
               len(doc.layers.layers))
        if key == self._shown_key:
            return
        self._shown_key = key
        self._update_ui_values()

    def _update_ui_values(self):
        if self._block_signals or not self._doc:
            return

        layer = self._doc.layers.active_layer
        if not layer:
            return

        self._block_signals = True

        from ...core.enums import LayerType
        
        # We try to use composite_group_tight if group
        pixels = None
        if layer.layer_type == LayerType.GROUP:
            from ...engine.compositor import Compositor
            pixels = Compositor().composite_group_tight(layer, self._doc.layers)
        else:
            pixels = layer.pixels

        self.item_r.update_state(layer.channel_r, pixels)
        self.item_g.update_state(layer.channel_g, pixels)
        self.item_b.update_state(layer.channel_b, pixels)
        self.item_a.update_state(layer.channel_a, pixels)

        self._block_signals = False

    def _on_toggled(self, channel: str, state: bool):
        if self._block_signals or not self._doc:
            return

        layer = self._doc.layers.active_layer
        if not layer:
            return

        setattr(layer, channel, state)
        self.value_changed.emit()
