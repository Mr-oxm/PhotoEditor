"""Text tool — full in-canvas rich text editing with drag-to-create text boxes.

Modes
-----
IDLE       – waiting for user interaction
DRAWING    – dragging to define a new text box
EDITING    – cursor is active inside a text layer; typing modifies text
BOX_RESIZE – dragging a handle on the text bounding box (resizes box only,
             never changes character sizes)

The bounding box shown in text-tool mode controls the *text area* size
(line-wrapping boundary).  Resizing the box does NOT scale glyphs —
it only reflows text.  Full affine transforms (scale, rotate) are
done via the Move tool, but the text tool preserves any rotation
applied by the Move tool and draws the text box accordingly.
"""

from __future__ import annotations

import math
from enum import Enum, auto

import numpy as np

from .tool_base import Tool
from ..core.document import Document
from ..core.enums import LayerType
from ..core.layer import Layer
from ..core.text_layer import (
    TextLayerData, TextRun, CharFormat, ParagraphFormat, SolidFill,
)
from ..core.color import Color


class _Mode(Enum):
    IDLE = auto()
    DRAWING = auto()   # drag to create text box
    EDITING = auto()   # live editing text
    BOX_RESIZE = auto()


class _BoxHandle(Enum):
    NONE = auto()
    TL = auto(); T = auto(); TR = auto()
    L = auto(); R = auto()
    BL = auto(); B = auto(); BR = auto()


_HANDLE_MARGIN = 8  # hit-test radius in document pixels


class TextTool(Tool):
    """Rich text tool with in-canvas editing and character-level formatting."""

    def __init__(self) -> None:
        super().__init__("Text")
        # --- Default formatting for new text ---
        self.font_family: str = "arial"
        self.font_size: int = 36
        self.color: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.bold: bool = False
        self.italic: bool = False
        self.underline: bool = False
        self.strikethrough: bool = False
        self.alignment: str = "left"
        self.line_height: float = 1.2
        self.letter_spacing: float = 0.0
        self.paragraph_spacing: float = 0.0

        # Preview state for hover functionality
        self._preview_active = False
        self._preview_original_font: str | None = None

        # --- Internal state ---
        self._mode: _Mode = _Mode.IDLE
        self._editing_layer: Layer | None = None
        self._text_data: TextLayerData | None = None

        # Drawing state
        self._draw_start: tuple[int, int] = (0, 0)
        self._draw_end: tuple[int, int] = (0, 0)

        # Box resize state
        self._resize_handle: _BoxHandle = _BoxHandle.NONE
        self._resize_start: tuple[int, int] = (0, 0)
        self._resize_orig_box: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._resize_orig_pos: tuple[int, int] = (0, 0)

        # Callback for requesting UI refresh (set by MainWindow)
        self._refresh_callback = None
        # Lightweight callback for overlay-only updates (no pipeline render)
        self._overlay_callback = None
        # Reference to document for history snapshots
        self._doc_ref: Document | None = None
        # Mouse-drag selection state
        self._drag_selecting: bool = False
        self._drag_select_anchor: int = 0

    # ------------------------------------------------------------------
    # Public API for MainWindow / CanvasView
    # ------------------------------------------------------------------

    @property
    def is_editing(self) -> bool:
        return self._mode in (_Mode.EDITING, _Mode.BOX_RESIZE)

    @property
    def is_drawing(self) -> bool:
        return self._mode == _Mode.DRAWING

    @property
    def editing_layer(self) -> Layer | None:
        return self._editing_layer

    @property
    def text_data(self) -> TextLayerData | None:
        return self._text_data

    @property
    def draw_rect(self) -> tuple[int, int, int, int] | None:
        """Return (x, y, w, h) of the box being drawn, or None."""
        if self._mode != _Mode.DRAWING:
            return None
        x0, y0 = self._draw_start
        x1, y1 = self._draw_end
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        return (x, y, max(w, 1), max(h, 1))

    def editing_box(self) -> tuple[int, int, int, int] | None:
        """Return the text bounding box in doc coords for the editing layer."""
        if self._editing_layer is None or self._text_data is None:
            return None
        lx, ly = self._editing_layer.position
        return (lx, ly, self._text_data.box_width, self._text_data.box_height)

    def editing_rotation(self) -> float:
        """Return the rotation of the editing layer (from move tool transforms)."""
        if self._editing_layer is None:
            return 0.0
        return self._editing_layer.transform_angle

    def default_char_format(self) -> CharFormat:
        """Build a CharFormat from the current tool settings."""
        return CharFormat(
            font_family=self.font_family,
            font_size=float(self.font_size),
            bold=self.bold,
            italic=self.italic,
            underline=self.underline,
            strikethrough=self.strikethrough,
            color=SolidFill(color=Color.from_array(self.color)),
            letter_spacing=self.letter_spacing,
        )

    # ------------------------------------------------------------------
    # Tool lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        super().activate()
        # Don't reset editing state — user may switch away and back

    def deactivate(self) -> None:
        self.commit_editing()
        super().deactivate()

    # ------------------------------------------------------------------
    # Canvas event handlers
    # ------------------------------------------------------------------

    def on_press(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        self._doc_ref = doc
        self._drag_selecting = False

        if self._mode == _Mode.EDITING:
            # Check if click is on the current editing layer's bounding box handle
            handle = self._hit_test_handle(x, y)
            if handle != _BoxHandle.NONE:
                self._start_box_resize(handle, x, y)
                return
            # Check if click is inside the editing text box
            if self._point_in_text_box(x, y):
                self._click_in_text(doc, x, y)
                # Start drag-select
                if self._text_data is not None:
                    self._drag_selecting = True
                    self._drag_select_anchor = self._text_data.cursor_pos
                    self._text_data.selection_start = self._drag_select_anchor
                return
            # Click outside — commit and check if we hit another text layer
            self.commit_editing(doc)

        if self._mode == _Mode.IDLE:
            # Check if clicking on an existing text layer
            layer = self._find_text_layer_at(doc, x, y)
            if layer is not None:
                self._start_editing(layer, doc)
                self._click_in_text(doc, x, y)
                # Start drag-select
                if self._text_data is not None:
                    self._drag_selecting = True
                    self._drag_select_anchor = self._text_data.cursor_pos
                    self._text_data.selection_start = self._drag_select_anchor
                return
            # Start drawing a new text box
            self._mode = _Mode.DRAWING
            self._draw_start = (x, y)
            self._draw_end = (x, y)

    def on_move(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        if self._mode == _Mode.DRAWING:
            self._draw_end = (x, y)
            # Overlay update handled by main_window's _on_canvas_move
        elif self._mode == _Mode.BOX_RESIZE:
            self._do_box_resize(x, y)
            self._re_render_text()
            self._request_refresh()
        elif self._mode == _Mode.EDITING and self._drag_selecting:
            # Drag-to-select in text
            self._click_in_text(doc, x, y)
            if self._text_data is not None:
                self._text_data.selection_start = self._drag_select_anchor
            self._request_overlay()

    def on_release(self, doc: Document, x: int, y: int) -> None:
        self._doc_ref = doc
        if self._mode == _Mode.DRAWING:
            self._draw_end = (x, y)
            rect = self.draw_rect
            if rect is None or rect[2] < 10 or rect[3] < 10:
                # Too small — create a default-sized text box at click point
                bw, bh = 300, 80
                rx = self._draw_start[0]
                ry = self._draw_start[1]
                rect = (rx, ry, bw, bh)
            self._create_text_layer(doc, *rect)
            self._mode = _Mode.EDITING
            self._request_refresh()
        elif self._mode == _Mode.BOX_RESIZE:
            self._mode = _Mode.EDITING
            # Save snapshot after box resize
            doc.save_snapshot("Resize Text Box")
            self._request_refresh()
        elif self._drag_selecting:
            self._drag_selecting = False

    # ------------------------------------------------------------------
    # Key input (called from MainWindow)
    # ------------------------------------------------------------------

    def on_key_press(self, key: int, text: str, modifiers) -> bool:
        """Handle a key press while editing.  Return True if consumed."""
        if self._mode != _Mode.EDITING or self._text_data is None:
            return False

        from PySide6.QtCore import Qt
        td = self._text_data
        ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

        if key == Qt.Key.Key_Escape:
            self.commit_editing()
            return True

        if ctrl and key == Qt.Key.Key_A:
            td.select_all()
            self._request_refresh()
            return True

        if ctrl and key == Qt.Key.Key_B:
            self._toggle_format("bold")
            return True

        if ctrl and key == Qt.Key.Key_I:
            self._toggle_format("italic")
            return True

        if ctrl and key == Qt.Key.Key_U:
            self._toggle_format("underline")
            return True

        if key == Qt.Key.Key_Backspace:
            self._flush_text_snapshot("Delete Text")
            if td.has_selection:
                lo, hi = td.selection_range
                td.delete_range(lo, hi)
                td.cursor_pos = lo
                td.selection_start = None
            elif td.cursor_pos > 0:
                td.delete_range(td.cursor_pos - 1, td.cursor_pos)
                td.cursor_pos -= 1
            self._last_edit_action = "delete"
            self._pending_text_changes = True
            self._pending_action_label = "Delete Text"
            self._re_render_text()
            self._request_refresh()
            return True

        if key == Qt.Key.Key_Delete:
            self._flush_text_snapshot("Delete Text")
            if td.has_selection:
                lo, hi = td.selection_range
                td.delete_range(lo, hi)
                td.cursor_pos = lo
                td.selection_start = None
            elif td.cursor_pos < td.char_count:
                td.delete_range(td.cursor_pos, td.cursor_pos + 1)
            self._last_edit_action = "delete"
            self._pending_text_changes = True
            self._pending_action_label = "Delete Text"
            self._re_render_text()
            self._request_refresh()
            return True

        if key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            self._insert_at_cursor("\n")
            return True

        if key == Qt.Key.Key_Left:
            if shift:
                if td.selection_start is None:
                    td.selection_start = td.cursor_pos
                td.cursor_pos = max(0, td.cursor_pos - 1)
            else:
                if td.has_selection:
                    td.cursor_pos = td.selection_range[0]
                    td.selection_start = None
                else:
                    td.cursor_pos = max(0, td.cursor_pos - 1)
            self._request_refresh()
            return True

        if key == Qt.Key.Key_Right:
            if shift:
                if td.selection_start is None:
                    td.selection_start = td.cursor_pos
                td.cursor_pos = min(td.char_count, td.cursor_pos + 1)
            else:
                if td.has_selection:
                    td.cursor_pos = td.selection_range[1]
                    td.selection_start = None
                else:
                    td.cursor_pos = min(td.char_count, td.cursor_pos + 1)
            self._request_refresh()
            return True

        if key == Qt.Key.Key_Home:
            if shift:
                if td.selection_start is None:
                    td.selection_start = td.cursor_pos
            else:
                td.selection_start = None
            td.cursor_pos = 0
            self._request_refresh()
            return True

        if key == Qt.Key.Key_End:
            if shift:
                if td.selection_start is None:
                    td.selection_start = td.cursor_pos
            else:
                td.selection_start = None
            td.cursor_pos = td.char_count
            self._request_refresh()
            return True

        # Normal printable text
        if text and text.isprintable():
            self._insert_at_cursor(text)
            return True

        return False

    # ------------------------------------------------------------------
    # Editing lifecycle
    # ------------------------------------------------------------------

    def _start_editing(self, layer: Layer, doc: Document | None = None) -> None:
        """Enter edit mode for an existing text layer."""
        self._editing_layer = layer
        self._text_data = getattr(layer, "_text_data", None)
        if self._text_data is None:
            # Not a proper text layer — shouldn't happen
            return
        # Absorb any scale / rotation that was applied via the Move tool
        # so that the text bounding box and font sizes reflect the visual state.
        self._absorb_layer_transform(layer)
        self._mode = _Mode.EDITING
        self._text_data.cursor_pos = self._text_data.char_count
        self._text_data.selection_start = None
        if doc is not None:
            doc.layers.active_index = doc.layers._layers.index(layer)
            self._doc_ref = doc
        # Save a snapshot that captures the state *before* any edits so that
        # the first Ctrl-Z returns here.
        d = doc or self._doc_ref
        if d is not None:
            d.save_snapshot("Edit Text Value")
        self._last_edit_action: str = ""
        self._pending_text_changes: bool = False

    def commit_editing(self, doc: Document | None = None) -> None:
        """Commit current edits and exit edit mode."""
        if self._editing_layer is not None and self._text_data is not None:
            self._re_render_text()
            self._text_data.selection_start = None
            # Flush any remaining unsaved typing changes
            d = doc or self._doc_ref
            if d is not None and getattr(self, "_pending_text_changes", False):
                label = getattr(self, "_pending_action_label", None) or "Edit Text"
                d.save_snapshot(label)
                self._pending_text_changes = False
                self._pending_action_label = None
        self._mode = _Mode.IDLE
        self._editing_layer = None
        self._text_data = None
        self._drag_selecting = False
        self._request_refresh()

    # ------------------------------------------------------------------
    # Transform absorption (Move-tool scale / rotation → text properties)
    # ------------------------------------------------------------------

    def _absorb_layer_transform(self, layer: Layer) -> None:
        """Fold any move-tool transform into the text data.

        When the user resizes or rotates a text layer with the Move tool
        and then re-enters text editing, the bounding box and font sizes
        are updated to match the visual result so that the text does not
        snap back to its original size.
        """
        td = self._text_data
        if td is None:
            return

        has_rotation = (
            layer.transform_angle != 0.0 and layer.transform_base_w > 0
        )

        # Determine the "logical" layer size (pre-rotation)
        if has_rotation:
            actual_w = layer.transform_base_w
            actual_h = layer.transform_base_h
        else:
            actual_w = layer.width
            actual_h = layer.height

        # What the text renderer currently produces
        rendered = td.render()
        rh, rw = rendered.shape[:2]
        expected_w = max(rw, td.box_width)
        expected_h = max(rh, td.box_height)

        sx = actual_w / max(expected_w, 1)
        sy = actual_h / max(expected_h, 1)

        scaled = abs(sx - 1.0) > 0.02 or abs(sy - 1.0) > 0.02
        if not scaled and not has_rotation:
            return  # nothing to absorb

        # Remember the visual centre so we can reposition after re-render
        old_cx = layer.position[0] + layer.width / 2.0
        old_cy = layer.position[1] + layer.height / 2.0

        # --- Apply scale to text metrics ---------------------------------
        if scaled:
            td.box_width = max(40, int(td.box_width * sx))
            td.box_height = max(20, int(td.box_height * sy))
            font_scale = (sx + sy) / 2.0
            for run in td.runs:
                run.fmt.font_size = round(run.fmt.font_size * font_scale, 1)
                if run.fmt.letter_spacing != 0:
                    run.fmt.letter_spacing = round(
                        run.fmt.letter_spacing * font_scale, 1
                    )
            td.invalidate()

        # --- Clear transform bookkeeping on the layer --------------------
        layer.rasterize_transform()

        # Re-render with the updated metrics
        self._render_text_to_layer(layer, td)

        # Keep the visual centre in the same place (important after rotation)
        if has_rotation:
            new_cx = layer.position[0] + layer.width / 2.0
            new_cy = layer.position[1] + layer.height / 2.0
            lx, ly = layer.position
            layer.position = (
                int(lx + old_cx - new_cx),
                int(ly + old_cy - new_cy),
            )

    # ------------------------------------------------------------------
    # Text layer creation
    # ------------------------------------------------------------------

    def _create_text_layer(self, doc: Document, x: int, y: int,
                           w: int, h: int) -> None:
        """Create a new text layer at (x, y) with box size (w, h)."""
        doc.save_snapshot("New Text Layer")
        layer = Layer(
            name="Text Layer",
            width=w, height=h,
            layer_type=LayerType.TEXT,
        )
        layer.position = (x, y)
        # Attach text data
        td = TextLayerData(box_width=w, box_height=h)
        td.runs = [TextRun(text="", fmt=self.default_char_format())]
        td.paragraph_fmt = ParagraphFormat(
            alignment=self.alignment,
            line_height=self.line_height,
            paragraph_spacing=self.paragraph_spacing,
        )
        layer._text_data = td
        # Initial render (blank)
        self._render_text_to_layer(layer, td)
        doc.layers.add(layer)
        self._editing_layer = layer
        self._text_data = td

    def _render_text_to_layer(self, layer: Layer, td: TextLayerData) -> None:
        """Re-render text data and write into layer pixels."""
        rendered = td.render()
        rh, rw = rendered.shape[:2]
        # Ensure layer pixel buffer is big enough
        need_h = max(rh, td.box_height)
        need_w = max(rw, td.box_width)
        if layer.width != need_w or layer.height != need_h:
            # Through the setter, so content_version moves -- history shares
            # buffers by version and would otherwise keep the pre-edit text.
            layer.pixels = np.zeros((need_h, need_w, 4), dtype=np.float32)
        else:
            layer.begin_write()
            layer._pixels[:] = 0.0
        # Paste rendered text
        ph, pw = min(rh, need_h), min(rw, need_w)
        layer._pixels[:ph, :pw] = rendered[:ph, :pw]

    def _re_render_text(self) -> None:
        """Re-render the current editing text layer."""
        if self._editing_layer is not None and self._text_data is not None:
            self._render_text_to_layer(self._editing_layer, self._text_data)

    # ------------------------------------------------------------------
    # Text editing helpers
    # ------------------------------------------------------------------

    def _insert_at_cursor(self, text: str) -> None:
        td = self._text_data
        if td is None:
            return
        # If switching from a non-typing action, flush the old action first
        if getattr(self, "_last_edit_action", "") not in ("", "type"):
            self._flush_text_snapshot("Edit Text")
        if td.has_selection:
            # Deleting a selection is its own undoable step
            self._flush_text_snapshot("Delete Selection")
            lo, hi = td.selection_range
            td.delete_range(lo, hi)
            td.cursor_pos = lo
            td.selection_start = None
        td.insert_text(td.cursor_pos, text)
        td.cursor_pos += len(text)
        self._last_edit_action = "type"
        self._pending_text_changes = True
        self._pending_action_label = "Type Text"
        # Save a snapshot at word boundaries (space, newline, punctuation)
        if text in (" ", "\n") or (len(text) == 1 and not text.isalnum()):
            self._flush_text_snapshot("Type Text")
        self._re_render_text()
        self._request_refresh()

    def _flush_text_snapshot(self, action: str | None = None) -> None:
        """Save a history snapshot if there are pending text changes."""
        if getattr(self, "_pending_text_changes", False):
            d = self._doc_ref
            if d is not None:
                label = action or getattr(self, "_pending_action_label", "Edit Text")
                d.save_snapshot(label)
            self._pending_text_changes = False
            self._pending_action_label = None

    def _toggle_format(self, attr: str) -> None:
        """Toggle a boolean format attribute on the selection or at cursor."""
        td = self._text_data
        if td is None:
            return
        label = attr.replace("_", " ").title()  # e.g. "bold" → "Bold"
        self._flush_text_snapshot(label)
        if td.has_selection:
            lo, hi = td.selection_range
            current = getattr(td.format_at(lo), attr, False)
            td.apply_format(lo, hi, **{attr: not current})
        else:
            # Toggle for future typing
            setattr(self, attr, not getattr(self, attr))
        self._last_edit_action = f"format:{attr}"
        self._pending_text_changes = True
        self._pending_action_label = label
        self._re_render_text()
        self._request_refresh()

    def _click_in_text(self, doc: Document, x: int, y: int) -> None:
        """Place cursor at the clicked position within the text box."""
        td = self._text_data
        if td is None or self._editing_layer is None:
            return
        lx, ly = self._editing_layer.position
        # Account for rotation
        angle = self._editing_layer.transform_angle
        if angle != 0.0:
            cx = lx + td.box_width / 2
            cy = ly + td.box_height / 2
            rad = math.radians(angle)
            dx, dy = x - cx, y - cy
            rx = dx * math.cos(rad) + dy * math.sin(rad)
            ry = -dx * math.sin(rad) + dy * math.cos(rad)
            local_x = rx + td.box_width / 2
            local_y = ry + td.box_height / 2
        else:
            local_x = x - lx
            local_y = y - ly
        td.cursor_pos = td.xy_to_cursor(int(local_x), int(local_y))
        td.selection_start = None
        self._request_refresh()

    # ------------------------------------------------------------------
    # Find text layer under cursor
    # ------------------------------------------------------------------

    def _find_text_layer_at(self, doc: Document, x: int, y: int) -> Layer | None:
        """Find a text layer whose bounding box contains (x, y)."""
        for layer in reversed(list(doc.layers)):
            if layer.layer_type != LayerType.TEXT:
                continue
            if not layer.visible:
                continue
            td = getattr(layer, "_text_data", None)
            if td is None:
                continue
            lx, ly = layer.position
            angle = layer.transform_angle
            if angle != 0.0:
                # Inverse-rotate point into box local space
                cx = lx + td.box_width / 2
                cy = ly + td.box_height / 2
                rad = math.radians(angle)
                dx, dy = x - cx, y - cy
                rx = dx * math.cos(rad) + dy * math.sin(rad)
                ry = -dx * math.sin(rad) + dy * math.cos(rad)
                if (abs(rx) <= td.box_width / 2 + _HANDLE_MARGIN
                        and abs(ry) <= td.box_height / 2 + _HANDLE_MARGIN):
                    return layer
            else:
                if (lx - _HANDLE_MARGIN <= x <= lx + td.box_width + _HANDLE_MARGIN
                        and ly - _HANDLE_MARGIN <= y <= ly + td.box_height + _HANDLE_MARGIN):
                    return layer
        return None

    # ------------------------------------------------------------------
    # Bounding box handle hit-testing & resize
    # ------------------------------------------------------------------

    def _point_in_text_box(self, x: int, y: int) -> bool:
        """Check if (x, y) is inside the current editing text box."""
        if self._editing_layer is None or self._text_data is None:
            return False
        lx, ly = self._editing_layer.position
        td = self._text_data
        angle = self._editing_layer.transform_angle
        if angle != 0.0:
            cx = lx + td.box_width / 2
            cy = ly + td.box_height / 2
            rad = math.radians(angle)
            dx, dy = x - cx, y - cy
            rx = dx * math.cos(rad) + dy * math.sin(rad)
            ry = -dx * math.sin(rad) + dy * math.cos(rad)
            return abs(rx) <= td.box_width / 2 and abs(ry) <= td.box_height / 2
        return lx <= x <= lx + td.box_width and ly <= y <= ly + td.box_height

    def _hit_test_handle(self, x: int, y: int) -> _BoxHandle:
        """Check if (x, y) hits a resize handle on the text bounding box."""
        if self._editing_layer is None or self._text_data is None:
            return _BoxHandle.NONE
        lx, ly = self._editing_layer.position
        td = self._text_data
        bw, bh = td.box_width, td.box_height
        angle = self._editing_layer.transform_angle

        # Transform point to local box space
        if angle != 0.0:
            cx = lx + bw / 2
            cy = ly + bh / 2
            rad = math.radians(angle)
            dx, dy = x - cx, y - cy
            px = dx * math.cos(rad) + dy * math.sin(rad) + bw / 2
            py = -dx * math.sin(rad) + dy * math.cos(rad) + bh / 2
        else:
            px, py = x - lx, y - ly

        m = _HANDLE_MARGIN
        mx, my = bw / 2, bh / 2
        handles = [
            (_BoxHandle.TL, 0, 0), (_BoxHandle.T, mx, 0), (_BoxHandle.TR, bw, 0),
            (_BoxHandle.L, 0, my), (_BoxHandle.R, bw, my),
            (_BoxHandle.BL, 0, bh), (_BoxHandle.B, mx, bh), (_BoxHandle.BR, bw, bh),
        ]
        for hid, hx, hy in handles:
            if abs(px - hx) <= m and abs(py - hy) <= m:
                return hid
        return _BoxHandle.NONE

    def _start_box_resize(self, handle: _BoxHandle, x: int, y: int) -> None:
        self._mode = _Mode.BOX_RESIZE
        self._resize_handle = handle
        self._resize_start = (x, y)
        td = self._text_data
        lx, ly = self._editing_layer.position
        self._resize_orig_box = (lx, ly, td.box_width, td.box_height)
        self._resize_orig_pos = (lx, ly)

    def _do_box_resize(self, x: int, y: int) -> None:
        """Resize the text bounding box (not character sizes)."""
        if self._text_data is None or self._editing_layer is None:
            return
        td = self._text_data
        ox, oy, ow, oh = self._resize_orig_box
        dx = x - self._resize_start[0]
        dy = y - self._resize_start[1]

        # Account for rotation
        angle = self._editing_layer.transform_angle
        if angle != 0.0:
            rad = math.radians(angle)
            ldx = dx * math.cos(rad) + dy * math.sin(rad)
            ldy = -dx * math.sin(rad) + dy * math.cos(rad)
        else:
            ldx, ldy = float(dx), float(dy)

        new_w, new_h = float(ow), float(oh)
        new_x, new_y = float(ox), float(oy)
        h = self._resize_handle

        # Width
        if h in (_BoxHandle.TL, _BoxHandle.L, _BoxHandle.BL):
            new_w = max(40, ow - ldx)
            new_x = ox + (ow - new_w)
        elif h in (_BoxHandle.TR, _BoxHandle.R, _BoxHandle.BR):
            new_w = max(40, ow + ldx)

        # Height
        if h in (_BoxHandle.TL, _BoxHandle.T, _BoxHandle.TR):
            new_h = max(20, oh - ldy)
            new_y = oy + (oh - new_h)
        elif h in (_BoxHandle.BL, _BoxHandle.B, _BoxHandle.BR):
            new_h = max(20, oh + ldy)

        td.box_width = int(new_w)
        td.box_height = int(new_h)
        td.invalidate()
        self._editing_layer.position = (int(new_x), int(new_y))

    # ------------------------------------------------------------------
    # Apply formatting from properties panel
    # ------------------------------------------------------------------

    def apply_property(self, key: str, value: object) -> None:
        """Apply a property change from the properties panel."""
        td = self._text_data
        # If in EDITING/BOX_RESIZE, also update tool defaults (but not while previewing)
        if td is not None and self._mode in (_Mode.EDITING, _Mode.BOX_RESIZE) and not self._preview_active:
            if hasattr(self, key):
                setattr(self, key, value)
            # Don't return, continue to apply to text data
        elif td is None:
            # Not editing — just update defaults
            if hasattr(self, key):
                setattr(self, key, value)
            # If not editing and not a preview, we can return
            if not key.startswith("_preview_"):
                return

        # Handle preview signals
        if key == "_preview_font_family":
            self._start_font_preview(value)
            return
        elif key == "_preview_font_end":
            self._end_font_preview()
            return
        elif key == "_preview_font_size":
            self._start_size_preview(value)
            return
        elif key == "_preview_font_size_end":
            self._end_size_preview()
            return

        # Human-readable labels for each property key
        _PROP_LABELS = {
            "font_family": "Font",
            "font_size": "Font Size",
            "bold": "Bold",
            "italic": "Italic",
            "underline": "Underline",
            "strikethrough": "Strikethrough",
            "letter_spacing": "Letter Spacing",
            "alignment": "Alignment",
            "line_height": "Line Height",
            "paragraph_spacing": "Paragraph Spacing",
            "fill_color": "Color",
        }
        label = _PROP_LABELS.get(key, key.replace("_", " ").title())

        # Flush any pending edits before applying a property change
        self._flush_text_snapshot(label)

        # Map property keys to actions
        if key == "font_family":
            self._apply_char_attr("font_family", value)
        elif key == "font_size":
            self._apply_char_attr("font_size", float(value))
            self.font_size = int(value)
        elif key == "bold":
            self._apply_char_attr("bold", bool(value))
            self.bold = bool(value)
        elif key == "italic":
            self._apply_char_attr("italic", bool(value))
            self.italic = bool(value)
        elif key == "underline":
            self._apply_char_attr("underline", bool(value))
            self.underline = bool(value)
        elif key == "strikethrough":
            self._apply_char_attr("strikethrough", bool(value))
            self.strikethrough = bool(value)
        elif key == "letter_spacing":
            self._apply_char_attr("letter_spacing", float(value))
            self.letter_spacing = float(value)
        elif key == "alignment":
            # Set alignment for the current paragraph only
            if td is not None:
                td.set_current_paragraph_alignment(str(value))
            self.alignment = str(value)
        elif key == "line_height":
            td.paragraph_fmt.line_height = float(value)
            td.invalidate()
            self.line_height = float(value)
        elif key == "paragraph_spacing":
            td.paragraph_fmt.paragraph_spacing = float(value)
            td.invalidate()
            self.paragraph_spacing = float(value)
        elif key == "fill_color":
            # value is a ColorFill
            self._apply_char_attr("color", value)
        else:
            if hasattr(self, key):
                setattr(self, key, value)
            return

        self._last_edit_action = f"property:{key}"
        self._pending_text_changes = True
        self._pending_action_label = label
        # Property panel changes are discrete actions — flush immediately
        self._flush_text_snapshot(label)
        self._re_render_text()
        self._request_refresh()

    def _apply_char_attr(self, attr: str, value: object) -> None:
        """Apply a character attribute to the selection or all text."""
        td = self._text_data
        if td is None:
            return
        if td.has_selection:
            lo, hi = td.selection_range
            td.apply_format(lo, hi, **{attr: value})
        else:
            # Apply to all text
            td.apply_format(0, td.char_count, **{attr: value})

    # ------------------------------------------------------------------
    # Hover cursor for bounding box handles
    # ------------------------------------------------------------------

    def hit_test_cursor_shape(self, x: int, y: int) -> str | None:
        """Return a cursor hint for the point (x, y) in doc coords.

        Returns one of "resize_tl", "resize_t", ..., "resize_br",
        "text" (inside box), or None (outside everything).
        """
        if self._mode != _Mode.EDITING or self._editing_layer is None:
            return None
        handle = self._hit_test_handle(x, y)
        if handle != _BoxHandle.NONE:
            return f"resize_{handle.name.lower()}"
        if self._point_in_text_box(x, y):
            return "text"
        return None

    # ------------------------------------------------------------------
    # Refresh callbacks
    # ------------------------------------------------------------------

    def _request_refresh(self) -> None:
        if self._refresh_callback is not None:
            self._refresh_callback()

    def _request_overlay(self) -> None:
        """Request overlay-only update (no pipeline render)."""
        if self._overlay_callback is not None:
            self._overlay_callback()
        elif self._refresh_callback is not None:
            self._refresh_callback()

    def set_refresh_callback(self, cb) -> None:
        self._refresh_callback = cb

    def set_overlay_callback(self, cb) -> None:
        self._overlay_callback = cb

    def _start_font_preview(self, font_family: str) -> None:
        """Start a temporary font preview (during hover)."""
        if self._text_data is None:
            return
        
        # Save original font if not already previewing
        if not self._preview_active:
            self._preview_original_font = self.font_family
            self._preview_active = True
        
        # Apply preview font temporarily
        self._apply_char_attr("font_family", font_family)
        self._re_render_text()
        self._request_refresh()
    
    def _end_font_preview(self) -> None:
        """End font preview and restore original font."""
        if not self._preview_active or self._text_data is None:
            return
        
        # Restore original font
        if self._preview_original_font is not None:
            self._apply_char_attr("font_family", self._preview_original_font)
            self._re_render_text()
            self._request_refresh()
        
        # Clear preview state
        self._preview_active = False
        self._preview_original_font = None

    def _start_size_preview(self, size: int) -> None:
        """Start a temporary font-size preview (during hover)."""
        if self._text_data is None:
            return
        if not self._preview_active:
            # Capture original sizes per-run so we can restore exactly
            self._preview_original_sizes = [
                run.fmt.font_size for run in self._text_data.runs
            ]
            self._preview_active = True
        self._apply_char_attr("font_size", float(size))
        self._re_render_text()
        self._request_refresh()

    def _end_size_preview(self) -> None:
        """End font-size preview and restore original sizes."""
        if not self._preview_active or self._text_data is None:
            return
        orig = getattr(self, "_preview_original_sizes", None)
        if orig is not None:
            # Restore per-run sizes
            for run, orig_size in zip(self._text_data.runs, orig):
                run.fmt.font_size = orig_size
            self._text_data.invalidate()
            self._re_render_text()
            self._request_refresh()
        self._preview_active = False
        self._preview_original_sizes = None
