"""MoveTool — the main move/resize/rotate layer tool.

This module assembles the sub-module mixins into the final ``MoveTool``
class.  All heavy lifting is delegated:

* Hit-testing          → :mod:`.hit_test`
* Layer auto-select    → :mod:`.auto_select`
* Floating selection   → :mod:`.float_selection.FloatSelectionMixin`
* Resize operations    → :mod:`.resize_ops.ResizeMixin`
* Rotate operations    → :mod:`.rotate_ops.RotateMixin`
* Vector baking        → :mod:`.vector_commit.VectorCommitMixin`
* Alignment / flip     → :mod:`.align_ops`
"""

from __future__ import annotations

import numpy as np

from ..tool_base import Tool
from ...core.document import Document
from ...core.enums import LayerType

from ._enums import _Mode, _Handle
from . import hit_test as _ht
from .auto_select import find_layer_at, point_on_layer
from .float_selection import FloatSelectionMixin
from .resize_ops import ResizeMixin
from .rotate_ops import RotateMixin
from .vector_commit import VectorCommitMixin
from . import align_ops as _align


class MoveTool(FloatSelectionMixin, ResizeMixin, RotateMixin, VectorCommitMixin, Tool):
    """Click-drag to move; drag handles to resize; drag outside box to rotate.

    Uses the layer's non-destructive transform system so resize and
    rotate operations always recompute display pixels from the stored
    original source — quality is never lost no matter how many times
    the user transforms a layer (Affinity-style).
    """

    # Expose these as class-level constants so external code can read them
    HANDLE_MARGIN = _ht.HANDLE_MARGIN
    ROTATE_HANDLE_OFFSET = _ht.ROTATE_HANDLE_OFFSET

    def __init__(self) -> None:
        super().__init__("Move")
        self._mode = _Mode.NONE
        self._handle = _Handle.NONE
        self._start_x: int = 0
        self._start_y: int = 0
        self._orig_position: tuple[int, int] = (0, 0)
        self._orig_pixels: np.ndarray | None = None
        self._orig_width: int = 0
        self._orig_height: int = 0
        self._dragging: bool = False
        self._current_angle: float = 0.0
        # Reference to the layer being transformed (for mid-drag writes)
        self._active_layer = None
        # Anchor state for rotated resize
        self._anchor_screen: tuple[float, float] = (0.0, 0.0)
        self._anchor_sign: tuple[int, int] = (0, 0)
        self._is_rotated_resize: bool = False
        # Non-destructive transform: angle at the start of the drag
        self._base_angle: float = 0.0
        # Auto-select: when True, clicking outside the active layer
        # picks the topmost visible layer whose opaque pixels are
        # under the cursor.
        self.auto_select: bool = True
        # Callback set by MainWindow to handle layer selection changes.
        # Signature: (layer_index: int) -> None
        self.on_layer_auto_selected: callable | None = None
        # Group support: track children and their original state
        self._group_children: list = []
        self._group_child_positions: dict[str, tuple[int, int]] = {}
        self._group_child_pixels: dict[str, np.ndarray] = {}
        self._group_orig_bbox: tuple[int, int, int, int] | None = None
        # Non-destructive group transforms (Affinity-style)
        self._group_child_base_sx: dict[str, float] = {}
        self._group_child_base_sy: dict[str, float] = {}
        self._group_child_base_angle: dict[str, float] = {}
        self._group_child_dims: dict[str, tuple[int, int]] = {}
        # Floating selection state (used by FloatSelectionMixin)
        self._floating: bool = False
        self._float_pixels: np.ndarray | None = None
        self._float_base: np.ndarray | None = None
        self._float_orig: np.ndarray | None = None
        self._float_dx: int = 0
        self._float_dy: int = 0
        self._float_committed_dx: int = 0
        self._float_committed_dy: int = 0
        # Vector transform pivot (used by VectorCommitMixin)
        self._orig_center: tuple[float, float] = (0.0, 0.0)
        # Shift-click multi-select: set externally by CanvasController
        self.shift_held: bool = False
        # Callback for deselect-all (no layer active)
        # Signature: () -> None
        self.on_deselect_all: callable | None = None
        # Marquee drag-select state
        self._marquee_start: tuple[int, int] | None = None
        self._marquee_current: tuple[int, int] | None = None
        self._marquee_active: bool = False
        # Callback for marquee layer selection
        # Signature: (indices: list[int]) -> None
        self.on_marquee_select: callable | None = None
        # Deferred auto-select: when press lands on a MOVE zone we hold off
        # switching until we know it's a click (not a drag).  Stores the
        # topmost layer index found by find_layer_at, or None.
        self._pending_autoselect_idx: int | None = None

    # ------------------------------------------------------------------
    # Public query (used by MainWindow for bounding-box overlay)
    # ------------------------------------------------------------------

    def rotation_info_for(self, layer) -> tuple[int, int, float] | None:
        """Return ``(base_w, base_h, total_angle)`` for *layer*.

        The total angle includes any mid-drag rotation that has not yet
        been committed.  Returns ``None`` when the layer has no rotation.
        """
        if layer is None:
            return None
        extra = self._current_angle if (layer is self._active_layer) else 0.0
        total = layer.transform_angle + extra
        if total != 0.0 and layer.transform_base_w > 0:
            return (layer.transform_base_w, layer.transform_base_h, total)
        return None

    # ------------------------------------------------------------------
    # Hit-test wrappers (delegate to hit_test module)
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox(doc: Document) -> tuple[int, int, int, int] | None:
        return _ht.bbox(doc)

    @staticmethod
    def _group_bbox(doc: Document, group) -> tuple[int, int, int, int] | None:
        return _ht.group_bbox(doc, group)

    def _hit_test(self, doc: Document, x: int, y: int) -> tuple[_Mode, _Handle]:
        return _ht.hit_test(doc, x, y, current_angle=self._current_angle)

    @staticmethod
    def _hit_test_rect(
        bx: float, by: float, bw: float, bh: float,
        x: float, y: float,
    ) -> tuple[_Mode, _Handle]:
        return _ht.hit_test_rect(bx, by, bw, bh, x, y)

    # ------------------------------------------------------------------
    # Auto-select wrappers (delegate to auto_select module)
    # ------------------------------------------------------------------

    @staticmethod
    def _point_on_layer(layer, x: int, y: int, alpha_threshold: float = 0.01) -> bool:
        return point_on_layer(layer, x, y, alpha_threshold)

    @staticmethod
    def _find_layer_at(
        doc: Document, x: int, y: int,
        exclude_id: str | None = None,
        alpha_threshold: float = 0.01,
    ) -> int | None:
        return find_layer_at(doc, x, y, exclude_id, alpha_threshold)

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def on_press(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        layer = doc.layers.active_layer

        # --- Reset marquee state ---
        self._marquee_start = None
        self._marquee_current = None
        self._marquee_active = False

        # --- Auto-select logic -----------------------------------------------
        # Strategy:
        #   • RESIZE / ROTATE handles  → never auto-select (suppress immediately).
        #   • Multi-bbox interior       → skip auto-select (move the group).
        #   • MOVE zone on active layer → DEFER auto-select to on_release so
        #     that a drag on an already-selected layer moves it instead of
        #     accidentally pulling a layer from behind.
        #   • Click on empty / transparent area → marquee on drag, deselect on
        #     click.
        auto_switched = False
        mode_hit = _Mode.NONE
        self._pending_autoselect_idx = None  # reset deferred state every press
        if self.auto_select:
            skip_autoselect = False
            if layer is not None:
                mode_hit, _hh = self._hit_test(doc, x, y)
                # Only suppress for direct handle / rotate interactions
                if mode_hit in (_Mode.RESIZE, _Mode.ROTATE):
                    skip_autoselect = True

            # Also skip auto-select when there are multiple layers
            # selected and the click lands inside the multi-bbox
            # (the user wants to move/transform the group, not re-select).
            sel_indices = doc.layers.selected_indices
            if not skip_autoselect and len(sel_indices) > 1:
                mb = _ht.multi_bbox(doc)
                if mb is not None:
                    mbx, mby, mbw, mbh = mb
                    if mbx <= x <= mbx + mbw and mby <= y <= mby + mbh:
                        skip_autoselect = True

            if not skip_autoselect:
                topmost_idx = find_layer_at(doc, x, y)
                if topmost_idx is not None:
                    topmost = doc.layers.layers[topmost_idx]
                    if self.shift_held:
                        # Shift+click: immediate — toggling a multi-select is
                        # always a deliberate single-click action.
                        doc.layers.select_toggle(topmost_idx)
                        if self.on_layer_auto_selected:
                            self.on_layer_auto_selected(
                                doc.layers.active_index if doc.layers.active_index >= 0 else topmost_idx)
                        layer = doc.layers.active_layer
                        auto_switched = True
                    elif layer is not None and topmost.id == layer.id:
                        # Topmost IS the current layer → normal move, no switch.
                        pass
                    elif mode_hit == _Mode.MOVE and layer is not None:
                        # Click is inside the active layer's bbox (opaque OR
                        # transparent pixel) — defer the switch until release.
                        # This prevents dragging from an empty area of the layer
                        # from accidentally selecting the layer behind it.
                        # If the user just clicks (no drag), on_release will
                        # apply the switch to the correct topmost layer.
                        self._pending_autoselect_idx = topmost_idx
                    else:
                        # Clearly clicking on a different layer (e.g. transparent
                        # area of active layer, or no active layer) — switch now.
                        doc.layers.active_index = topmost_idx
                        if self.on_layer_auto_selected:
                            self.on_layer_auto_selected(topmost_idx)
                        layer = doc.layers.active_layer
                        auto_switched = True
                else:
                    # No visible layer at click point.
                    # If click is on/near any bbox zone (interior, border,
                    # handle, or rotation zone), do NOT start a marquee —
                    # fall through to normal setup (e.g. transparent PNG).
                    # Only start a marquee when truly on empty canvas.
                    if mode_hit != _Mode.NONE:
                        pass  # on/near bbox — fall through to normal setup
                    else:
                        # Begin marquee drag-select instead of immediate deselect
                        self._marquee_start = (x, y)
                        self._marquee_current = (x, y)
                        self._dragging = True
                        return

        if layer is None or layer.locked:
            return

        self._mode, self._handle = self._hit_test(doc, x, y)

        # If the hit-test returns NONE (click outside any interaction zone)
        # and auto-select didn't switch, there's nothing to do.
        if self._mode == _Mode.NONE:
            return

        # If the click lands in the ROTATE zone and auto-select switched
        # layers, demote to MOVE (on opaque pixels) or NONE (empty).
        if self.auto_select and self._mode == _Mode.ROTATE:
            if auto_switched:
                if point_on_layer(layer, x, y):
                    self._mode = _Mode.MOVE
                    self._handle = _Handle.NONE
                else:
                    self._mode = _Mode.NONE
                    return

        label = {_Mode.MOVE: "Move", _Mode.RESIZE: "Resize", _Mode.ROTATE: "Rotate"}
        doc.save_snapshot(label.get(self._mode, "Transform"))

        self._start_x, self._start_y = x, y
        self._orig_position = layer.position
        self._dragging = True
        self._active_layer = layer
        self._is_rotated_resize = False
        self._current_angle = 0.0
        self._base_angle = layer.transform_angle

        # -- Floating selection: if a selection is active and mode is MOVE,
        #    cut selected pixels into a floating buffer. -------------------
        if (self._mode == _Mode.MOVE
                and doc.selection.active
                and layer.layer_type not in (LayerType.GROUP, LayerType.ADJUSTMENT, LayerType.FILTER)):

            # Continue an in-progress float without re-cutting
            if self._floating and self._float_pixels is not None:
                self._float_dx = self._float_committed_dx
                self._float_dy = self._float_committed_dy
                return

            sel_mask = self._get_sel_mask(doc)
            if sel_mask is not None and sel_mask.max() > 0:
                mask4 = sel_mask[..., np.newaxis]
                self._float_orig = layer.pixels.copy()
                self._float_pixels = layer.pixels * mask4
                layer.pixels[:] = layer.pixels * (1.0 - mask4)
                layer.touch()
                np.clip(layer.pixels, 0, 1, out=layer.pixels)
                self._float_base = layer.pixels.copy()
                self._floating = True
                self._float_dx = 0
                self._float_dy = 0
                self._float_committed_dx = 0
                self._float_committed_dy = 0
                return  # skip normal setup

        # If there's an existing float but we're not continuing it, commit it
        if self._floating:
            self.commit_float(doc)

        # -- Multi-selection setup (treat like a virtual group) -------------
        self._multi_layers: list = []
        self._multi_positions: dict[str, tuple[int, int]] = {}
        self._multi_orig_bbox: tuple[int, int, int, int] | None = None
        self._multi_base_sx: dict[str, float] = {}
        self._multi_base_sy: dict[str, float] = {}
        self._multi_base_angle: dict[str, float] = {}
        self._multi_dims: dict[str, tuple[int, int]] = {}
        sel_indices = doc.layers.selected_indices
        if len(sel_indices) > 1:
            for si in sorted(sel_indices):
                if 0 <= si < len(doc.layers.layers):
                    sl = doc.layers.layers[si]
                    if not sl.locked:
                        self._multi_layers.append(sl)
                        self._multi_positions[sl.id] = sl.position
                        if self._mode in (_Mode.RESIZE, _Mode.ROTATE):
                            sl.init_non_destructive()
                            self._multi_base_sx[sl.id] = sl.transform_scale_x
                            self._multi_base_sy[sl.id] = sl.transform_scale_y
                            self._multi_base_angle[sl.id] = sl.transform_angle
                            self._multi_dims[sl.id] = (sl.width, sl.height)
            if self._multi_layers:
                mb = _ht.multi_bbox(doc)
                self._multi_orig_bbox = mb
                if mb and self._mode in (_Mode.RESIZE, _Mode.ROTATE):
                    self._orig_width = mb[2]
                    self._orig_height = mb[3]
                    if self._mode == _Mode.ROTATE:
                        self._orig_position = (mb[0], mb[1])
                        # _apply_rotate uses orig_position + size for centre
                    return  # skip single-layer setup below

        # -- Group setup ---------------------------------------------------
        self._group_children = []
        self._group_child_positions = {}
        self._group_child_pixels = {}
        self._group_orig_bbox = None
        self._group_child_base_sx = {}
        self._group_child_base_sy = {}
        self._group_child_base_angle = {}
        self._group_child_dims = {}

        if layer.layer_type == LayerType.GROUP:
            for child in doc.layers:
                if child.parent_id != layer.id:
                    continue
                self._group_children.append(child)
                self._group_child_positions[child.id] = child.position
                if child.layer_type in (LayerType.ADJUSTMENT, LayerType.FILTER):
                    continue
                # Composite group children to get raster pixels for scaling
                if child.layer_type == LayerType.GROUP:
                    from ...engine.compositor import Compositor
                    from ...core.layer_stack import LayerStack
                    compositor = Compositor()
                    px = compositor.composite_group_tight(child, doc.layers)
                    if px is not None and px.size > 0:
                        bounds = LayerStack._content_bounds(child, doc.layers)
                        if bounds is not None:
                            child.position = (int(bounds[0]), int(bounds[1]))
                        child.pixels = px
                child.init_non_destructive()
                self._group_child_base_sx[child.id] = child.transform_scale_x
                self._group_child_base_sy[child.id] = child.transform_scale_y
                self._group_child_base_angle[child.id] = child.transform_angle
                self._group_child_dims[child.id] = (child.width, child.height)
            bbox = _ht.group_bbox(doc, layer)
            self._group_orig_bbox = bbox
            if bbox:
                self._orig_width = bbox[2]
                self._orig_height = bbox[3]
            else:
                self._orig_width = layer.width
                self._orig_height = layer.height
            self._orig_pixels = None
            return  # skip single-layer setup below

        # -- Non-group parent with children (pseudo-group) -----------------
        # Treat the parent + its regular children as a group so move /
        # resize / rotate propagate to every child layer.  Mask children
        # are handled separately via _mask_children + _sync_mask_transforms
        # so they stay pixel-aligned with the parent.
        if layer.children:
            mask_child_ids = set(layer.mask_layers)
            layer.init_non_destructive()
            # Parent itself is the first "group child"
            self._group_children.append(layer)
            self._group_child_positions[layer.id] = layer.position
            self._group_child_base_sx[layer.id] = layer.transform_scale_x
            self._group_child_base_sy[layer.id] = layer.transform_scale_y
            self._group_child_base_angle[layer.id] = layer.transform_angle
            self._group_child_dims[layer.id] = (layer.width, layer.height)
            for child in doc.layers:
                if child.parent_id != layer.id:
                    continue
                # Mask children stay aligned with the parent — skip here
                if child.id in mask_child_ids:
                    continue
                self._group_children.append(child)
                self._group_child_positions[child.id] = child.position
                if child.layer_type in (LayerType.ADJUSTMENT, LayerType.FILTER):
                    continue
                child.init_non_destructive()
                self._group_child_base_sx[child.id] = child.transform_scale_x
                self._group_child_base_sy[child.id] = child.transform_scale_y
                self._group_child_base_angle[child.id] = child.transform_angle
                self._group_child_dims[child.id] = (child.width, child.height)
            # Collect mask children so transforms stay aligned with parent
            self._mask_children: list = []
            self._mask_child_positions: dict[str, tuple[int, int]] = {}
            self._mask_child_base_sx: dict[str, float] = {}
            self._mask_child_base_sy: dict[str, float] = {}
            self._mask_child_base_angle: dict[str, float] = {}
            for mid in layer.mask_layers:
                mc = doc.layers.get(mid)
                if mc is not None:
                    mc.init_non_destructive()
                    self._mask_children.append(mc)
                    self._mask_child_positions[mc.id] = mc.position
                    self._mask_child_base_sx[mc.id] = mc.transform_scale_x
                    self._mask_child_base_sy[mc.id] = mc.transform_scale_y
                    self._mask_child_base_angle[mc.id] = mc.transform_angle
            # Bbox = parent's own bounds (clipped children's overflow is
            # invisible and must NOT inflate the transform reference frame).
            lx, ly = layer.position
            bbox = (lx, ly, layer.width, layer.height)
            self._group_orig_bbox = bbox
            self._orig_width = layer.width if layer.width > 0 else 1
            self._orig_height = layer.height if layer.height > 0 else 1
            self._orig_pixels = None
            return  # skip single-layer setup below

        # -- Single-layer non-destructive setup ----------------------------
        layer.init_non_destructive()

        # Collect mask children so transforms propagate to them
        self._mask_children: list = []
        self._mask_child_positions: dict[str, tuple[int, int]] = {}
        self._mask_child_base_sx: dict[str, float] = {}
        self._mask_child_base_sy: dict[str, float] = {}
        self._mask_child_base_angle: dict[str, float] = {}
        for mid in layer.mask_layers:
            mc = doc.layers.get(mid)
            if mc is not None:
                mc.init_non_destructive()
                self._mask_children.append(mc)
                self._mask_child_positions[mc.id] = mc.position
                self._mask_child_base_sx[mc.id] = mc.transform_scale_x
                self._mask_child_base_sy[mc.id] = mc.transform_scale_y
                self._mask_child_base_angle[mc.id] = mc.transform_angle

        if self._mode == _Mode.ROTATE:
            self._orig_width = layer.width
            self._orig_height = layer.height

        elif self._mode == _Mode.RESIZE:
            if layer.transform_angle != 0.0 and layer.transform_base_w > 0:
                self._is_rotated_resize = True
                self._orig_width = layer.transform_base_w
                self._orig_height = layer.transform_base_h
                self._setup_resize_anchor(layer)
            else:
                self._orig_width = layer.width
                self._orig_height = layer.height

        # Record original bbox centre for vector commit
        self._orig_center = (
            layer.position[0] + layer.width / 2.0,
            layer.position[1] + layer.height / 2.0,
        )

    @property
    def marquee_rect(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Return the current marquee rectangle for overlay drawing, or None."""
        if self._marquee_active and self._marquee_start and self._marquee_current:
            return (self._marquee_start, self._marquee_current)
        return None

    def _finish_marquee_select(self, doc: Document) -> None:
        """Select all layers whose bounding boxes intersect the marquee rect."""
        if self._marquee_start is None or self._marquee_current is None:
            return
        sx, sy = self._marquee_start
        ex, ey = self._marquee_current
        # Normalise to min/max rect
        rx0, rx1 = min(sx, ex), max(sx, ex)
        ry0, ry1 = min(sy, ey), max(sy, ey)
        # Minimum 4px drag to count as a marquee
        if abs(rx1 - rx0) < 4 and abs(ry1 - ry0) < 4:
            doc.layers.select_clear()
            if self.on_deselect_all:
                self.on_deselect_all()
            return
        hit_indices: list[int] = []
        for i, layer in enumerate(doc.layers.layers):
            if not layer.visible or layer.layer_type in (
                    LayerType.GROUP, LayerType.ADJUSTMENT, LayerType.FILTER):
                continue
            lx, ly = layer.position
            lx2, ly2 = lx + layer.width, ly + layer.height
            # AABB overlap test
            if lx < rx1 and lx2 > rx0 and ly < ry1 and ly2 > ry0:
                hit_indices.append(i)
        if hit_indices:
            doc.layers.select_clear()
            for idx in hit_indices:
                doc.layers.select_add(idx)
            # Set active to the topmost (smallest index) selected layer
            doc.layers._active_index = min(hit_indices)
            if self.on_marquee_select:
                self.on_marquee_select(hit_indices)
        else:
            doc.layers.select_clear()
            if self.on_deselect_all:
                self.on_deselect_all()

    def on_move(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        if not self._dragging:
            return

        # Marquee drag-select update
        if self._marquee_start is not None:
            self._marquee_active = True
            self._marquee_current = (x, y)
            return

        # A real drag has started — cancel any pending deferred auto-select
        # so we never switch layers mid-drag.
        self._pending_autoselect_idx = None

        layer = doc.layers.active_layer
        if layer is None:
            return

        dx = x - self._start_x
        dy = y - self._start_y

        # -- Floating selection move --
        if self._floating and self._float_pixels is not None and self._float_base is not None:
            self._float_dx = self._float_committed_dx + dx
            self._float_dy = self._float_committed_dy + dy
            layer.pixels[:] = self._float_base
            layer.touch()
            self._composite_float(layer.pixels, self._float_dx, self._float_dy)
            return

        is_group = layer.layer_type == LayerType.GROUP
        is_multi = bool(getattr(self, "_multi_layers", []))

        if self._mode == _Mode.MOVE:
            if self._group_children:
                # Pseudo-group or real group: move every member via the
                # stored original positions (the active layer is already
                # included in _group_children for pseudo-groups).
                for child in self._group_children:
                    cox, coy = self._group_child_positions[child.id]
                    child.position = (cox + dx, coy + dy)
            else:
                ox, oy = self._orig_position
                layer.position = (ox + dx, oy + dy)
            # Move mask children with the parent
            for mc in getattr(self, "_mask_children", []):
                mcox, mcoy = self._mask_child_positions[mc.id]
                mc.position = (mcox + dx, mcoy + dy)
            # Move all multi-selected layers together
            for sl in getattr(self, "_multi_layers", []):
                if sl.id != layer.id:
                    sox, soy = self._multi_positions.get(sl.id, sl.position)
                    sl.position = (sox + dx, soy + dy)

        elif self._mode == _Mode.RESIZE:
            if is_multi:
                self._apply_multi_resize(dx, dy)
            elif is_group or self._group_children:
                self._apply_group_resize(dx, dy)
                self._sync_mask_transforms(layer)
            else:
                self._apply_resize(layer, dx, dy)
                self._sync_mask_transforms(layer)

        elif self._mode == _Mode.ROTATE:
            if is_multi:
                self._apply_multi_rotate(x, y)
            elif is_group or self._group_children:
                self._apply_group_rotate(x, y)
                self._sync_mask_transforms(layer)
            else:
                self._apply_rotate(layer, x, y)
                self._sync_mask_transforms(layer)

    def on_release(self, doc: Document, x: int, y: int) -> None:
        # Marquee drag-select: finish and select layers inside the box
        if self._marquee_start is not None:
            if self._marquee_active:
                self._finish_marquee_select(doc)
            else:
                # Click without drag on empty space → deselect all
                doc.layers.select_clear()
                if self.on_deselect_all:
                    self.on_deselect_all()
            self._marquee_start = None
            self._marquee_current = None
            self._marquee_active = False
            self._dragging = False
            return

        # Deferred auto-select: if a pending switch was recorded on press and
        # _pending_autoselect_idx is still set (i.e. no drag cleared it),
        # apply the layer switch now — this was a pure click.
        if self._pending_autoselect_idx is not None:
            idx = self._pending_autoselect_idx
            self._pending_autoselect_idx = None
            doc.layers.active_index = idx
            if self.on_layer_auto_selected:
                self.on_layer_auto_selected(idx)

        # Floating selection: keep the float alive, just commit the offset
        if self._floating and self._active_layer is not None:
            self._float_committed_dx = self._float_dx
            self._float_committed_dy = self._float_dy
            self._dragging = False
            self._mode = _Mode.NONE
            self._handle = _Handle.NONE
            return

        if self._mode in (_Mode.ROTATE, _Mode.RESIZE) and self._active_layer is not None:
            if self._active_layer.layer_type == LayerType.GROUP or self._group_children:
                for child in self._group_children:
                    if child.layer_type == LayerType.RASTER:
                        child.compute_display(fast=False)
                # Finalise mask children aligned to the parent
                for mc in getattr(self, "_mask_children", []):
                    mc.compute_display(fast=False)
            elif self._active_layer.layer_type == LayerType.RASTER:
                self._active_layer.compute_display(fast=False)
                for mc in getattr(self, "_mask_children", []):
                    mc.compute_display(fast=False)
            # Multi-layer final quality pass
            for sl in getattr(self, "_multi_layers", []):
                if sl.layer_type == LayerType.RASTER and sl._source_pixels is not None:
                    sl.compute_display(fast=False)

        # --- Vector layer commit ---
        if self._active_layer is not None and self._mode != _Mode.NONE:
            if getattr(self, "_multi_layers", []):
                self._commit_multi_vector_transforms(doc)
            elif self._active_layer.layer_type == LayerType.GROUP:
                self._commit_group_vector_transforms(doc)
                doc.layers.update_group_bbox(self._active_layer)
            elif self._group_children:
                # Pseudo-group: commit vector transforms for SHAPE children
                self._commit_group_vector_transforms(doc)
            elif self._active_layer.layer_type == LayerType.SHAPE:
                vl = getattr(self._active_layer, "_vector_data", None)
                if vl:
                    self._commit_vector_transform(doc, self._active_layer, vl)

        self._dragging = False
        self._orig_pixels = None
        self._mode = _Mode.NONE
        self._handle = _Handle.NONE
        self._current_angle = 0.0
        self._base_angle = 0.0
        self._active_layer = None
        self._is_rotated_resize = False
        self._mask_children = []
        self._mask_child_positions = {}
        self._mask_child_base_sx = {}
        self._mask_child_base_sy = {}
        self._mask_child_base_angle = {}
        self._group_children = []
        self._group_child_positions = {}
        self._group_child_pixels = {}
        self._group_orig_bbox = None
        self._group_child_base_sx = {}
        self._group_child_base_sy = {}
        self._group_child_base_angle = {}
        self._group_child_dims = {}
        self._multi_layers = []
        self._multi_positions = {}
        self._multi_orig_bbox = None
        self._multi_base_sx = {}
        self._multi_base_sy = {}
        self._multi_base_angle = {}
        self._multi_dims = {}

    # ------------------------------------------------------------------
    # Alignment helpers — delegate to align_ops module
    # ------------------------------------------------------------------

    @staticmethod
    def align_left(doc: Document) -> None:
        _align.align_left(doc)

    @staticmethod
    def align_center_h(doc: Document) -> None:
        _align.align_center_h(doc)

    @staticmethod
    def align_right(doc: Document) -> None:
        _align.align_right(doc)

    @staticmethod
    def align_top(doc: Document) -> None:
        _align.align_top(doc)

    @staticmethod
    def align_middle_v(doc: Document) -> None:
        _align.align_middle_v(doc)

    @staticmethod
    def align_bottom(doc: Document) -> None:
        _align.align_bottom(doc)

    # ------------------------------------------------------------------
    # Flip / Rotate helpers — delegate to align_ops module
    # ------------------------------------------------------------------

    @staticmethod
    def flip_horizontal(doc: Document) -> None:
        _align.flip_horizontal(doc)

    @staticmethod
    def flip_vertical(doc: Document) -> None:
        _align.flip_vertical(doc)

    @staticmethod
    def rotate_90_cw(doc: Document) -> None:
        _align.rotate_90_cw(doc)

    @staticmethod
    def rotate_90_ccw(doc: Document) -> None:
        _align.rotate_90_ccw(doc)
