"""Image transforms — flip, rotate, alignment, Move tool transform box."""

from __future__ import annotations

from ...core.enums import LayerType, ToolType
from ...transforms.transform_engine import TransformEngine
from .base import ControllerBase


class TransformController(ControllerBase):
    """Handles flip, rotate, and alignment/distribution from Move tool bar."""

    def __init__(self) -> None:
        super().__init__()

    def wire(self, main_window) -> None:
        """Connect to main window and wire menu/panel signals."""
        super().wire(main_window)
        mw = self.mw

        a = mw._menu.actions_map
        a["flip_h"].triggered.connect(lambda: self.on_transform("flip_h"))
        a["flip_v"].triggered.connect(lambda: self.on_transform("flip_v"))
        a["rotate_cw"].triggered.connect(lambda: self.on_transform("rotate_cw"))
        a["rotate_ccw"].triggered.connect(lambda: self.on_transform("rotate_ccw"))

        mw._props_panel.align_requested.connect(self.on_align_requested)

    def on_transform(self, op: str) -> None:
        if not self.doc or not self.doc.layers.active_layer:
            return
        layer = self.doc.layers.active_layer
        if layer.locked:
            return
        self.doc.save_snapshot(op.replace("_", " ").title())
        px = layer.pixels
        if op == "flip_h":
            layer.pixels = TransformEngine.flip_h(px)
        elif op == "flip_v":
            layer.pixels = TransformEngine.flip_v(px)
        elif op == "rotate_cw":
            layer.pixels = TransformEngine.rotate(px, -90)
        elif op == "rotate_ccw":
            layer.pixels = TransformEngine.rotate(px, 90)
        self.ctx.refresh()

    def on_align_requested(self, action: str) -> None:
        """Handle alignment/distribution from the Move properties bar."""
        mw = self.mw
        if self.doc is None:
            return
        from ...tools.move_tool import MoveTool
        method = getattr(MoveTool, action, None)
        if method is not None:
            method(self.doc)
            self.ctx.refresh()

    def update_transform_box(self) -> None:
        """Show transform bounding box when the Move tool is active."""
        mw = self.mw
        if mw._tools.active_type != ToolType.MOVE or not self.doc:
            mw._canvas.set_transform_box(None)
            return
        layer = self.doc.layers.active_layer
        if not layer:
            mw._canvas.set_transform_box(None)
            return

        # Multi-selection: display a union bounding box (with rotation)
        sel = self.doc.layers.selected_indices
        if len(sel) > 1:
            from ...tools.move.hit_test import multi_bbox
            tool = mw._tools.active_tool
            angle = 0.0
            if tool is not None:
                angle = getattr(tool, '_current_angle', 0.0)

            # During an active rotation or resize drag, use the stored
            # original bbox so the visual box stays a fixed size and
            # rotates/scales with the drag instead of jittering.
            orig = getattr(tool, '_multi_orig_bbox', None) if tool else None
            if orig is not None and angle != 0.0:
                mw._canvas.set_transform_box(orig, angle)
            else:
                mb = multi_bbox(self.doc)
                if mb:
                    mw._canvas.set_transform_box(mb, angle)
                else:
                    mw._canvas.set_transform_box(None)
            return

        if self.doc.selection.active and self.doc.selection._mask is not None:
            # Selection.bounds is cached per selection version. This used to
            # run two full-document reductions plus two np.where scans on
            # every rendered frame -- four passes over 8.3 M elements at 4K,
            # thirty times a second, for an answer that rarely changes.
            bounds = self.doc.selection.bounds
            if bounds is not None:
                x0, y0, x1, y1 = bounds
                tool = mw._tools.active_tool
                fdx = fdy = 0
                if tool is not None and getattr(tool, '_floating', False):
                    fdx = getattr(tool, '_float_dx', 0)
                    fdy = getattr(tool, '_float_dy', 0)
                mw._canvas.set_transform_box(
                    (x0 + fdx, y0 + fdy, x1 - x0 + 1, y1 - y0 + 1))
            else:
                mw._canvas.set_transform_box(None)
            return

        if layer.layer_type == LayerType.GROUP:
            box = self._group_bbox(layer)
            mw._canvas.set_transform_box(box if box else None)
            return

        # Non-group parent with children (pseudo-group) — show the
        # parent's own bounds (clipped children don't inflate the BB).
        # Support rotation: use the same logic as single-layer rotation.
        if layer.children:
            tool = mw._tools.active_tool
            info = None
            if hasattr(tool, "rotation_info_for"):
                info = tool.rotation_info_for(layer)
            if info is not None:
                bw, bh, angle = info
                lx, ly = layer.position
                cx = lx + layer.width / 2
                cy = ly + layer.height / 2
                box = (int(cx - bw / 2), int(cy - bh / 2), bw, bh)
                mw._canvas.set_transform_box(box, angle)
                return
            if layer.transform_angle != 0.0 and layer.transform_base_w > 0:
                bw = layer.transform_base_w
                bh = layer.transform_base_h
                lx, ly = layer.position
                cx = lx + layer.width / 2
                cy = ly + layer.height / 2
                box = (int(cx - bw / 2), int(cy - bh / 2), bw, bh)
                mw._canvas.set_transform_box(box, layer.transform_angle)
                return
            lx, ly = layer.position
            mw._canvas.set_transform_box((lx, ly, layer.width, layer.height))
            return

        tool = mw._tools.active_tool
        info = None
        if hasattr(tool, "rotation_info_for"):
            info = tool.rotation_info_for(layer)
        if info is not None:
            bw, bh, angle = info
            lx, ly = layer.position
            cx = lx + layer.width / 2
            cy = ly + layer.height / 2
            box = (int(cx - bw / 2), int(cy - bh / 2), bw, bh)
            mw._canvas.set_transform_box(box, angle)
            return
        if layer.transform_angle != 0.0 and layer.transform_base_w > 0:
            bw = layer.transform_base_w
            bh = layer.transform_base_h
            lx, ly = layer.position
            cx = lx + layer.width / 2
            cy = ly + layer.height / 2
            box = (int(cx - bw / 2), int(cy - bh / 2), bw, bh)
            mw._canvas.set_transform_box(box, layer.transform_angle)
            return
        lx, ly = layer.position
        mw._canvas.set_transform_box((lx, ly, layer.width, layer.height))

    def _group_bbox(self, group) -> tuple[int, int, int, int] | None:
        """Compute bounding box for a group from its children."""
        mw = self._mw
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")
        found = False
        for child in self.doc.layers:
            if child.parent_id != group.id:
                continue
            cx, cy = child.position
            min_x = min(min_x, cx)
            min_y = min(min_y, cy)
            max_x = max(max_x, cx + child.width)
            max_y = max(max_y, cy + child.height)
            found = True
        if not found:
            return None
        return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
