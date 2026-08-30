"""Ordered layer collection with group support."""

from __future__ import annotations

import numpy as np

from .layer import Layer
from .enums import LayerType


class LayerStack:
    """Manages the ordered list of layers in a document."""

    def __init__(self) -> None:
        self._layers: list[Layer] = []
        self._active_index: int = -1
        self._selected_indices: set[int] = set()

    # ---- Properties ---------------------------------------------------------

    @property
    def layers(self) -> list[Layer]:
        return self._layers

    @property
    def active_layer(self) -> Layer | None:
        if 0 <= self._active_index < len(self._layers):
            return self._layers[self._active_index]
        return None

    @property
    def active_index(self) -> int:
        return self._active_index

    @active_index.setter
    def active_index(self, value: int) -> None:
        if -1 <= value < len(self._layers):
            self._active_index = value
            # Single-select resets multi-selection to just this layer
            if value >= 0:
                self._selected_indices = {value}
            else:
                self._selected_indices = set()

    # ---- Multi-selection ----------------------------------------------------

    @property
    def selected_indices(self) -> set[int]:
        """Return the set of currently selected layer indices."""
        return set(self._selected_indices)

    @property
    def selected_layers(self) -> list[Layer]:
        """Return layers at the selected indices, ordered bottom-to-top."""
        return [self._layers[i] for i in sorted(self._selected_indices)
                if 0 <= i < len(self._layers)]

    def select_add(self, index: int) -> None:
        """Add *index* to the multi-selection and make it active."""
        if 0 <= index < len(self._layers):
            self._selected_indices.add(index)
            self._active_index = index

    def select_toggle(self, index: int) -> None:
        """Toggle *index* in the multi-selection."""
        if 0 <= index < len(self._layers):
            if index in self._selected_indices:
                self._selected_indices.discard(index)
                # If we removed the active, pick another
                if self._active_index == index:
                    if self._selected_indices:
                        self._active_index = max(self._selected_indices)
                    else:
                        self._active_index = -1
            else:
                self._selected_indices.add(index)
                self._active_index = index

    def select_clear(self) -> None:
        """Clear all selection — no layer is active."""
        self._selected_indices.clear()
        self._active_index = -1

    def select_only(self, index: int) -> None:
        """Select only *index* (single-select)."""
        if 0 <= index < len(self._layers):
            self._selected_indices = {index}
            self._active_index = index
        else:
            self.select_clear()

    # ---- Mutations ----------------------------------------------------------

    def add(self, layer: Layer, index: int = -1) -> None:
        if index < 0:
            self._layers.append(layer)
            self._active_index = len(self._layers) - 1
        else:
            self._layers.insert(index, layer)
            self._active_index = index

    def remove(self, layer_id: str) -> Layer | None:
        for i, layer in enumerate(self._layers):
            if layer.id == layer_id:
                removed = self._layers.pop(i)
                self._active_index = min(self._active_index, len(self._layers) - 1)
                return removed
        return None

    def get(self, layer_id: str) -> Layer | None:
        for layer in self._layers:
            if layer.id == layer_id:
                return layer
        return None

    def move(self, from_index: int, to_index: int) -> None:
        if 0 <= from_index < len(self._layers) and 0 <= to_index < len(self._layers):
            layer = self._layers.pop(from_index)
            self._layers.insert(to_index, layer)
            self._active_index = to_index

    def duplicate(self, layer_id: str) -> Layer | None:
        original = self.get(layer_id)
        if original is None:
            return None
        copy = original.duplicate()
        idx = self._layers.index(original)
        self.add(copy, idx + 1)
        return copy

    # ---- Group / reparent ---------------------------------------------------

    @staticmethod
    def _content_bounds(layer: Layer, stack: "LayerStack") -> tuple[float, float, float, float] | None:
        """Return (min_x, min_y, max_x, max_y) for a layer's content, or None."""
        if layer.layer_type == LayerType.GROUP:
            min_x, min_y = float("inf"), float("inf")
            max_x, max_y = float("-inf"), float("-inf")
            found = False
            for child in stack:
                if child.parent_id != layer.id or not child.visible:
                    continue
                if child.layer_type in (LayerType.MASK, LayerType.ADJUSTMENT, LayerType.FILTER):
                    continue
                cb = LayerStack._content_bounds(child, stack)
                if cb is None:
                    continue
                cx0, cy0, cx1, cy1 = cb
                min_x = min(min_x, cx0)
                min_y = min(min_y, cy0)
                max_x = max(max_x, cx1)
                max_y = max(max_y, cy1)
                found = True
            return (min_x, min_y, max_x, max_y) if found else None
        try:
            lx, ly = layer.position
            return (float(lx), float(ly), float(lx + layer.width), float(ly + layer.height))
        except (AttributeError, TypeError):
            return None

    def update_group_bbox(self, group: Layer, *, recurse_to_parents: bool = True) -> bool:
        """Update a group's position, width, height to fit its content.

        Returns True if the group was updated. If recurse_to_parents is True,
        also updates any parent groups.
        """
        if group.layer_type != LayerType.GROUP:
            return False
        bounds = self._content_bounds(group, self)
        if bounds is None:
            # Empty group: keep current or use minimal size
            group.position = (0, 0)
            group.width = max(1, group.width)
            group.height = max(1, group.height)
            updated = True
        else:
            min_x, min_y, max_x, max_y = bounds
            w = max(1, int(max_x - min_x))
            h = max(1, int(max_y - min_y))
            if (group.position != (int(min_x), int(min_y))
                    or group.width != w or group.height != h):
                group.position = (int(min_x), int(min_y))
                group.width = w
                group.height = h
                updated = True
            else:
                updated = False
        if updated and recurse_to_parents and group.parent_id:
            parent = self.get(group.parent_id)
            if parent is not None and parent.layer_type == LayerType.GROUP:
                self.update_group_bbox(parent, recurse_to_parents=True)
        return updated

    def reparent(self, layer_ids: list[str], new_parent_id: str | None) -> None:
        """Move *layer_ids* into the group identified by *new_parent_id*.

        Pass ``None`` to un-parent (move to top level).
        The layers are also repositioned in the stack so that children
        sit just before (below) their new group, or at the top of the
        stack when un-parenting.
        """
        new_parent = self.get(new_parent_id) if new_parent_id else None
        affected_old_parent_ids: set[str] = set()
        for lid in layer_ids:
            layer = self.get(lid)
            if layer is None or lid == new_parent_id:
                continue
            # Remove from old parent's children / mask_layers list
            if layer.parent_id:
                affected_old_parent_ids.add(layer.parent_id)
                old_parent = self.get(layer.parent_id)
                if old_parent:
                    if lid in old_parent.children:
                        old_parent.children.remove(lid)
                    if lid in old_parent.mask_layers:
                        old_parent.mask_layers.remove(lid)
            # When a mask layer is explicitly unparented (made standalone),
            # clear ex_parent_id so it acts as a true standalone mask.
            if layer.layer_type == LayerType.MASK and new_parent_id is None:
                layer.ex_parent_id = None
            # Clear clipping_mask / clips_parent when unparenting — these
            # only apply while the layer is scoped to a parent.
            if new_parent_id is None:
                layer.clipping_mask = False
                layer.clips_parent = False
            # Set new parent
            layer.parent_id = new_parent_id
            if new_parent and lid not in new_parent.children:
                new_parent.children.append(lid)

        # Reposition layers in the stack.
        # First, pull the affected layers out of the list.
        moved: list[Layer] = []
        for lid in layer_ids:
            for i, layer in enumerate(self._layers):
                if layer.id == lid:
                    moved.append(self._layers.pop(i))
                    break

        if new_parent_id:
            # Insert just before the group so compositing order is correct.
            group_idx = 0
            for i, layer in enumerate(self._layers):
                if layer.id == new_parent_id:
                    group_idx = i
                    break
            for j, layer in enumerate(moved):
                self._layers.insert(group_idx + j, layer)
            new_parent = self.get(new_parent_id)
            if new_parent and new_parent.layer_type == LayerType.GROUP:
                self.update_group_bbox(new_parent)
        else:
            # Un-parenting: append at the end (top of the stack).
            self._layers.extend(moved)
            # Update old parent groups that lost children
            for old_pid in affected_old_parent_ids:
                old_parent = self.get(old_pid)
                if old_parent and old_parent.layer_type == LayerType.GROUP:
                    self.update_group_bbox(old_parent)

    def create_group_from(self, layer_ids: list[str], group_name: str = "Group") -> "Layer | None":
        """Create a new group layer containing *layer_ids*.

        Non-selected children (clips_parent, mask_layers) of selected
        layers are automatically brought along so hierarchies are preserved.
        """
        from .enums import LayerType

        # Include non-selected children that belong to selected parents
        all_ids = list(layer_ids)
        selected_set = set(all_ids)
        for lid in list(layer_ids):
            layer = self.get(lid)
            if layer is None:
                continue
            for child_id in list(layer.children) + list(layer.mask_layers):
                if child_id not in selected_set:
                    all_ids.append(child_id)
                    selected_set.add(child_id)

        indices = []
        for lid in all_ids:
            for i, layer in enumerate(self._layers):
                if layer.id == lid:
                    indices.append(i)
                    break
        if not indices:
            return None

        # Insert the group above the topmost selected layer
        top_idx = max(indices)
        first = self._layers[indices[0]]
        group = Layer(
            name=group_name,
            width=first.width,
            height=first.height,
            layer_type=LayerType.GROUP,
        )
        self._layers.insert(top_idx + 1, group)

        # Reparent only top-level selected layers into the new group.
        # Layers whose parent is also being grouped keep their existing
        # parent-child relationship so clips_parent hierarchies are preserved.
        for lid in all_ids:
            layer = self.get(lid)
            if layer is None:
                continue
            # Skip children whose parent is also being grouped —
            # they stay attached to their existing parent.
            if layer.parent_id and layer.parent_id in selected_set:
                continue
            # Remove from any previous parent
            if layer.parent_id:
                old_parent = self.get(layer.parent_id)
                if old_parent and lid in old_parent.children:
                    old_parent.children.remove(lid)
            layer.parent_id = group.id
            group.children.append(lid)

        self._active_index = self._layers.index(group)
        self.update_group_bbox(group)
        return group

    def reorder_by_ids(self, ordered_ids: list[str]) -> None:
        """Reorder the internal list to match *ordered_ids* (bottom → top).

        Any IDs not present in *ordered_ids* are appended at the end.
        """
        id_to_layer = {layer.id: layer for layer in self._layers}
        new_order: list[Layer] = []
        for lid in ordered_ids:
            if lid in id_to_layer:
                new_order.append(id_to_layer.pop(lid))
        # Append any remaining layers that weren't mentioned
        for layer in self._layers:
            if layer.id in id_to_layer:
                new_order.append(layer)
                del id_to_layer[layer.id]
        self._layers = new_order
        # Keep active_index in bounds
        if self._active_index >= len(self._layers):
            self._active_index = max(0, len(self._layers) - 1)

    def reposition_before(self, layer_id: str, target_id: str) -> None:
        """Move *layer_id* to the position just before *target_id* in the stack."""
        layer = self.get(layer_id)
        target = self.get(target_id)
        if layer is None or target is None:
            return
        self._layers = [l for l in self._layers if l.id != layer_id]
        for i, l in enumerate(self._layers):
            if l.id == target_id:
                self._layers.insert(i, layer)
                return
        # Fallback: append
        self._layers.append(layer)

    # ---- Dunder -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):
        return iter(self._layers)

    def __getitem__(self, index: int) -> Layer:
        return self._layers[index]

    # ---- Mask layer operations ----------------------------------------------

    def add_mask_layer(
        self,
        target_id: str | None,
        width: int,
        height: int,
        fill_white: bool = True,
        name: str | None = None,
    ) -> Layer | None:
        """Create a new MASK layer and optionally attach it to *target_id*.

        Parameters
        ----------
        target_id : str | None
            Parent layer ID.  ``None`` creates a standalone mask layer.
        width, height : int
            Canvas dimensions for the mask.
        fill_white : bool
            If *True* the mask starts fully white (opaque); otherwise black.
        name : str | None
            Custom name — auto-generated if *None*.

        Returns the newly created mask layer.
        """
        target = self.get(target_id) if target_id else None
        idx_name = 1
        if target:
            idx_name = len(target.mask_layers) + 1
        auto_name = name or (
            f"Mask {idx_name}" if target else "Mask Layer"
        )
        mask_layer = Layer(
            name=auto_name,
            width=width,
            height=height,
            layer_type=LayerType.MASK,
        )
        # Fill with white (fully visible) or black (fully hidden)
        fill_val = 1.0 if fill_white else 0.0
        mask_layer._pixels[:] = np.array(
            [fill_val, fill_val, fill_val, 1.0], dtype=np.float32
        )
        mask_layer.touch()

        if target:
            mask_layer.parent_id = target_id
            # Inherit the target's position so brush and compositing align
            mask_layer.position = target.position
            target.mask_layers.append(mask_layer.id)
            # Insert just before the target so that compositing order works
            target_idx = 0
            for i, layer in enumerate(self._layers):
                if layer.id == target_id:
                    target_idx = i
                    break
            self._layers.insert(target_idx, mask_layer)
            self._active_index = target_idx
        else:
            # Standalone mask layer — add at the top
            self._layers.append(mask_layer)
            self._active_index = len(self._layers) - 1

        return mask_layer

    def remove_mask_layer(self, mask_layer_id: str) -> Layer | None:
        """Remove a mask layer, detaching it from any parent."""
        mask_layer = self.get(mask_layer_id)
        if mask_layer is None:
            return None
        # Detach from parent — record the ex-parent so compositing
        # can scope the now-standalone mask to only its former parent.
        if mask_layer.parent_id:
            mask_layer.ex_parent_id = mask_layer.parent_id
            parent = self.get(mask_layer.parent_id)
            if parent and mask_layer_id in parent.mask_layers:
                parent.mask_layers.remove(mask_layer_id)
        return self.remove(mask_layer_id)

    def convert_layer_to_mask(self, layer_id: str, target_id: str) -> Layer | None:
        """Convert an existing raster layer into a mask layer for *target_id*.

        The layer's pixel data is preserved — its luminance will be used
        as the mask intensity.
        """
        layer = self.get(layer_id)
        target = self.get(target_id)
        if layer is None or target is None:
            return None
        if layer.layer_type == LayerType.MASK:
            return layer  # already a mask

        layer.layer_type = LayerType.MASK
        # Remove from any previous group parent
        if layer.parent_id:
            old_parent = self.get(layer.parent_id)
            if old_parent and layer_id in old_parent.children:
                old_parent.children.remove(layer_id)
        layer.parent_id = target_id
        target.mask_layers.append(layer_id)

        # Reposition just before the target
        for i, l in enumerate(self._layers):
            if l.id == layer_id:
                self._layers.pop(i)
                break
        target_idx = 0
        for i, l in enumerate(self._layers):
            if l.id == target_id:
                target_idx = i
                break
        self._layers.insert(target_idx, layer)
        return layer

    def get_mask_layers(self, layer_id: str) -> list[Layer]:
        """Return all MASK layers attached to *layer_id*, in order."""
        layer = self.get(layer_id)
        if layer is None:
            return []
        result: list[Layer] = []
        for mid in layer.mask_layers:
            mask = self.get(mid)
            if mask is not None:
                result.append(mask)
        return result

    def selection_to_mask_layer(
        self,
        target_id: str | None,
        selection_mask: np.ndarray,
        width: int,
        height: int,
    ) -> Layer | None:
        """Create a MASK layer from a selection mask array.

        *selection_mask* should be a float32 (H,W) array in [0,1].
        """
        mask_layer = self.add_mask_layer(
            target_id, width, height, fill_white=False,
            name="Mask from Selection",
        )
        if mask_layer is None:
            return None
        # Broadcast the grayscale selection into RGB channels, keep alpha=1
        mask_layer._pixels[..., 0] = selection_mask
        mask_layer._pixels[..., 1] = selection_mask
        mask_layer._pixels[..., 2] = selection_mask
        mask_layer._pixels[..., 3] = 1.0
        mask_layer.touch()
        return mask_layer
