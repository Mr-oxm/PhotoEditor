"""Document model — represents a single open image project."""

from __future__ import annotations

import numpy as np

from .enums import BlendMode, LayerType
from .history import HistoryManager, HistoryState
from .layer import Layer
from .layer_stack import LayerStack
from .selection import Selection


def _shared_meta(prev, lid: str, key: str, version: int, build):
    """Reuse the previous snapshot's serialised metadata when unchanged.

    Vector and text layers serialise their whole scene graph into every
    snapshot. Those are deep Python structures, one per snapshot per layer,
    and the history byte budget only counts numpy arrays -- so unshared they
    grow with no bound the budget can see or reclaim.
    """
    if prev is not None and prev.layer_versions.get(f"{key}:{lid}") == version:
        cached = prev.metadata.get(f"{key}:{lid}")
        if cached is not None:
            return cached
    return build()


class Document:
    """Top-level container for an editing session."""

    def __init__(
        self,
        width: int,
        height: int,
        name: str = "Untitled",
        *,
        color_mode: str = "RGB",
        color_profile: str = "sRGB IEC61966-2.1",
        unit: str = "px",
    ) -> None:
        self.name = name
        self.width = width
        self.height = height
        self.file_path: str | None = None
        self.dpi: int = 72

        # Color and unit metadata
        self.color_mode: str = color_mode
        self.color_profile: str = color_profile
        self.unit: str = unit  # display unit used in rulers / dialogs

        self.layers = LayerStack()
        self.history = HistoryManager()
        self.selection = Selection(width, height)
        self._dirty = False

        bg = Layer(name="Background", width=width, height=height)
        bg.pixels[:] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.layers.add(bg)

    # ---- Dirty flag ---------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_dirty(self) -> None:
        self._dirty = True

    def mark_clean(self) -> None:
        self._dirty = False

    # ---- Layer operations ---------------------------------------------------

    def add_layer(
        self, name: str = "Layer", layer_type: LayerType = LayerType.RASTER,
    ) -> Layer:
        layer = Layer(name=name, width=self.width, height=self.height, layer_type=layer_type)
        self.layers.add(layer)
        self._snapshot(f"Add {name}")
        self._dirty = True
        # The snapshot freezes every layer for copy-on-write. A layer we are
        # handing straight back to the caller is almost always about to be
        # filled, so un-share it now rather than making every caller do it.
        layer.begin_write()
        return layer

    def add_group(self, name: str = "Group") -> Layer:
        group = Layer(name=name, width=self.width, height=self.height, layer_type=LayerType.GROUP)
        self.layers.add(group)
        self.layers.update_group_bbox(group)  # Empty group: position (0,0), minimal size
        self._snapshot(f"Add Group {name}")
        self._dirty = True
        group.begin_write()
        return group

    def group_selected_layers(self, layer_ids: list[str], name: str = "Group") -> Layer | None:
        """Create a new group containing the given layers."""
        group = self.layers.create_group_from(layer_ids, name)
        if group:
            self._snapshot("Group Layers")
            self._dirty = True
        return group

    def place_image(self, pixels: np.ndarray, name: str = "Placed Image") -> Layer:
        """Import an RGBA image as a new layer."""
        h, w = pixels.shape[:2]
        layer = Layer(name=name, width=w, height=h)
        layer.pixels = pixels
        self.layers.add(layer)
        self._snapshot(f"Place {name}")
        self._dirty = True
        layer.begin_write()
        return layer

    def add_vector_layer(self, name: str = "Vector Layer") -> Layer:
        """Create a new layer with an empty VectorLayer scene graph.

        Starts with a tiny 1×1 buffer — the first rasterize pass will resize
        it to the tight bounding box of its contents.
        """
        from ..vector.scene import VectorLayer as VL
        layer = Layer(
            name=name, width=1, height=1,
            layer_type=LayerType.SHAPE,
        )
        layer._vector_data = VL()
        self.layers.add(layer)
        self._snapshot(f"Add {name}")
        self._dirty = True
        layer.begin_write()
        return layer

    def remove_layer(self, layer_id: str) -> None:
        layer = self.layers.get(layer_id)
        if layer is None:
            return
        # If this is a group, recursively remove all children first
        if layer.layer_type == LayerType.GROUP:
            child_ids = [
                c.id for c in list(self.layers)
                if c.parent_id == layer_id
            ]
            for cid in child_ids:
                self.remove_layer(cid)
        # If this layer has mask layers, remove them too
        for mid in list(layer.mask_layers):
            ml = self.layers.get(mid)
            if ml is not None:
                self.layers.remove(mid)
        # If this layer has child adj/filter layers, remove them too
        adj_child_ids = [
            c.id for c in list(self.layers)
            if c.parent_id == layer_id
            and c.layer_type in (LayerType.ADJUSTMENT, LayerType.FILTER)
        ]
        for cid in adj_child_ids:
            self.layers.remove(cid)
        # If this is a mask layer, detach from parent
        if layer.layer_type == LayerType.MASK and layer.parent_id:
            parent = self.layers.get(layer.parent_id)
            if parent and layer_id in parent.mask_layers:
                parent.mask_layers.remove(layer_id)
        removed = self.layers.remove(layer_id)
        if removed:
            self._snapshot(f"Delete {removed.name}")
            self._dirty = True

    def duplicate_layer(self, layer_id: str) -> Layer | None:
        original = self.layers.get(layer_id)
        if original is None:
            return None

        dup = self.layers.duplicate(layer_id)
        if dup is None:
            return None

        # Deep-duplicate all child layers (masks, adjustments, filters,
        # clips_parent children) and remap references to the new parent.
        id_map: dict[str, str] = {original.id: dup.id}

        # Collect children in stack order so duplicates keep the same order.
        child_layers = [
            c for c in list(self.layers)
            if c.parent_id == original.id and c.id != dup.id
        ]

        # Determine insertion point: right before the duplicate parent
        dup_idx = self.layers.layers.index(dup)

        for child in child_layers:
            child_dup = child.duplicate()
            id_map[child.id] = child_dup.id
            child_dup.parent_id = dup.id
            child_dup.clips_parent = child.clips_parent
            child_dup.clipping_mask = child.clipping_mask
            self.layers.add(child_dup, dup_idx)
            # Each child inserted before dup shifts dup_idx
            dup_idx = self.layers.layers.index(dup)

        # Remap the duplicate's children and mask_layers lists to new IDs
        dup.children = [id_map[cid] for cid in original.children if cid in id_map]
        dup.mask_layers = [id_map[mid] for mid in original.mask_layers if mid in id_map]

        # Deep-duplicate mask layers for each duplicated child that also
        # had mask layers (recursive one level — mask layers on children).
        for child in child_layers:
            child_dup_id = id_map[child.id]
            child_dup = self.layers.get(child_dup_id)
            if child_dup is None:
                continue
            if child.mask_layers:
                new_mask_ids = []
                cd_idx = self.layers.layers.index(child_dup)
                for mid in child.mask_layers:
                    ml = self.layers.get(mid)
                    if ml is None:
                        continue
                    ml_dup = ml.duplicate()
                    ml_dup.parent_id = child_dup_id
                    self.layers.add(ml_dup, cd_idx)
                    new_mask_ids.append(ml_dup.id)
                    cd_idx = self.layers.layers.index(child_dup)
                child_dup.mask_layers = new_mask_ids
            else:
                child_dup.mask_layers = []

        self._snapshot(f"Duplicate {dup.name}")
        self._dirty = True
        return dup

    def flatten(self) -> None:
        """Merge all visible layers into the background."""
        from ..engine.render_pipeline import RenderPipeline
        pipeline = RenderPipeline()
        merged = pipeline.execute(self)
        # Remove all layers, create single flattened one
        self.layers = LayerStack()
        bg = Layer(name="Background", width=self.width, height=self.height)
        bg.pixels = merged
        bg.locked = True
        self.layers.add(bg)
        self._snapshot("Flatten Image")
        self._dirty = True

    def merge_down(self) -> bool:
        """Merge the active layer onto the layer directly below it.

        Returns ``True`` on success, ``False`` if there is nothing to
        merge (e.g. no active layer, no layer below, or a group).
        """
        active = self.layers.active_layer
        if active is None:
            return False

        # Find the active layer's index in the flat list
        idx = self.layers.active_index
        if idx <= 0:
            return False  # nothing below

        below = self.layers.layers[idx - 1]

        # Skip non-raster targets (groups, adjustments, masks, etc.)
        if below.layer_type != LayerType.RASTER:
            return False
        if active.layer_type != LayerType.RASTER:
            return False

        from ..blending.blending_engine import BlendingEngine

        # Build document-sized canvases for both layers
        canvas_below = np.zeros((self.height, self.width, 4), dtype=np.float32)
        bx, by = below.position
        bp = below.pixels
        bh, bw = bp.shape[:2]
        # Clip to canvas bounds
        sx0, sy0 = max(0, bx), max(0, by)
        sx1 = min(self.width, bx + bw)
        sy1 = min(self.height, by + bh)
        if sx1 > sx0 and sy1 > sy0:
            canvas_below[sy0:sy1, sx0:sx1] = bp[sy0 - by:sy1 - by, sx0 - bx:sx1 - bx]

        canvas_top = np.zeros((self.height, self.width, 4), dtype=np.float32)
        ax, ay = active.position
        ap = active.pixels
        ah, aw = ap.shape[:2]
        tx0, ty0 = max(0, ax), max(0, ay)
        tx1 = min(self.width, ax + aw)
        ty1 = min(self.height, ay + ah)
        if tx1 > tx0 and ty1 > ty0:
            canvas_top[ty0:ty1, tx0:tx1] = ap[ty0 - ay:ty1 - ay, tx0 - ax:tx1 - ax]

        merged = BlendingEngine.blend(
            canvas_below, canvas_top,
            mode=active.blend_mode,
            opacity=active.opacity,
        )

        # Crop merged back to the below-layer bounds
        if sx1 > sx0 and sy1 > sy0:
            below.pixels = merged[sy0:sy1, sx0:sx1].copy()
        else:
            below.pixels = merged

        # Remove the active layer (and its mask/adj children)
        self.remove_layer(active.id)
        self.layers.active_index = idx - 1
        self._snapshot("Merge Down")
        self._dirty = True
        return True

    # ---- Mask layer operations ----------------------------------------------

    def add_mask_layer(
        self,
        target_id: str | None = None,
        fill_white: bool = True,
        name: str | None = None,
    ) -> Layer | None:
        """Add a mask layer to the document.

        Parameters
        ----------
        target_id : str | None
            If provided, the mask is attached as a child of this layer.
            If ``None`` and there is an active layer, it attaches to that.
            Pass ``"__standalone__"`` to force a standalone mask layer.
        fill_white : bool
            White = fully visible (default); black = fully hidden.
        name : str | None
            Custom name for the mask layer.
        """
        standalone = target_id == "__standalone__"
        if standalone:
            target_id = None
        elif target_id is None and self.layers.active_layer is not None:
            active = self.layers.active_layer
            # Don't attach a mask to another mask layer
            if active.layer_type != LayerType.MASK:
                target_id = active.id

        # Use the target layer's current pixel dimensions so the mask
        # matches a transformed (scaled/rotated) layer correctly.
        mw, mh = self.width, self.height
        if target_id:
            target = self.layers.get(target_id)
            if target is not None:
                mw, mh = target.width, target.height

        mask = self.layers.add_mask_layer(
            target_id, mw, mh,
            fill_white=fill_white, name=name,
        )
        if mask:
            self._snapshot(f"Add Mask Layer")
            self._dirty = True
        return mask

    def remove_mask_layer(self, mask_layer_id: str) -> None:
        """Remove a mask layer from the document."""
        removed = self.layers.remove_mask_layer(mask_layer_id)
        if removed:
            self._snapshot(f"Remove Mask {removed.name}")
            self._dirty = True

    def selection_to_mask_layer(self, target_id: str | None = None) -> Layer | None:
        """Convert the current selection to a mask layer.

        If no target_id is given, attaches to the active layer.
        The selection is cropped to the target layer's spatial extent so
        the mask layer dimensions and position match the target.
        """
        if not self.selection.active or self.selection.mask is None:
            return None
        if target_id is None and self.layers.active_layer is not None:
            active = self.layers.active_layer
            if active.layer_type != LayerType.MASK:
                target_id = active.id

        sel_mask = self.selection.mask
        mw, mh = self.width, self.height

        # When attaching to a target layer, crop the canvas-sized selection
        # to the target's spatial extent so the mask is properly aligned.
        if target_id:
            target = self.layers.get(target_id)
            if target is not None:
                lx, ly = target.position
                tw, th = target.width, target.height
                mw, mh = tw, th
                dh, dw = sel_mask.shape[:2]
                cropped = np.zeros((th, tw), dtype=np.float32)
                # Compute overlapping region
                dy0, dy1 = max(0, ly), min(dh, ly + th)
                dx0, dx1 = max(0, lx), min(dw, lx + tw)
                if dy1 > dy0 and dx1 > dx0:
                    sy0, sy1 = dy0 - ly, dy1 - ly
                    sx0, sx1 = dx0 - lx, dx1 - lx
                    cropped[sy0:sy1, sx0:sx1] = sel_mask[dy0:dy1, dx0:dx1]
                sel_mask = cropped

        mask = self.layers.selection_to_mask_layer(
            target_id, sel_mask, mw, mh,
        )
        if mask:
            self._snapshot("Selection to Mask")
            self._dirty = True
        return mask

    def convert_layer_to_mask(self, layer_id: str, target_id: str | None = None) -> Layer | None:
        """Convert an existing layer to a mask layer.

        If *target_id* is ``None``, the layer directly above in the stack is used.
        """
        if target_id is None:
            # Find the layer directly above this one
            for i, l in enumerate(self.layers):
                if l.id == layer_id and i + 1 < len(self.layers):
                    target_id = self.layers[i + 1].id
                    break
        if target_id is None:
            return None
        result = self.layers.convert_layer_to_mask(layer_id, target_id)
        if result:
            self._snapshot("Convert to Mask")
            self._dirty = True
        return result

    def apply_mask_layer(self, mask_layer_id: str) -> None:
        """Burn a mask layer into its parent's old-style single mask, then remove it."""
        mask_layer = self.layers.get(mask_layer_id)
        if mask_layer is None or mask_layer.layer_type != LayerType.MASK:
            return
        parent = self.layers.get(mask_layer.parent_id) if mask_layer.parent_id else None
        if parent is None:
            return
        # Combine this mask layer's grayscale into the parent's alpha
        grayscale = mask_layer.get_mask_grayscale()
        parent.begin_write()
        parent.pixels[..., 3] *= grayscale
        self.layers.remove_mask_layer(mask_layer_id)
        self._snapshot("Apply Mask Layer")
        self._dirty = True

    # ---- History ------------------------------------------------------------

    def _save_live_state(self) -> None:
        """Push the current uncommitted changes to history as __Live__ so we can return to it."""
        if not self.history.states:
            return
        if self.history.states[-1].name == "__Live__":
            return
        # We only want to push the live state if we are currently AT the end of history
        if self.history.current_index == len(self.history.states): 
            # Note: history.current_index has +1 offset when at end!
            self._snapshot("__Live__")
            self._dirty = True

    def undo(self) -> None:
        if self.history.current_index == len(self.history.states) and getattr(self, "history").states[-1].name != "__Live__":
            self._save_live_state()
        state = self.history.undo()
        if state:
            self._restore(state)

    def redo(self) -> None:
        # Redo doesn't need to save live, because if we can redo, we are NOT at the end
        state = self.history.redo()
        if state:
            self._restore(state)

    def navigate_history(self, target_index: int) -> None:
        """Jump to a specific history state by index."""
        if self.history.current_index == len(self.history.states) and getattr(self, "history").states[-1].name != "__Live__":
            self._save_live_state()
            
        while self.history.current_index > target_index and self.history.can_undo:
            self.history.undo()
        while self.history.current_index < target_index and self.history.can_redo:
            self.history.redo()
        current = self.history.current()
        if current:
            self._restore(current)

    def save_snapshot(self, action: str) -> None:
        self._snapshot(action)

    def _build_history_state(self, action: str) -> HistoryState:
        """Create a serializable snapshot of the current document state."""
        state = HistoryState(name=action)
        # Save pixel and mask data for every layer, sharing buffers with the
        # previous snapshot for layers that have not changed. An edit usually
        # touches one layer, so this turns an O(all layers) deep copy into an
        # O(changed layers) one -- 2,531 MB -> ~127 MB for a 20-layer 4K doc.
        prev = self.history.latest()

        def _capture(key: str, arr, version: int) -> None:
            """Store *arr* under *key* without copying.

            History takes a *reference*; the layer is then frozen, so the
            next in-place edit copy-on-writes and leaves this snapshot
            holding the old content. Unchanged layers additionally reuse
            the previous state's entry, so a static layer costs nothing at
            all no matter how many snapshots are taken.
            """
            if arr is None:
                return
            if (prev is not None
                    and prev.layer_versions.get(key) == version
                    and key in prev.layer_data):
                state.layer_data[key] = prev.layer_data[key]
            else:
                state.layer_data[key] = arr
            state.layer_versions[key] = version

        for layer in self.layers:
            version = layer.content_version
            _capture(layer.id, layer.pixels, version)
            _capture(f"_src_{layer.id}", layer._source_pixels, version)
            _capture(f"_srcmask_{layer.id}", layer._source_mask, version)
            _capture(f"_mask_{layer.id}", layer._mask, version)
            layer.freeze()
        # Save the full layer structure so add/remove can be undone
        layer_metas = []
        for layer in self.layers:
            lid = layer.id
            version = layer.content_version
            meta = {
                "id": layer.id,
                "name": layer.name,
                "width": layer.width,
                "height": layer.height,
                "layer_type": layer.layer_type,
                "opacity": layer.opacity,
                "blend_mode": layer.blend_mode,
                "visible": layer.visible,
                "locked": layer.locked,
                "position": layer.position,
                "mask_enabled": layer.mask_enabled,
                "clipping_mask": layer.clipping_mask,
                "parent_id": layer.parent_id,
                "children": list(layer.children),
                "mask_layers": list(layer.mask_layers),
                "ex_parent_id": layer.ex_parent_id,
                "transform_angle": layer.transform_angle,
                "transform_scale_x": layer.transform_scale_x,
                "transform_scale_y": layer.transform_scale_y,
                "transform_base_w": layer.transform_base_w,
                "transform_base_h": layer.transform_base_h,
            }
            # Save text layer data if present. Shared with the previous
            # snapshot when the layer has not changed: these are deep dict
            # trees, one per snapshot per layer, and the history byte budget
            # only counts numpy arrays -- so unshared they grow without any
            # bound the budget can see.
            td = getattr(layer, "_text_data", None)
            if td is not None:
                meta["_text_data"] = _shared_meta(
                    prev, lid, "_text_data", version, td.to_dict)
                state.metadata[f"_text_data:{lid}"] = meta["_text_data"]
                state.layer_versions[f"_text_data:{lid}"] = version
            # Save adjustment / filter layer data if present
            if layer.adjustment is not None:
                meta["_adjustment_name"] = layer.adjustment.name
                meta["_adjustment_params"] = dict(layer.adjustment_params)
            # Save vector layer data if present -- shared as above. An
            # imported SVG re-serialised on every snapshot was measured at
            # ~0.66 MB per snapshot for a small scene, invisible to the
            # budget and bounded only by the 200-state cap.
            vd = getattr(layer, "_vector_data", None)
            if vd is not None and hasattr(vd, "to_dict"):
                meta["_vector_data"] = _shared_meta(
                    prev, lid, "_vector_data", version, vd.to_dict)
                state.metadata[f"_vector_data:{lid}"] = meta["_vector_data"]
                state.layer_versions[f"_vector_data:{lid}"] = version
            layer_metas.append(meta)
        state.metadata["_layer_order"] = [l.id for l in self.layers]
        state.metadata["_layer_meta"] = {m["id"]: m for m in layer_metas}
        state.metadata["_active_index"] = self.layers.active_index
        state.metadata["_doc_width"] = self.width
        state.metadata["_doc_height"] = self.height
        # Save selection mask, sharing when the mask object is unchanged.
        sel_mask = self.selection._mask
        if sel_mask is not None:
            # Selection has no copy-on-write hook and mutates in place, so
            # this one is still copied -- it is a single-channel mask, ~1/16
            # the cost of a layer.
            # Keyed on the selection's version, not id(sel_mask): CPython
            # reuses the addresses of freed ndarrays, so a new selection
            # could land on the old one's address and the snapshot would
            # silently share -- and restore -- the *previous* selection.
            version = self.selection.version
            prev_sel = prev.layer_data.get("__selection_mask__") if prev else None
            prev_ver = prev.layer_versions.get("__selection_mask__") if prev else None
            if prev_sel is not None and prev_ver == version:
                state.layer_data["__selection_mask__"] = prev_sel
            else:
                state.layer_data["__selection_mask__"] = sel_mask.copy()
            state.layer_versions["__selection_mask__"] = version
        return state

    def freeze_for_read(self) -> None:
        """Mark every layer's buffers read-only without touching history.

        A background save reads layer pixels for several seconds while the
        user keeps painting. Painting writes in place, so the file could
        contain half-finished strokes -- or, because the saver scans each
        array's min/max to decide whether to quantise, a buffer that changed
        between the scan and the write.

        Freezing reuses the copy-on-write machinery the undo system already
        relies on: the reader sees stable arrays, and the next edit takes a
        private copy rather than mutating underneath it.
        """
        for layer in self.layers:
            layer.freeze()

    def live_buffer_ids(self) -> set[int]:
        """Identities of every array the live layer stack currently holds.

        History shares buffers with the document until a layer
        copy-on-writes, so its memory budget must discount these.
        """
        ids: set[int] = set()
        for layer in self.layers:
            for arr in (layer._pixels, layer._mask,
                        layer._source_pixels, layer._source_mask):
                if arr is not None:
                    ids.add(id(arr))
        return ids

    def _snapshot(self, action: str) -> None:
        state = self._build_history_state(action)
        self.history.push(state, live_ids=self.live_buffer_ids())
        # Every snapshot represents an edit — mark document dirty so the
        # unsaved-changes guard in closeEvent / tab-close picks it up.
        # (Structural ops like add_layer set _dirty explicitly too, but tool
        # strokes only call save_snapshot() → _snapshot(), so we must set it
        # here to cover that path.)
        if action != "__Live__":
            self._dirty = True

    def _restore(self, state: HistoryState) -> None:
        order: list[str] | None = state.metadata.get("_layer_order")
        meta_map: dict | None = state.metadata.get("_layer_meta")

        if order is not None and meta_map is not None:
            # Rebuild the layer stack from the snapshot
            from .layer_stack import LayerStack
            new_stack = LayerStack()
            for lid in order:
                meta = meta_map[lid]
                layer = Layer(
                    name=meta["name"],
                    width=meta["width"],
                    height=meta["height"],
                    layer_type=meta["layer_type"],
                    id=lid,
                    opacity=meta["opacity"],
                    blend_mode=meta["blend_mode"],
                    visible=meta["visible"],
                    locked=meta["locked"],
                    position=meta["position"],
                    mask_enabled=meta["mask_enabled"],
                    clipping_mask=meta["clipping_mask"],
                    parent_id=meta["parent_id"],
                    transform_angle=meta.get("transform_angle", 0.0),
                    transform_scale_x=meta.get("transform_scale_x", 1.0),
                    transform_scale_y=meta.get("transform_scale_y", 1.0),
                    transform_base_w=meta.get("transform_base_w", 0),
                    transform_base_h=meta.get("transform_base_h", 0),
                )
                # Restore children list and mask_layers
                layer.children = list(meta.get("children", []))
                layer.mask_layers = list(meta.get("mask_layers", []))
                layer.ex_parent_id = meta.get("ex_parent_id")
                # Restore pixel data by reference and re-freeze: the layer
                # copy-on-writes on its next edit, so undo itself is O(1) in
                # pixel bytes instead of copying the whole stack.
                if lid in state.layer_data:
                    layer._adopt(state.layer_data[lid], frozen=True)
                src_key = f"_src_{lid}"
                if src_key in state.layer_data:
                    layer._source_pixels = state.layer_data[src_key]
                srcmask_key = f"_srcmask_{lid}"
                if srcmask_key in state.layer_data:
                    layer._source_mask = state.layer_data[srcmask_key]
                # Restore mask data
                mask_key = f"_mask_{lid}"
                if mask_key in state.layer_data:
                    layer._mask = state.layer_data[mask_key]
                layer.freeze()
                # Restore text layer data
                td_dict = meta.get("_text_data")
                if td_dict is not None:
                    from .text_layer import TextLayerData
                    layer._text_data = TextLayerData.from_dict(td_dict)
                # Restore adjustment / filter layer data
                adj_name = meta.get("_adjustment_name")
                if adj_name is not None:
                    from ..registries import get_adjustment_class, get_filter_name_map
                    layer_lt = meta.get("layer_type")
                    if layer_lt == LayerType.FILTER:
                        cls = get_filter_name_map().get(adj_name)
                    else:
                        cls = get_adjustment_class(adj_name)
                    if cls is not None:
                        layer._adjustment = cls()
                        layer._adjustment_params = dict(meta.get("_adjustment_params", {}))
                # Restore vector layer data
                vd_dict = meta.get("_vector_data")
                if vd_dict is not None:
                    try:
                        from ..vector.scene import VectorLayer as VL
                        layer._vector_data = VL.from_dict(vd_dict)
                    except Exception:
                        pass
                new_stack.add(layer)
            new_stack.active_index = state.metadata.get("_active_index", 0)
            self.layers = new_stack
        else:
            # Legacy fallback: only pixel data & positions stored
            for layer in self.layers:
                if layer.id in state.layer_data:
                    layer.pixels = state.layer_data[layer.id].copy()
                pos_key = f"pos_{layer.id}"
                if pos_key in state.metadata:
                    layer.position = state.metadata[pos_key]
        # Restore document dimensions (canvas crop undo)
        saved_w = state.metadata.get("_doc_width")
        saved_h = state.metadata.get("_doc_height")
        if saved_w is not None and saved_h is not None:
            self.width = saved_w
            self.height = saved_h
            self.selection.resize(saved_w, saved_h)
        # Restore selection mask
        sel_key = "__selection_mask__"
        if sel_key in state.layer_data:
            self.selection._set_mask(state.layer_data[sel_key].copy())
        else:
            self.selection._set_mask(None)
        self._dirty = True

    # ---- Canvas ops ---------------------------------------------------------

    def resize(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.selection.resize(width, height)
