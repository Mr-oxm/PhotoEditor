"""Layer compositor operating in planar (4, H, W) space.

This is a semantics-preserving port of :mod:`photo_editor.engine.compositor`.
Every clipping, grouping, masking and adjustment rule is reproduced exactly --
the render-fidelity suite (``tests/test_render_fidelity.py``) pins the output
to within 1/255 of the original for 42 reference scenes.

What changed is the memory layout. The interleaved compositor spent most of
its time in ``[..., :3]`` slices whose innermost stride is 4 floats, which
NumPy cannot vectorise; see :mod:`photo_editor.blending.planar` for the
measurements. Here the canvas is ``(4, H, W)``, so the colour block and the
alpha plane are each contiguous.

Layer pixel data stays interleaved -- that is what tools, filters and
adjustments expect. Conversion happens once per layer per frame and is
cached by :class:`~photo_editor.engine.layer_cache.LayerRasterCache`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..blending.planar import (
    PlanarScratch, blend_planar_region, to_interleaved, to_planar,
)
from ..core.enums import LayerType
from ..core.layer import Layer
from ..core.layer_stack import LayerStack
from ..masks.mask_manager import MaskManager
from ..styles.style_engine import StyleEngine

if TYPE_CHECKING:
    from .layer_cache import LayerRasterCache


class PlanarCompositor:
    """Composites a LayerStack into a flat planar RGBA buffer."""

    def __init__(self, cache: "LayerRasterCache | None" = None) -> None:
        self._scratch = PlanarScratch(max_per_shape=3)
        self._cache = cache

    # ------------------------------------------------------------------
    # Padding / adjustment helpers (unchanged semantics)
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_filter_padding(adj_layers: list[Layer]) -> int:
        """Pixel padding needed so a blur can extend past the layer edge."""
        pad = 0
        for adj_layer in adj_layers:
            if adj_layer.layer_type != LayerType.FILTER:
                continue
            params = adj_layer.adjustment_params or {}
            r = params.get("radius",
                           params.get("distance", params.get("amount", 0)))
            try:
                pad = max(pad, int(float(r) * 3) + 4)
            except (TypeError, ValueError):
                pass
        return pad

    def _apply_filters_padded(
        self, pixels: np.ndarray, adj_layers: list[Layer],
    ) -> tuple[np.ndarray, int]:
        """Apply adj/filter layers to interleaved *pixels*, with blur padding."""
        pad = self._calc_filter_padding(adj_layers)
        if pad > 0:
            h, w = pixels.shape[:2]
            padded = np.zeros((h + 2 * pad, w + 2 * pad, 4), dtype=np.float32)
            padded[pad:pad + h, pad:pad + w] = pixels
            pixels = padded
        else:
            pixels = pixels.copy()
        for adj_layer in adj_layers:
            adj = adj_layer.adjustment
            if adj is not None:
                pixels = adj.apply(pixels, adj_layer.adjustment_params)
        np.clip(pixels, 0, 1, out=pixels)
        return pixels, pad

    @staticmethod
    def _apply_channels(pixels: np.ndarray, layer: Layer) -> np.ndarray:
        if layer.channel_r and layer.channel_g and layer.channel_b and layer.channel_a:
            return pixels
        res = pixels.copy()
        if not layer.channel_r: res[..., 0] = 0.0
        if not layer.channel_g: res[..., 1] = 0.0
        if not layer.channel_b: res[..., 2] = 0.0
        if not layer.channel_a: res[..., 3] = 0.0
        return res

    def _get_effective_mask(self, layer: Layer, stack: LayerStack) -> np.ndarray | None:
        return MaskManager.get_combined_mask(layer, stack)

    # ------------------------------------------------------------------
    # Per-layer pixel preparation
    # ------------------------------------------------------------------

    def _prepare_layer(
        self, layer: Layer, adj_children: dict[str, list[Layer]],
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Return (planar pixels, blend position) for a raster-like layer.

        Applies styles, channel toggles and scoped adjustment/filter
        children, then converts to planar. Cached when a cache is attached.
        """
        if self._cache is not None:
            cached = self._cache.get_prepared(layer, adj_children.get(layer.id))
            if cached is not None:
                return cached

        pixels = layer.pixels
        if layer.styles:
            pixels = StyleEngine.apply_styles(pixels, layer.styles)
        pixels = self._apply_channels(pixels, layer)
        blend_pos = layer.position
        kids = adj_children.get(layer.id)
        if kids:
            pixels, pad = self._apply_filters_padded(pixels, kids)
            if pad > 0:
                blend_pos = (layer.position[0] - pad, layer.position[1] - pad)

        planar = to_planar(pixels)
        result = (planar, blend_pos)
        if self._cache is not None:
            self._cache.put_prepared(layer, kids, result)
        return result

    # ------------------------------------------------------------------
    # Placement helpers
    # ------------------------------------------------------------------

    def _place_planar(self, planar: np.ndarray, position: tuple[int, int],
                      cw: int, ch: int) -> np.ndarray:
        """Place planar *pixels* at *position* onto a canvas-sized planar buffer."""
        canvas = self._scratch.acquire((4, ch, cw), zero=True)
        lx, ly = position
        lh, lw = planar.shape[1], planar.shape[2]
        sx, sy = max(0, -lx), max(0, -ly)
        dx, dy = max(0, lx), max(0, ly)
        w = min(lw - sx, cw - dx)
        h = min(lh - sy, ch - dy)
        if w > 0 and h > 0:
            canvas[:, dy:dy + h, dx:dx + w] = planar[:, sy:sy + h, sx:sx + w]
        return canvas

    @staticmethod
    def _place_mask_array(mask: np.ndarray, position: tuple[int, int],
                          cw: int, ch: int) -> np.ndarray:
        canvas = np.zeros((ch, cw), dtype=np.float32)
        lx, ly = position
        mh, mw = mask.shape[:2]
        sx, sy = max(0, -lx), max(0, -ly)
        dx, dy = max(0, lx), max(0, ly)
        w = min(mw - sx, cw - dx)
        h = min(mh - sy, ch - dy)
        if w > 0 and h > 0:
            canvas[dy:dy + h, dx:dx + w] = mask[sy:sy + h, sx:sx + w]
        return canvas

    def _place_mask_combined(self, layer: Layer, stack: LayerStack,
                             cw: int, ch: int) -> np.ndarray | None:
        combined = self._get_effective_mask(layer, stack)
        if combined is None:
            return None
        return self._place_mask_array(combined, layer.position, cw, ch)

    # ------------------------------------------------------------------
    # Adjustment application on a canvas-sized planar buffer
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_adjustment_planar(planar: np.ndarray, adj, params) -> np.ndarray:
        """Run an interleaved-API adjustment over a planar buffer."""
        interleaved = to_interleaved(planar)
        result = adj.apply(interleaved, params or {})
        np.clip(result, 0, 1, out=result)
        return to_planar(result)

    # ------------------------------------------------------------------
    # Main composite
    # ------------------------------------------------------------------

    def composite(self, stack: LayerStack, width: int, height: int) -> np.ndarray:
        """Composite *stack* into a planar (4, height, width) float32 buffer."""
        canvas = np.zeros((4, height, width), dtype=np.float32)
        layers = list(stack)

        mask_layer_ids: set[str] = set()
        for l in layers:
            for mid in l.mask_layers:
                mask_layer_ids.add(mid)

        adj_children: dict[str, list[Layer]] = {}
        adj_child_ids: set[str] = set()
        for l in layers:
            if (l.parent_id
                    and l.layer_type in (LayerType.ADJUSTMENT, LayerType.FILTER)
                    and l.visible):
                adj_children.setdefault(l.parent_id, []).append(l)
                adj_child_ids.add(l.id)

        standalone_mask_ids: set[str] = set()
        for l in layers:
            if (l.layer_type == LayerType.MASK
                    and l.parent_id is None
                    and l.id not in mask_layer_ids):
                standalone_mask_ids.add(l.id)

        group_ids = {l.id for l in layers if l.layer_type == LayerType.GROUP}
        regular_children: dict[str, list[Layer]] = {}
        for l in layers:
            if (l.parent_id and l.visible
                    and l.parent_id not in group_ids
                    and l.layer_type not in (
                        LayerType.ADJUSTMENT, LayerType.FILTER, LayerType.MASK)
                    and l.id not in mask_layer_ids
                    and l.id not in adj_child_ids):
                regular_children.setdefault(l.parent_id, []).append(l)

        visible = [
            l for l in layers
            if l.visible and l.parent_id is None
            and l.id not in mask_layer_ids
            and (l.layer_type != LayerType.MASK or l.id in standalone_mask_ids)
            and l.id not in adj_child_ids
        ]
        needs_placed: set[str] = set()
        clippable = [
            l for l in visible
            if l.layer_type not in (LayerType.ADJUSTMENT, LayerType.FILTER)
        ]
        for i in range(len(clippable) - 1):
            if clippable[i + 1].clipping_mask:
                needs_placed.add(clippable[i].id)

        prev_img: np.ndarray | None = None
        borrowed: list[np.ndarray] = []

        def release_prev(buf: np.ndarray | None) -> None:
            if buf is not None and buf in borrowed:
                borrowed.remove(buf)
                self._scratch.release(buf)

        for layer in visible:
            # --- Root-level adjustment/filter: applies to the canvas ----
            if layer.layer_type in (LayerType.ADJUSTMENT, LayerType.FILTER):
                adj = layer.adjustment
                if adj is not None:
                    canvas = self._apply_adjustment_planar(
                        canvas, adj, layer.adjustment_params)
                release_prev(prev_img)
                prev_img = None
                continue

            # --- Standalone mask: attenuates the canvas built so far ----
            if layer.layer_type == LayerType.MASK and layer.id in standalone_mask_ids:
                gray = layer.get_mask_grayscale()
                placed_gray = self._place_mask_array(
                    gray, layer.position, width, height)
                if layer.ex_parent_id:
                    # Detached mask -- alpha only.
                    canvas[3] *= placed_gray
                else:
                    # Global standalone mask -- every channel.
                    canvas *= placed_gray
                continue

            if layer.layer_type == LayerType.GROUP:
                group_img = self._composite_group(layer, stack, width, height)
                if layer.id in adj_children:
                    for adj_layer in adj_children[layer.id]:
                        adj = adj_layer.adjustment
                        if adj is not None:
                            group_img = self._apply_adjustment_planar(
                                group_img, adj, adj_layer.adjustment_params)
                if layer.styles:
                    styled = StyleEngine.apply_styles(
                        to_interleaved(group_img), layer.styles)
                    np.clip(styled, 0, 1, out=styled)
                    group_img = to_planar(styled)
                if not (layer.channel_r and layer.channel_g
                        and layer.channel_b and layer.channel_a):
                    if not layer.channel_r: group_img[0] = 0.0
                    if not layer.channel_g: group_img[1] = 0.0
                    if not layer.channel_b: group_img[2] = 0.0
                    if not layer.channel_a: group_img[3] = 0.0
                group_mask = self._get_effective_mask(layer, stack)
                if group_mask is not None:
                    placed_mask = self._place_mask_array(
                        group_mask, layer.position, width, height)
                    group_img[3] *= placed_mask
                blend_planar_region(canvas, group_img, (0, 0),
                                    layer.blend_mode, layer.opacity)
                release_prev(prev_img)
                prev_img = group_img
                continue

            mask = self._get_effective_mask(layer, stack)
            pixels, blend_pos = self._prepare_layer(layer, adj_children)

            _has_clip_child = any(
                rc.clips_parent for rc in regular_children.get(layer.id, ()))

            if layer.clipping_mask and prev_img is not None:
                placed = self._place_planar(pixels, blend_pos, width, height)
                borrowed.append(placed)
                placed[3] *= prev_img[3]
                placed_mask = (
                    self._place_mask_combined(layer, stack, width, height)
                    if mask is not None else None
                )
                blend_planar_region(canvas, placed, (0, 0),
                                    layer.blend_mode, layer.opacity, placed_mask)
                release_prev(prev_img)
                prev_img = placed

            elif _has_clip_child:
                parent_placed = self._place_planar(
                    pixels, blend_pos, width, height)
                borrowed.append(parent_placed)
                for child in regular_children[layer.id]:
                    if not child.clips_parent:
                        continue
                    c_pix, c_pos = self._prepare_layer(child, adj_children)
                    c_placed = self._place_planar(c_pix, c_pos, width, height)
                    parent_placed[3] *= c_placed[3]
                    self._scratch.release(c_placed)
                placed_mask = (
                    self._place_mask_combined(layer, stack, width, height)
                    if mask is not None else None
                )
                blend_planar_region(canvas, parent_placed, (0, 0),
                                    layer.blend_mode, layer.opacity, placed_mask)
                for child in regular_children[layer.id]:
                    if child.clips_parent:
                        continue
                    self._blend_clipped_child(
                        canvas, child, stack, adj_children,
                        parent_placed, width, height)
                release_prev(prev_img)
                prev_img = parent_placed

            else:
                blend_planar_region(canvas, pixels, blend_pos,
                                    layer.blend_mode, layer.opacity, mask)
                release_prev(prev_img)
                if layer.id in needs_placed or layer.id in regular_children:
                    prev_img = self._place_planar(
                        pixels, blend_pos, width, height)
                    borrowed.append(prev_img)
                else:
                    prev_img = None

                if layer.id in regular_children:
                    parent_placed = prev_img
                    for child in regular_children[layer.id]:
                        self._blend_clipped_child(
                            canvas, child, stack, adj_children,
                            parent_placed, width, height)
                    prev_img = parent_placed
                elif layer.id not in needs_placed:
                    release_prev(prev_img)
                    prev_img = None

        release_prev(prev_img)
        for buf in borrowed:
            self._scratch.release(buf)
        return canvas

    def _blend_clipped_child(
        self, canvas: np.ndarray, child: Layer, stack: LayerStack,
        adj_children: dict, parent_placed: np.ndarray, width: int, height: int,
    ) -> None:
        """Composite *child* clipped to *parent_placed*'s alpha."""
        c_mask = self._get_effective_mask(child, stack)
        c_pix, c_pos = self._prepare_layer(child, adj_children)
        c_placed = self._place_planar(c_pix, c_pos, width, height)
        c_placed[3] *= parent_placed[3]
        c_placed_mask = (
            self._place_mask_combined(child, stack, width, height)
            if c_mask is not None else None
        )
        blend_planar_region(canvas, c_placed, (0, 0),
                            child.blend_mode, child.opacity, c_placed_mask)
        self._scratch.release(c_placed)

    # ------------------------------------------------------------------
    # Groups
    # ------------------------------------------------------------------

    def _composite_group(
        self, group: Layer, stack: LayerStack, w: int, h: int,
    ) -> np.ndarray:
        canvas = np.zeros((4, h, w), dtype=np.float32)
        mask_ids: set[str] = set()
        group_child_ids: set[str] = set()
        for layer in stack:
            if layer.parent_id == group.id:
                group_child_ids.add(layer.id)
                for mid in layer.mask_layers:
                    mask_ids.add(mid)

        adj_children: dict[str, list] = {}
        adj_child_ids: set[str] = set()
        for layer in stack:
            if (layer.parent_id
                    and layer.layer_type in (LayerType.ADJUSTMENT, LayerType.FILTER)
                    and layer.visible
                    and layer.parent_id in group_child_ids):
                adj_children.setdefault(layer.parent_id, []).append(layer)
                adj_child_ids.add(layer.id)

        regular_children: dict[str, list[Layer]] = {}
        for layer in stack:
            if (layer.parent_id and layer.visible
                    and layer.parent_id in group_child_ids
                    and layer.parent_id != group.id
                    and layer.layer_type not in (
                        LayerType.ADJUSTMENT, LayerType.FILTER, LayerType.MASK)
                    and layer.id not in mask_ids
                    and layer.id not in adj_child_ids):
                regular_children.setdefault(layer.parent_id, []).append(layer)

        for layer in stack:
            if layer.parent_id != group.id or not layer.visible:
                continue
            if layer.id in mask_ids or layer.layer_type == LayerType.MASK:
                continue
            if layer.id in adj_child_ids:
                continue
            if layer.layer_type in (LayerType.ADJUSTMENT, LayerType.FILTER):
                continue

            mask = self._get_effective_mask(layer, stack)
            pixels, blend_pos = self._prepare_layer(layer, adj_children)

            _has_clip_child = any(
                rc.clips_parent for rc in regular_children.get(layer.id, ()))

            if _has_clip_child:
                parent_placed = self._place_planar(pixels, blend_pos, w, h)
                for child in regular_children[layer.id]:
                    if not child.clips_parent:
                        continue
                    c_pix, c_pos = self._prepare_layer(child, adj_children)
                    c_placed = self._place_planar(c_pix, c_pos, w, h)
                    parent_placed[3] *= c_placed[3]
                    self._scratch.release(c_placed)
                placed_mask = (
                    self._place_mask_combined(layer, stack, w, h)
                    if mask is not None else None
                )
                blend_planar_region(canvas, parent_placed, (0, 0),
                                    layer.blend_mode, layer.opacity, placed_mask)
                for child in regular_children[layer.id]:
                    if child.clips_parent:
                        continue
                    self._blend_clipped_child(
                        canvas, child, stack, adj_children,
                        parent_placed, w, h)
                self._scratch.release(parent_placed)
            else:
                blend_planar_region(canvas, pixels, blend_pos,
                                    layer.blend_mode, layer.opacity, mask)
                if layer.id in regular_children:
                    parent_placed = self._place_planar(pixels, blend_pos, w, h)
                    for child in regular_children[layer.id]:
                        self._blend_clipped_child(
                            canvas, child, stack, adj_children,
                            parent_placed, w, h)
                    self._scratch.release(parent_placed)
        return canvas
