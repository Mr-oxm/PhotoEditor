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

import math
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


def _round_half_up(value: float) -> int:
    """Round .5 away from zero -- translation-invariant, unlike round()."""
    return int(math.floor(value + 0.5)) if value >= 0 else -int(
        math.floor(-value + 0.5))


class PlanarCompositor:
    """Composites a LayerStack into a flat planar RGBA buffer."""

    def __init__(self, cache: "LayerRasterCache | None" = None,
                 sandwich: "SandwichCache | None" = None) -> None:
        self._scratch = PlanarScratch(max_per_shape=3)
        self._cache = cache
        # Reuses the composited layers above and below the one being edited.
        # Only set on the single-threaded path: band-parallel rendering
        # already splits the work, and a per-band sandwich would cache
        # twenty slivers instead of two frames.
        self._sandwich = sandwich
        self.focus_layer_id: str | None = None
        # Origin of the band currently being composited. Band-parallel
        # rendering composites horizontal slices of the document
        # independently; each slice is a normal composite whose canvas is
        # band-sized and whose layer positions are shifted by -origin.
        self._ox = 0
        self._oy = 0
        # Mip level for this composite. Level L renders at 1 / 2**L scale;
        # level 0 is full resolution and is what export uses.
        self._level = 0
        self._scale = 1.0
        # The whole frame's region in level coordinates, shared by every
        # band. Layer preparation is cropped to it, so a zoomed-in view of a
        # large document only converts the pixels it is going to show.
        self._frame_roi: tuple[int, int, int, int] | None = None

    def _pos(self, position: tuple[int, int]) -> tuple[int, int]:
        """Translate a document-space position into scaled band space.

        Uses round-half-up rather than Python's ``round``. Banker's
        rounding is not translation-invariant -- round(2.5) is 2 but
        round(3.5) is 4 -- so a layer at an odd position landed on a
        different output pixel depending on where the viewport crop began,
        which showed up as content shifting by a pixel while panning.
        Round-half-up satisfies f(a + n) == f(a) + n for integer n, so a
        layer lands in the same place whatever the crop.
        """
        if self._scale != 1.0:
            position = (_round_half_up(position[0] * self._scale),
                        _round_half_up(position[1] * self._scale))
        if self._ox == 0 and self._oy == 0:
            return position
        return (position[0] - self._ox, position[1] - self._oy)

    def _downscale(self, arr: np.ndarray) -> np.ndarray:
        """Downscale an interleaved image (or 2-D mask) to the active level.

        INTER_AREA is the right filter for minification -- it averages the
        source pixels covered by each destination pixel, which is what a mip
        level should be, and it avoids the aliasing a naive resize gives on
        detailed photographic content.
        """
        if self._scale == 1.0:
            return arr
        h, w = arr.shape[:2]
        nw = max(1, int(round(w * self._scale)))
        nh = max(1, int(round(h * self._scale)))
        if nw == w and nh == h:
            return arr
        try:
            import cv2
            return cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
        except ImportError:
            ys = (np.arange(nh) * (h / nh)).astype(np.int32)
            xs = (np.arange(nw) * (w / nw)).astype(np.int32)
            return arr[ys][:, xs]

    def _scaled_mask(self, mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None or self._scale == 1.0:
            return mask
        return self._downscale(mask)

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
        """Full-resolution combined mask, or None.

        Deliberately *not* scaled here: a mask has to be cropped by the same
        offset as its layer's pixels before being downscaled, or it slides
        out of registration whenever a viewport ROI crops the layer.
        """
        return MaskManager.get_combined_mask(layer, stack)

    def _prepare_mask(self, layer: Layer, stack: LayerStack,
                      crop: tuple[int, int],
                      shape: tuple[int, int]) -> np.ndarray | None:
        """Mask for *layer*, cropped and scaled to match its prepared pixels."""
        mask = MaskManager.get_combined_mask(layer, stack)
        if mask is None:
            return None
        cx, cy = crop
        if cx or cy:
            mask = mask[cy:, cx:]
        mask = self._downscale(mask)
        # Clamp to the prepared pixel block so blend_planar_region's overlap
        # maths cannot be truncated by a mask that is a pixel short.
        h, w = shape
        if mask.shape[0] < h or mask.shape[1] < w:
            padded = np.zeros((h, w), dtype=np.float32)
            mh = min(h, mask.shape[0])
            mw = min(w, mask.shape[1])
            padded[:mh, :mw] = mask[:mh, :mw]
            mask = padded
        return mask

    # ------------------------------------------------------------------
    # Per-layer pixel preparation
    # ------------------------------------------------------------------

    def _crop_to_frame(self, pixels: np.ndarray, blend_pos: tuple[int, int],
                       ):
        """Crop full-resolution *pixels* to the part inside the frame ROI.

        *blend_pos* is in full-resolution document coordinates; the frame
        ROI is in level coordinates, so it is scaled up to match. Returns
        None when the layer lies entirely outside the frame.
        """
        roi = self._frame_roi
        if roi is None:
            return pixels, blend_pos, (0, 0)
        f = 1 << self._level
        rx, ry, rw, rh = roi[0] * f, roi[1] * f, roi[2] * f, roi[3] * f
        px, py = blend_pos
        lh, lw = pixels.shape[:2]
        x0 = max(0, rx - px)
        y0 = max(0, ry - py)
        x1 = min(lw, rx + rw - px)
        y1 = min(lh, ry + rh - py)
        if x1 <= x0 or y1 <= y0:
            return None
        if x0 == 0 and y0 == 0 and x1 == lw and y1 == lh:
            return pixels, blend_pos, (0, 0)
        # Snap both edges to the level grid. INTER_AREA averages each
        # destination pixel over a box of source pixels, so a crop that does
        # not start and end on a box boundary shifts the whole downscale by
        # a fraction of a pixel -- visible as content jittering while
        # panning, and as a seam where two ROIs meet.
        x0 -= x0 % f
        y0 -= y0 % f
        if x1 % f:
            x1 = min(lw, x1 + (f - x1 % f))
        if y1 % f:
            y1 = min(lh, y1 + (f - y1 % f))
        return pixels[y0:y1, x0:x1], (px + x0, py + y0), (x0, y0)

    def _prepare_layer(
        self, layer: Layer, adj_children: dict[str, list[Layer]],
    ) -> tuple[np.ndarray, tuple[int, int]] | None:
        """Return (planar pixels, blend position) for a raster-like layer.

        Applies styles, channel toggles and scoped adjustment/filter
        children, then converts to planar. Cached when a cache is attached.
        """
        kids = adj_children.get(layer.id)
        slot = (self._level, self._frame_roi)
        if self._cache is not None:
            cached = self._cache.get_prepared(layer, kids, slot)
            if cached is not None:
                return cached if cached[0] is not None else None

        # Styles and filters are applied at full resolution and only then
        # downscaled. Scaling their *parameters* instead would be cheaper
        # but visibly wrong -- a drop shadow's offset and a blur's radius
        # do not survive naive scaling. The full-res work is cached, so it
        # happens once per edit rather than once per frame.
        pixels = layer.pixels
        if layer.styles:
            pixels = StyleEngine.apply_styles(pixels, layer.styles)
        pixels = self._apply_channels(pixels, layer)
        blend_pos = layer.position
        if kids:
            pixels, pad = self._apply_filters_padded(pixels, kids)
            if pad > 0:
                blend_pos = (layer.position[0] - pad, layer.position[1] - pad)

        cropped = self._crop_to_frame(pixels, blend_pos)
        if cropped is None:
            # Entirely outside the frame -- cache the miss so the whole
            # preparation is skipped on subsequent bands and frames.
            if self._cache is not None:
                self._cache.put_prepared(
                    layer, kids, (None, blend_pos, (0, 0)), slot, nbytes=0)
            return None
        pixels, blend_pos, crop = cropped

        planar = to_planar(self._downscale(pixels))
        result = (planar, blend_pos, crop)
        if self._cache is not None:
            self._cache.put_prepared(layer, kids, result, slot)
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
                             cw: int, ch: int,
                             crop: tuple[int, int] = (0, 0)) -> np.ndarray | None:
        combined = self._get_effective_mask(layer, stack)
        if combined is None:
            return None
        cx, cy = crop
        if cx or cy:
            combined = combined[cy:, cx:]
        combined = self._downscale(combined)
        px, py = layer.position
        return self._place_mask_array(
            combined, self._pos((px + cx, py + cy)), cw, ch)

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

    def composite(self, stack: LayerStack, width: int, height: int,
                  origin: tuple[int, int] = (0, 0),
                  out: np.ndarray | None = None,
                  level: int = 0,
                  frame_roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """Composite *stack* into a planar (4, height, width) float32 buffer.

        *width* and *height* are the canvas size in *output* pixels. When
        *level* is non-zero that is the scaled size, not the document size.

        *origin* shifts the composited window within the output, so a
        caller can render a horizontal band (or any sub-rectangle) by
        passing its top-left corner and a matching canvas size. *out*, when
        given, is written in place instead of allocating.
        """
        self._ox, self._oy = origin
        self._level = level
        self._scale = 1.0 / (1 << level)
        self._frame_roi = frame_roi
        # Root-level adjustment layers rebind `canvas` to the adjusted
        # result, so the buffer we finish with may not be the one we were
        # handed. Remember the caller's buffer and copy back at the end.
        out_buf = out
        if out is not None:
            canvas = out
            canvas.fill(0)
        else:
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

        # --- Sandwich caching -------------------------------------------
        # Split the draw list around the layer being edited so the layers
        # that did not change can be reused. Falls back to a full walk
        # whenever the split is not safe.
        sandwich_plan = self._plan_sandwich(visible, regular_children)
        if sandwich_plan is not None:
            under, split_index, over_run, over_key = sandwich_plan
            if under is not None:
                np.copyto(canvas, under)
                visible = visible[split_index:]
        else:
            under = None
            split_index = 0
            over_run = []
            over_key = None

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
                gray = self._scaled_mask(layer.get_mask_grayscale())
                placed_gray = self._place_mask_array(
                    gray, self._pos(layer.position), width, height)
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
                group_mask = self._scaled_mask(
                    self._get_effective_mask(layer, stack))
                if group_mask is not None:
                    placed_mask = self._place_mask_array(
                        group_mask, self._pos(layer.position), width, height)
                    group_img[3] *= placed_mask
                blend_planar_region(canvas, group_img, (0, 0),
                                    layer.blend_mode, layer.opacity)
                release_prev(prev_img)
                prev_img = group_img
                continue

            prepared = self._prepare_layer(layer, adj_children)
            if prepared is None:
                release_prev(prev_img)
                prev_img = None
                continue
            pixels, blend_pos, crop = prepared
            mask = self._prepare_mask(
                layer, stack, crop, pixels.shape[1:])

            _has_clip_child = any(
                rc.clips_parent for rc in regular_children.get(layer.id, ()))

            if layer.clipping_mask and prev_img is not None:
                placed = self._place_planar(
                    pixels, self._pos(blend_pos), width, height)
                borrowed.append(placed)
                placed[3] *= prev_img[3]
                placed_mask = (
                    self._place_mask_combined(layer, stack, width, height, crop)
                    if mask is not None else None
                )
                blend_planar_region(canvas, placed, (0, 0),
                                    layer.blend_mode, layer.opacity, placed_mask)
                release_prev(prev_img)
                prev_img = placed

            elif _has_clip_child:
                parent_placed = self._place_planar(
                    pixels, self._pos(blend_pos), width, height)
                borrowed.append(parent_placed)
                for child in regular_children[layer.id]:
                    if not child.clips_parent:
                        continue
                    child_prep = self._prepare_layer(child, adj_children)
                    if child_prep is None:
                        continue
                    c_pix, c_pos, _ = child_prep
                    c_placed = self._place_planar(
                        c_pix, self._pos(c_pos), width, height)
                    parent_placed[3] *= c_placed[3]
                    self._scratch.release(c_placed)
                placed_mask = (
                    self._place_mask_combined(layer, stack, width, height, crop)
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
                blend_planar_region(canvas, pixels, self._pos(blend_pos),
                                    layer.blend_mode, layer.opacity, mask)
                release_prev(prev_img)
                if layer.id in needs_placed or layer.id in regular_children:
                    prev_img = self._place_planar(
                        pixels, self._pos(blend_pos), width, height)
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
        if out_buf is not None and canvas is not out_buf:
            np.copyto(out_buf, canvas)
            canvas = out_buf
        return canvas

    # ------------------------------------------------------------------
    # Sandwich planning
    # ------------------------------------------------------------------

    def _sandwich_key(self, run) -> tuple:
        """Cache key: what the run contains, and how it is being rendered."""
        from .sandwich_cache import run_signature
        return (run_signature(run), self._level, self._frame_roi,
                self._ox, self._oy)

    def _plan_sandwich(self, visible, regular_children):
        """Decide whether the layers below the focus can be reused.

        Returns ``(under_canvas_or_None, split_index, over_run, over_key)``
        or ``None`` when caching does not apply to this composite.
        """
        cache = self._sandwich
        focus = self.focus_layer_id
        if cache is None or focus is None or len(visible) < 3:
            return None
        index = next((i for i, l in enumerate(visible) if l.id == focus), -1)
        # A focus at the very bottom has nothing below it worth caching.
        if index <= 0:
            return None

        below = visible[:index]
        # Any root adjustment or filter below consumes the accumulated
        # canvas, which is still fine -- it is part of what we cache. But a
        # clipping-mask chain crossing the split is not: the focus layer
        # would need the placed pixels of the layer beneath it, which the
        # cached buffer no longer carries separately.
        if visible[index].clipping_mask:
            return None

        key = self._sandwich_key(below)
        under = cache.get_under(key)
        if under is None:
            return (None, 0, [], None)      # nothing cached yet this frame
        return (under, index, [], None)

    def prime_sandwich(self, stack: LayerStack, width: int, height: int,
                       focus_layer_id: str, level: int = 0,
                       origin: tuple[int, int] = (0, 0),
                       frame_roi=None) -> bool:
        """Composite and cache everything below *focus_layer_id*.

        Called once when an interaction begins. Returns True when a usable
        under-cache was produced.
        """
        cache = self._sandwich
        if cache is None:
            return False
        self._ox, self._oy = origin
        self._level = level
        self._scale = 1.0 / (1 << level)
        self._frame_roi = frame_roi

        visible = self._root_draw_list(stack)
        index = next((i for i, l in enumerate(visible)
                      if l.id == focus_layer_id), -1)
        if index <= 0 or len(visible) < 3 or visible[index].clipping_mask:
            cache.clear()
            return False

        below = visible[:index]
        key = self._sandwich_key(below)
        if cache.get_under(key) is not None:
            return True

        # Composite just the layers below, through the ordinary path.
        saved_focus = self.focus_layer_id
        self.focus_layer_id = None
        try:
            partial = self._composite_subset(stack, below, width, height,
                                             level=level, origin=origin,
                                             frame_roi=frame_roi)
        finally:
            self.focus_layer_id = saved_focus
        cache.put_under(key, partial)
        return True

    def _root_draw_list(self, stack: LayerStack) -> list[Layer]:
        """The root-level layers that composite(), in order, would draw."""
        layers = list(stack)
        mask_layer_ids = {mid for l in layers for mid in l.mask_layers}
        adj_child_ids = {
            l.id for l in layers
            if l.parent_id
            and l.layer_type in (LayerType.ADJUSTMENT, LayerType.FILTER)
            and l.visible
        }
        standalone_mask_ids = {
            l.id for l in layers
            if l.layer_type == LayerType.MASK and l.parent_id is None
            and l.id not in mask_layer_ids
        }
        return [
            l for l in layers
            if l.visible and l.parent_id is None
            and l.id not in mask_layer_ids
            and (l.layer_type != LayerType.MASK or l.id in standalone_mask_ids)
            and l.id not in adj_child_ids
        ]

    def _composite_subset(self, stack: LayerStack, subset: list[Layer],
                          width: int, height: int, level: int,
                          origin, frame_roi) -> np.ndarray:
        """Composite only *subset* of the root layers, in order.

        Implemented by hiding the other root layers for the duration, so
        the one compositing routine stays the single source of truth for
        every clipping, grouping and masking rule.
        """
        keep = {l.id for l in subset}
        hidden = [l for l in self._root_draw_list(stack) if l.id not in keep]
        saved = [(l, l.visible) for l in hidden]
        for layer, _ in saved:
            layer.visible = False
        try:
            return self.composite(stack, width, height, origin=origin,
                                  level=level, frame_roi=frame_roi).copy()
        finally:
            for layer, was_visible in saved:
                layer.visible = was_visible

    def _blend_clipped_child(
        self, canvas: np.ndarray, child: Layer, stack: LayerStack,
        adj_children: dict, parent_placed: np.ndarray, width: int, height: int,
    ) -> None:
        """Composite *child* clipped to *parent_placed*'s alpha."""
        prepared = self._prepare_layer(child, adj_children)
        if prepared is None:
            return          # entirely outside the frame ROI
        c_pix, c_pos, c_crop = prepared
        c_mask = self._get_effective_mask(child, stack)
        c_placed = self._place_planar(c_pix, self._pos(c_pos), width, height)
        c_placed[3] *= parent_placed[3]
        c_placed_mask = (
            self._place_mask_combined(child, stack, width, height, c_crop)
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
        # Runs inside the caller's band: self._ox/_oy are already set, and
        # every placement below goes through _pos().
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

            prepared = self._prepare_layer(layer, adj_children)
            if prepared is None:
                continue
            pixels, blend_pos, crop = prepared
            mask = self._prepare_mask(layer, stack, crop, pixels.shape[1:])

            _has_clip_child = any(
                rc.clips_parent for rc in regular_children.get(layer.id, ()))

            if _has_clip_child:
                parent_placed = self._place_planar(
                    pixels, self._pos(blend_pos), w, h)
                for child in regular_children[layer.id]:
                    if not child.clips_parent:
                        continue
                    child_prep = self._prepare_layer(child, adj_children)
                    if child_prep is None:
                        continue
                    c_pix, c_pos, _ = child_prep
                    c_placed = self._place_planar(
                        c_pix, self._pos(c_pos), w, h)
                    parent_placed[3] *= c_placed[3]
                    self._scratch.release(c_placed)
                placed_mask = (
                    self._place_mask_combined(layer, stack, w, h, crop)
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
                blend_planar_region(canvas, pixels, self._pos(blend_pos),
                                    layer.blend_mode, layer.opacity, mask)
                if layer.id in regular_children:
                    parent_placed = self._place_planar(
                        pixels, self._pos(blend_pos), w, h)
                    for child in regular_children[layer.id]:
                        self._blend_clipped_child(
                            canvas, child, stack, adj_children,
                            parent_placed, w, h)
                    self._scratch.release(parent_placed)
        return canvas
