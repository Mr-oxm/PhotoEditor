"""Abstract base for all interactive tools."""

from abc import ABC, abstractmethod

import numpy as np

from ..core.document import Document


class Tool(ABC):
    """Base class for interactive canvas tools."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def activate(self) -> None:
        self._active = True

    def deactivate(self) -> None:
        self._active = False

    @abstractmethod
    def on_press(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        ...

    @abstractmethod
    def on_move(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        ...

    @abstractmethod
    def on_release(self, doc: Document, x: int, y: int) -> None:
        ...

    @staticmethod
    def _rasterize_if_needed(doc: Document) -> None:
        """If the active layer has a non-destructive transform, bake it.

        Destructive tools (brush, eraser, paint bucket, …) must call
        this before modifying pixel data so that the source is consistent.
        """
        layer = doc.layers.active_layer
        if layer is not None and layer._source_pixels is not None:
            layer.rasterize_transform()

    def _begin_stroke_mask(self, doc: Document) -> None:
        """Cache the layer-local selection mask for this stroke.

        Rebuilding it per event allocated and filled a whole layer-sized
        float32 array on every mouse-move -- 33 MB at 4K, 60-120 times a
        second, for a mask that cannot change mid-stroke. Measured cost of
        a brush move went from 1.12 ms to 0.04 ms with no selection; this
        closes the same gap when one is active.
        """
        self._stroke_sel_mask = self._compute_sel_mask(doc)
        self._stroke_sel_valid = True

    def _end_stroke_mask(self) -> None:
        self._stroke_sel_mask = None
        self._stroke_sel_valid = False

    def _get_sel_mask(self, doc: Document) -> np.ndarray | None:
        """Layer-local selection mask, cached for the duration of a stroke."""
        if getattr(self, "_stroke_sel_valid", False):
            return self._stroke_sel_mask
        return self._compute_sel_mask(doc)

    @staticmethod
    def _compute_sel_mask(doc: Document) -> np.ndarray | None:
        """Return a layer-local selection mask for the active layer, or None.

        The returned array has shape (lh, lw) float32 in [0,1] and is
        aligned to the layer's pixel grid.
        """
        if doc is None or not doc.selection.active:
            return None
        layer = doc.layers.active_layer
        if layer is None:
            return None
        mask = doc.selection._mask
        lx, ly = layer.position
        lh, lw = layer.pixels.shape[:2]
        dh, dw = mask.shape[:2]

        dy0 = max(0, ly)
        dy1 = min(dh, ly + lh)
        dx0 = max(0, lx)
        dx1 = min(dw, lx + lw)
        if dy1 <= dy0 or dx1 <= dx0:
            return np.zeros((lh, lw), dtype=np.float32)

        out = np.zeros((lh, lw), dtype=np.float32)
        sy0, sy1 = dy0 - ly, dy1 - ly
        sx0, sx1 = dx0 - lx, dx1 - lx
        out[sy0:sy1, sx0:sx1] = mask[dy0:dy1, dx0:dx1]
        return out

    def generate_preview_dab(self) -> np.ndarray | None:
        """Return an RGBA uint8 array showing what a single dab looks like.

        Override in subclasses to provide a real-time preview stamp.
        Returns ``None`` if this tool has no preview.
        """
        return None

    def _stamp_circle(
        self, target: np.ndarray, cx: int, cy: int,
        radius: int, color: np.ndarray, hardness: float = 1.0,
        opacity: float = 1.0, sel_mask: np.ndarray | None = None,
    ) -> None:
        """Stamp a circular brush dab onto *target* (region-optimised)."""
        h, w = target.shape[:2]
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        if y1 <= y0 or x1 <= x0:
            return
        # Use arange + broadcasting — half the memory of mgrid
        yy = np.arange(y0, y1, dtype=np.float32)[:, np.newaxis]
        xx = np.arange(x0, x1, dtype=np.float32)[np.newaxis, :]
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        r = max(radius, 1)
        mask = np.clip(1.0 - np.sqrt(dist_sq) / r, 0, 1)
        np.power(mask, 1.0 / max(hardness, 0.01), out=mask)
        mask *= opacity
        # Clip to selection
        if sel_mask is not None:
            mask *= sel_mask[y0:y1, x0:x1]
        mask = mask[..., np.newaxis]
        # In-place compositing on the target slice
        roi = target[y0:y1, x0:x1]
        inv_mask = 1.0 - mask
        np.multiply(roi, inv_mask, out=roi)
        roi += color * mask
        np.clip(roi, 0, 1, out=roi)

    def _stamp_tip(
        self, target: np.ndarray, cx: int, cy: int,
        tip: np.ndarray, tip_size: int,
        color: np.ndarray, opacity: float = 1.0,
        hardness: float = 1.0,
        sel_mask: np.ndarray | None = None,
    ) -> None:
        """Stamp a brush tip image onto *target*.

        *tip* is a grayscale H×W uint8 array where 255 = full opacity.
        It is scaled to *tip_size* pixels (longest dimension) before stamping.
        *hardness* (0-1) modulates the tip alpha via a power curve.
        """
        h, w = target.shape[:2]
        th, tw = tip.shape[:2]

        # Scale tip to requested size
        scale = tip_size / max(th, tw, 1)
        new_h = max(1, int(th * scale))
        new_w = max(1, int(tw * scale))

        if new_h != th or new_w != tw:
            import cv2
            scaled = cv2.resize(tip, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            scaled = tip

        # Place stamp centred at (cx, cy)
        half_h = new_h // 2
        half_w = new_w // 2
        y0 = cy - half_h
        x0 = cx - half_w

        # Clip to target bounds
        src_y0 = max(0, -y0)
        src_x0 = max(0, -x0)
        dst_y0 = max(0, y0)
        dst_x0 = max(0, x0)
        dst_y1 = min(h, y0 + new_h)
        dst_x1 = min(w, x0 + new_w)
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        src_x1 = src_x0 + (dst_x1 - dst_x0)

        if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
            return

        # Build alpha mask from tip slice
        tip_slice = scaled[src_y0:src_y1, src_x0:src_x1].astype(np.float32) / 255.0

        # Apply hardness: power curve makes soft edges more transparent
        if hardness < 0.99:
            np.power(tip_slice, 1.0 / max(hardness, 0.01), out=tip_slice)

        mask = tip_slice * opacity

        # Clip to selection
        if sel_mask is not None:
            mask *= sel_mask[dst_y0:dst_y1, dst_x0:dst_x1]

        mask = mask[..., np.newaxis]
        roi = target[dst_y0:dst_y1, dst_x0:dst_x1]
        inv_mask = 1.0 - mask
        np.multiply(roi, inv_mask, out=roi)
        roi += color * mask
        np.clip(roi, 0, 1, out=roi)

    def _erase_tip(
        self, target: np.ndarray, cx: int, cy: int,
        tip: np.ndarray, tip_size: int,
        opacity: float = 1.0,
        sel_mask: np.ndarray | None = None,
    ) -> None:
        """Erase (reduce alpha) using a brush tip image as the mask."""
        h, w = target.shape[:2]
        th, tw = tip.shape[:2]

        scale = tip_size / max(th, tw, 1)
        new_h = max(1, int(th * scale))
        new_w = max(1, int(tw * scale))

        if new_h != th or new_w != tw:
            import cv2
            scaled = cv2.resize(tip, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            scaled = tip

        half_h = new_h // 2
        half_w = new_w // 2
        y0 = cy - half_h
        x0 = cx - half_w

        src_y0 = max(0, -y0)
        src_x0 = max(0, -x0)
        dst_y0 = max(0, y0)
        dst_x0 = max(0, x0)
        dst_y1 = min(h, y0 + new_h)
        dst_x1 = min(w, x0 + new_w)
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        src_x1 = src_x0 + (dst_x1 - dst_x0)

        if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
            return

        tip_slice = scaled[src_y0:src_y1, src_x0:src_x1].astype(np.float32) / 255.0
        mask = tip_slice * opacity

        if sel_mask is not None:
            mask *= sel_mask[dst_y0:dst_y1, dst_x0:dst_x1]

        roi = target[dst_y0:dst_y1, dst_x0:dst_x1]
        roi *= (1.0 - mask[..., np.newaxis])
        np.clip(roi, 0, 1, out=roi)
