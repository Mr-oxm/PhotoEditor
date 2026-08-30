"""Paint Bucket tool — flood-fills a contiguous region with a solid colour."""

import cv2
import numpy as np

from .tool_base import Tool
from ..core.document import Document


class PaintBucketTool(Tool):
    """Fills connected pixels that match the target colour within a tolerance."""

    def __init__(self) -> None:
        super().__init__("Paint Bucket")
        self.color: np.ndarray = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.tolerance: int = 32  # 0–255 scale
        self.opacity: float = 1.0
        self.contiguous: bool = True

    # ------------------------------------------------------------------
    # Flood-fill
    # ------------------------------------------------------------------

    @staticmethod
    def _flood_fill_mask(pixels: np.ndarray, sx: int, sy: int,
                         tolerance: float) -> np.ndarray:
        """Return a boolean mask of connected pixels similar to the seed colour.

        Uses cv2.floodFill on a 3-channel (RGB) image for reliable results.
        """
        h, w = pixels.shape[:2]
        # Work on RGB only — cv2.floodFill is most reliable on 1- or 3-channel
        rgb = pixels[..., :3]
        rgb_u8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        # cv2.floodFill needs a mask 2px larger than the image
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        lo_diff = (int(tolerance),) * 3
        hi_diff = (int(tolerance),) * 3
        cv2.floodFill(rgb_u8, ff_mask, (sx, sy), 255,
                      loDiff=lo_diff, upDiff=hi_diff,
                      flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        # Extract the inner mask (strip the 1px border)
        mask = ff_mask[1:-1, 1:-1].astype(np.bool_)
        return mask

    @staticmethod
    def _global_tolerance_mask(pixels: np.ndarray, sx: int, sy: int,
                               tolerance: float) -> np.ndarray:
        """Return a boolean mask of ALL pixels similar to the seed colour (non-contiguous)."""
        seed_color = pixels[sy, sx, :3]
        diff = np.abs(pixels[..., :3] - seed_color).max(axis=-1)
        return diff <= (tolerance / 255.0)

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def on_press(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        self._rasterize_if_needed(doc)
        layer = doc.layers.active_layer
        if layer is None or layer.locked:
            return
        # Convert document coords to layer-local pixel coords
        lx, ly = layer.position
        px, py = x - lx, y - ly
        h, w = layer.pixels.shape[:2]
        if px < 0 or px >= w or py < 0 or py >= h:
            return

        doc.save_snapshot("Paint Bucket Fill")

        if self.contiguous:
            fill_mask = self._flood_fill_mask(layer.pixels, px, py, self.tolerance)
        else:
            fill_mask = self._global_tolerance_mask(layer.pixels, px, py, self.tolerance)

        mask = fill_mask.astype(np.float32)
        # Clip to selection
        sel_mask = self._get_sel_mask(doc)
        if sel_mask is not None:
            mask *= sel_mask
        mask = mask[..., np.newaxis] * self.opacity
        layer.begin_write()
        layer.pixels[:] = layer.pixels * (1 - mask) + self.color * mask
        np.clip(layer.pixels, 0, 1, out=layer.pixels)

    def on_move(self, doc: Document, x: int, y: int, pressure: float = 1.0) -> None:
        pass  # No drag behaviour

    def on_release(self, doc: Document, x: int, y: int) -> None:
        pass
