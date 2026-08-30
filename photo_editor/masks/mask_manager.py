"""High-level mask management for layers and mask layers."""

import numpy as np

from ..core.layer import Layer
from ..core.document import Document
from ..core.enums import LayerType


class MaskManager:
    """Create, apply, disable, and refine layer masks.

    Supports both the legacy single-mask-per-layer system *and* the
    new Affinity-style mask-layer system where dedicated MASK layers
    are children of a parent layer.
    """

    # ---- Legacy single-mask API (unchanged) ----------------------------------

    @staticmethod
    def add_mask(layer: Layer, fill_white: bool = True) -> None:
        layer.add_mask(fill_white)

    @staticmethod
    def remove_mask(layer: Layer) -> None:
        layer.remove_mask()

    @staticmethod
    def apply_mask(layer: Layer) -> None:
        """Burn the mask into the alpha channel permanently."""
        if layer.mask is not None:
            layer.pixels[..., 3] *= layer.mask
            layer.touch()
            layer.remove_mask()

    @staticmethod
    def disable_mask(layer: Layer) -> None:
        layer.mask_enabled = False

    @staticmethod
    def enable_mask(layer: Layer) -> None:
        layer.mask_enabled = True

    @staticmethod
    def invert_mask(layer: Layer) -> None:
        if layer.mask is not None:
            layer.mask = 1.0 - layer.mask

    @staticmethod
    def fill_mask(layer: Layer, value: float = 1.0) -> None:
        if layer.mask is not None:
            layer.mask[:] = value

    @staticmethod
    def gradient_mask(layer: Layer, vertical: bool = True) -> None:
        """Fill the mask with a linear gradient."""
        h, w = layer.height, layer.width
        if vertical:
            grad = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, np.newaxis]
            grad = np.broadcast_to(grad, (h, w)).copy()
        else:
            grad = np.linspace(1.0, 0.0, w, dtype=np.float32)[np.newaxis, :]
            grad = np.broadcast_to(grad, (h, w)).copy()
        layer.mask = grad

    # ---- Mask-layer API (new Affinity-style) --------------------------------

    @staticmethod
    def add_mask_layer(doc: Document, target_id: str | None = None,
                       fill_white: bool = True, name: str | None = None) -> Layer | None:
        """Add a new mask layer to *doc*, attached to *target_id*."""
        return doc.add_mask_layer(target_id, fill_white=fill_white, name=name)

    @staticmethod
    def add_standalone_mask_layer(doc: Document, fill_white: bool = True,
                                  name: str | None = None) -> Layer | None:
        """Add a standalone mask layer not attached to any parent."""
        return doc.add_mask_layer("__standalone__", fill_white=fill_white, name=name)

    @staticmethod
    def remove_mask_layer(doc: Document, mask_layer_id: str) -> None:
        doc.remove_mask_layer(mask_layer_id)

    @staticmethod
    def selection_to_mask(doc: Document, target_id: str | None = None) -> Layer | None:
        """Convert the current selection to a mask layer."""
        return doc.selection_to_mask_layer(target_id)

    @staticmethod
    def convert_to_mask(doc: Document, layer_id: str,
                        target_id: str | None = None) -> Layer | None:
        """Convert an existing layer into a mask layer for *target_id*."""
        return doc.convert_layer_to_mask(layer_id, target_id)

    @staticmethod
    def apply_mask_layer(doc: Document, mask_layer_id: str) -> None:
        """Burn a mask layer into its parent's alpha and remove it."""
        doc.apply_mask_layer(mask_layer_id)

    @staticmethod
    def invert_mask_layer(layer: Layer) -> None:
        """Invert a MASK layer's pixel data (white ↔ black)."""
        if layer.layer_type == LayerType.MASK:
            layer._pixels[..., :3] = 1.0 - layer._pixels[..., :3]
            layer.touch()

    @staticmethod
    def fill_mask_layer(layer: Layer, value: float = 1.0) -> None:
        """Fill a MASK layer with a uniform gray value."""
        if layer.layer_type == LayerType.MASK:
            layer._pixels[..., :3] = value
            layer._pixels[..., 3] = 1.0
            layer.touch()

    @staticmethod
    def get_combined_mask(layer: Layer, stack) -> np.ndarray | None:
        """Compute the effective combined mask for *layer*.

        Combines the legacy single mask *and* all child mask layers
        into a single grayscale array sized to match *layer*'s pixels.
        Returns ``None`` when no masks exist.
        """
        import cv2

        target_h, target_w = layer.pixels.shape[:2]
        masks: list[np.ndarray] = []

        # Legacy single mask
        if layer.mask is not None and layer.mask_enabled:
            m = layer.mask
            if m.shape[:2] != (target_h, target_w):
                m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            masks.append(m)

        # Mask layers
        for mid in layer.mask_layers:
            mask_layer = stack.get(mid)
            if mask_layer is not None and mask_layer.visible:
                m = mask_layer.get_mask_grayscale()
                if m.shape[:2] != (target_h, target_w):
                    m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                masks.append(m)

        if not masks:
            return None
        combined = masks[0].copy()
        for m in masks[1:]:
            combined *= m  # multiply = intersect
        return np.clip(combined, 0.0, 1.0)
