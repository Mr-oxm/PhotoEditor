"""Document resize operations extracted from UI controllers."""

from __future__ import annotations

from collections.abc import Callable


def resize_canvas(document, width: int, height: int) -> None:
    """Resize the canvas while preserving layer pixel data and offsets."""
    document._snapshot("Resize Canvas")
    document.resize(width, height)


def resize_image(document, new_width: int, new_height: int, resize_fn: Callable) -> None:
    """Resize the full document and all layers proportionally."""
    scale_x = new_width / max(document.width, 1)
    scale_y = new_height / max(document.height, 1)
    document._snapshot("Resize Image")

    for layer in document.layers.layers:
        pixels = layer.pixels
        layer_height, layer_width = pixels.shape[:2]
        next_width = max(1, round(layer_width * scale_x))
        next_height = max(1, round(layer_height * scale_y))
        # Assign through the property so content_version moves: history
        # shares buffers by version, so a silent swap makes the next
        # snapshot store the pre-resize pixels.
        layer.pixels = resize_fn(pixels, (next_width, next_height))
        layer.width, layer.height = next_width, next_height
        offset_x, offset_y = layer.position
        layer.position = (round(offset_x * scale_x), round(offset_y * scale_y))

    document.resize(new_width, new_height)