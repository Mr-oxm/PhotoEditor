"""Sandwich caching: reuse the layers that are not being edited.

While one layer is being dragged, the other nineteen do not change. Compositing
them again on every frame is the largest remaining waste in the render path, so
their result is cached:

    frame = copy(under) -> blend(active) -> blend(over)

Two blends instead of twenty, whatever the layer count.

Validity
--------
The **under** half -- everything below the active layer, composited in order --
is always safe to cache: it is exactly the canvas state the active layer would
have been blended onto anyway.

The **over** half is only safe when the layers above form an *isolated* group.
Porter-Duff `over` is associative, so pre-compositing a run of NORMAL layers
into one buffer and blending that gives the same pixels as blending them one at
a time. A non-NORMAL layer breaks this, because its result depends on whatever
is beneath it -- and so do clipping masks (which reference the layer below),
root adjustment and filter layers (which consume the accumulated canvas), and
layers with clipped children. Those cases fall back to a full walk.

Both halves are keyed on a signature of the layers they cover, so any change to
any of them -- pixels, position, opacity, blend mode, visibility, order --
invalidates the cache without anyone having to remember to say so.
"""

from __future__ import annotations

import numpy as np

from ..core.enums import BlendMode, LayerType

# Layer kinds that never participate in a cacheable over-run.
_NON_ISOLATABLE = (LayerType.ADJUSTMENT, LayerType.FILTER, LayerType.MASK)


def layer_signature(layer) -> tuple:
    """Everything about a layer that can change what it contributes."""
    return (
        layer.id,
        getattr(layer, "content_version", 0),
        layer.position,
        layer.opacity,
        layer.blend_mode,
        layer.visible,
        layer.clipping_mask,
        layer.clips_parent,
        layer.parent_id,
        tuple(layer.mask_layers),
        len(layer.styles or ()),
    )


def run_signature(layers) -> tuple:
    return tuple(layer_signature(l) for l in layers)


def over_run_is_isolatable(layers, regular_children: dict) -> bool:
    """True when *layers* can be pre-composited into one buffer.

    Conservative on purpose: anything whose result depends on what lies
    beneath it disqualifies the whole run, because a wrong answer here is a
    silently mis-composited image rather than a slow one.
    """
    for layer in layers:
        if layer.layer_type in _NON_ISOLATABLE:
            return False
        if layer.blend_mode is not BlendMode.NORMAL:
            return False
        if layer.clipping_mask or layer.clips_parent:
            return False
        if layer.id in regular_children:
            return False
    return True


class SandwichCache:
    """Holds the composited layers above and below the one being edited."""

    def __init__(self) -> None:
        self._under: np.ndarray | None = None
        self._under_key: tuple | None = None
        self._over: np.ndarray | None = None
        self._over_key: tuple | None = None
        self.hits = 0
        self.misses = 0

    # ---- Under ------------------------------------------------------------

    def get_under(self, key: tuple) -> np.ndarray | None:
        if self._under is not None and self._under_key == key:
            self.hits += 1
            return self._under
        self.misses += 1
        return None

    def put_under(self, key: tuple, canvas: np.ndarray) -> None:
        self._under = canvas.copy()
        self._under_key = key

    # ---- Over -------------------------------------------------------------

    def get_over(self, key: tuple) -> np.ndarray | None:
        if self._over is not None and self._over_key == key:
            self.hits += 1
            return self._over
        self.misses += 1
        return None

    def put_over(self, key: tuple, canvas: np.ndarray) -> None:
        self._over = canvas.copy()
        self._over_key = key

    # ---- Lifecycle --------------------------------------------------------

    def clear(self) -> None:
        self._under = None
        self._under_key = None
        self._over = None
        self._over_key = None

    def nbytes(self) -> int:
        return sum(b.nbytes for b in (self._under, self._over) if b is not None)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "under": self._under is not None,
            "over": self._over is not None,
            "mb": round(self.nbytes() / (1 << 20), 1),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0
