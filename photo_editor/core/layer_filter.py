"""Filtering the layer stack by name and kind.

A twenty-layer project is the point at which scrolling the layers panel
stops being a reasonable way to find anything, and that is exactly the
workload this editor is built for. Filtering narrows the panel to the
layers you care about without changing the document.

Two rules make the result readable rather than a flat list of hits:

* A match pulls its **ancestors** in with it, so a layer buried three
  groups deep still appears in its proper place rather than floating
  contextless at the root.
* A matching **group** pulls its whole subtree in, because "show me the
  Sky group" means the group and what is in it.

The module is pure -- no Qt, no document mutation -- so the matching rules
can be tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .enums import LayerType

# Kinds a user actually thinks in, mapped to the LayerTypes behind them.
KIND_GROUPS: dict[str, tuple[LayerType, ...]] = {
    "raster": (LayerType.RASTER, LayerType.SMART_OBJECT),
    "text": (LayerType.TEXT,),
    "shape": (LayerType.SHAPE,),
    "adjustment": (LayerType.ADJUSTMENT,),
    "filter": (LayerType.FILTER,),
    "group": (LayerType.GROUP,),
    "mask": (LayerType.MASK,),
}


@dataclass
class LayerFilter:
    """A layers-panel query. An empty filter matches everything."""

    text: str = ""
    kinds: set[str] = field(default_factory=set)
    visible_only: bool = False
    locked_only: bool = False
    with_effects_only: bool = False

    @property
    def is_active(self) -> bool:
        return bool(self.text.strip() or self.kinds or self.visible_only
                    or self.locked_only or self.with_effects_only)

    def clear(self) -> None:
        self.text = ""
        self.kinds.clear()
        self.visible_only = False
        self.locked_only = False
        self.with_effects_only = False

    # ---- Matching ----------------------------------------------------------

    def matches(self, layer) -> bool:
        """True when *layer* satisfies every active criterion."""
        needle = self.text.strip().lower()
        if needle and needle not in (layer.name or "").lower():
            return False
        if self.kinds:
            allowed: set[LayerType] = set()
            for kind in self.kinds:
                allowed.update(KIND_GROUPS.get(kind, ()))
            if layer.layer_type not in allowed:
                return False
        if self.visible_only and not layer.visible:
            return False
        if self.locked_only and not layer.locked:
            return False
        if self.with_effects_only and not _has_effects(layer):
            return False
        return True


def _has_effects(layer) -> bool:
    return bool(getattr(layer, "styles", None)
                or getattr(layer, "mask_layers", None)
                or layer.mask is not None)


def visible_layer_ids(document, layer_filter: LayerFilter | None) -> set[str] | None:
    """IDs the layers panel should show, or None when nothing is filtered.

    Returning None rather than "every id" lets the panel skip the filter
    path entirely in the common case.
    """
    if layer_filter is None or not layer_filter.is_active:
        return None

    layers = list(document.layers)
    by_id = {l.id: l for l in layers}
    children_of: dict[str, list] = {}
    for layer in layers:
        if layer.parent_id:
            children_of.setdefault(layer.parent_id, []).append(layer)

    keep: set[str] = set()

    def add_subtree(layer) -> None:
        if layer.id in keep:
            return
        keep.add(layer.id)
        for child in children_of.get(layer.id, ()):
            add_subtree(child)

    def add_ancestors(layer) -> None:
        parent_id = layer.parent_id
        seen = set()
        while parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent = by_id.get(parent_id)
            if parent is None:
                break
            keep.add(parent.id)
            parent_id = parent.parent_id

    for layer in layers:
        if not layer_filter.matches(layer):
            continue
        if layer.layer_type == LayerType.GROUP:
            add_subtree(layer)
        else:
            keep.add(layer.id)
        add_ancestors(layer)

    return keep


def match_count(document, layer_filter: LayerFilter | None) -> int:
    """How many layers match directly, ignoring ancestors pulled in for context."""
    if layer_filter is None or not layer_filter.is_active:
        return len(list(document.layers))
    return sum(1 for l in document.layers if layer_filter.matches(l))
