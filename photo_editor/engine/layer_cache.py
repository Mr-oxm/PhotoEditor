"""Cache of per-layer prepared raster data, at multiple resolutions.

Preparing a layer for compositing means applying its styles, channel
toggles and scoped adjustment/filter children, then converting the result
to planar layout. None of that depends on the rest of the stack, so it can
be cached and reused across frames -- which matters because the old
compositor redid all of it for every layer on every frame.

Cache validity
--------------
An entry is keyed by the layer id and validated against a *content key*:

* ``layer.content_version`` -- bumped by :class:`~photo_editor.core.layer.Layer`
  whenever pixel or mask data is replaced, and by ``Layer.touch()`` for the
  in-place writes that painting tools perform.
* the identity of the underlying pixel buffer, so a swapped array is caught
  even if a version bump was missed.
* a fingerprint of the styles and of the scoped adjustment parameters, so
  dragging an adjustment slider invalidates the layer it applies to.

The cache is bounded by total bytes and evicts least-recently-used entries.

Mip levels
----------
Entries are keyed by ``(layer id, level)`` where level *L* means a scale of
``1 / 2**L``. Compositing a 4K document for a 1600x1000 viewport only needs
level 1 (1920x1080), which is 31.6 MB per layer instead of 126.6 MB -- the
difference between twenty layers fitting in a 1 GB cache and thrashing it
five times over. Level 0 is the full-resolution path used for export.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock

import numpy as np


def _params_fingerprint(adj_layers) -> tuple:
    """Hashable fingerprint of scoped adjustment/filter layers."""
    if not adj_layers:
        return ()
    out = []
    for layer in adj_layers:
        params = layer.adjustment_params or {}
        try:
            items = tuple(sorted(
                (k, _hashable(v)) for k, v in params.items()
            ))
        except TypeError:
            # Unhashable parameter -- fall back to never caching this layer.
            return None
        out.append((layer.id, layer.visible, items))
    return tuple(out)


def _hashable(value):
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


def _styles_fingerprint(styles) -> tuple:
    if not styles:
        return ()
    out = []
    for style in styles:
        params = getattr(style, "params", None)
        if params is None:
            return None
        try:
            out.append((type(style).__name__, _hashable(vars(params))))
        except TypeError:
            return None
    return tuple(out)


class LayerRasterCache:
    """LRU cache of prepared planar layer rasters, bounded by bytes."""

    def __init__(self, budget_bytes: int = 512 << 20) -> None:
        self._budget = budget_bytes
        self._entries: OrderedDict[str, tuple] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        # Band-parallel rendering reads this cache from several worker
        # threads at once, and LRU bookkeeping mutates the ordering.
        self._lock = Lock()

    # ---- Keying ------------------------------------------------------------

    @staticmethod
    def level_scale(level: int) -> float:
        return 1.0 / (1 << level)

    @staticmethod
    def choose_level(width: int, height: int, max_size: int,
                     max_level: int = 5) -> int:
        """Smallest mip level whose longest side fits within *max_size*.

        Returns 0 (full resolution) when *max_size* is 0 or already large
        enough, so the export path is unaffected.
        """
        if max_size <= 0:
            return 0
        longest = max(width, height)
        level = 0
        while level < max_level and (longest >> level) > max_size:
            level += 1
        return level

    @staticmethod
    def _content_key(layer, adj_layers):
        styles = _styles_fingerprint(layer.styles)
        if styles is None:
            return None
        params = _params_fingerprint(adj_layers)
        if params is None:
            return None
        try:
            buf_id = layer._pixels.__array_interface__["data"][0]
        except (AttributeError, TypeError):
            buf_id = 0
        return (
            getattr(layer, "content_version", 0),
            buf_id,
            layer.position,
            (layer.channel_r, layer.channel_g, layer.channel_b, layer.channel_a),
            styles,
            params,
        )

    # ---- Access ------------------------------------------------------------

    def get_prepared(self, layer, adj_layers, level: int = 0):
        """Return the cached (planar, blend_pos) at *level*, or None."""
        key = self._content_key(layer, adj_layers)
        if key is None:
            return None
        slot = (layer.id, level)
        with self._lock:
            entry = self._entries.get(slot)
            if entry is None or entry[0] != key:
                self._misses += 1
                return None
            self._entries.move_to_end(slot)
            self._hits += 1
            return entry[1]

    def put_prepared(self, layer, adj_layers, value, level: int = 0) -> None:
        key = self._content_key(layer, adj_layers)
        if key is None:
            return
        planar = value[0]
        size = planar.nbytes
        if size > self._budget:
            return  # A single layer larger than the whole budget: don't cache.
        slot = (layer.id, level)
        with self._lock:
            old = self._entries.pop(slot, None)
            if old is not None:
                self._bytes -= old[1][0].nbytes
            self._entries[slot] = (key, value)
            self._bytes += size
            self._evict()

    def _evict(self) -> None:
        while self._bytes > self._budget and self._entries:
            _, entry = self._entries.popitem(last=False)
            self._bytes -= entry[1][0].nbytes

    # ---- Invalidation ------------------------------------------------------

    def invalidate(self, layer_id: str) -> None:
        """Drop every mip level cached for *layer_id*."""
        with self._lock:
            for slot in [k for k in self._entries if k[0] == layer_id]:
                entry = self._entries.pop(slot)
                self._bytes -= entry[1][0].nbytes

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    # ---- Introspection -----------------------------------------------------

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._entries),
            "bytes": self._bytes,
            "mb": round(self._bytes / (1 << 20), 1),
            "budget_mb": round(self._budget / (1 << 20), 1),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
