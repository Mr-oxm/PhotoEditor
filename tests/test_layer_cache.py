"""The layer raster cache must never serve stale pixels.

Caching prepared layer rasters is only safe if every mutation path
invalidates the entry. Painting tools write directly into
``layer.pixels[y0:y1, x0:x1]``, which no property setter observes, so they
must call ``Layer.touch()``. These tests pin that contract down -- a missed
touch() shows up here as a visibly stale render rather than as a bug report.
"""

from __future__ import annotations

import numpy as np
import pytest

from photo_editor.blending.planar import to_interleaved
from photo_editor.core.document import Document
from photo_editor.core.enums import ToolType
from photo_editor.engine.layer_cache import LayerRasterCache
from photo_editor.engine.render_pipeline import RenderPipeline


def _doc():
    doc = Document(64, 48, name="cache-test")
    doc.layers.active_index = 0
    return doc


def test_cache_populates_and_hits():
    doc = _doc()
    pipe = RenderPipeline()
    pipe.execute_planar(doc)
    assert pipe.layer_cache.stats()["entries"] >= 1

    # A targeted invalidation of a *different* layer must leave this
    # layer's entry intact, so the next composite hits the cache.
    pipe.layer_cache.reset_stats()
    pipe.invalidate("some-other-layer-id")
    pipe.execute_planar(doc)
    assert pipe.layer_cache.stats()["hits"] >= 1, pipe.layer_cache.stats()


def test_touch_invalidates_cached_raster():
    doc = _doc()
    pipe = RenderPipeline()
    layer = doc.layers.active_layer
    layer.pixels[:] = np.array([1, 0, 0, 1], dtype=np.float32)
    first = to_interleaved(pipe.execute_planar(doc)).copy()

    # Mutate in place exactly as a painting tool does, then touch().
    layer.pixels[10:20, 10:20] = np.array([0, 1, 0, 1], dtype=np.float32)
    layer.touch()
    pipe.invalidate(layer.id)
    second = to_interleaved(pipe.execute_planar(doc))

    assert not np.allclose(first, second), "cache served stale pixels"
    assert np.allclose(second[15, 15], [0, 1, 0, 1], atol=1e-5)


def test_content_version_advances_on_inplace_touch():
    doc = _doc()
    layer = doc.layers.active_layer
    before = layer.content_version
    layer.pixels[0:2, 0:2] = 0.5
    assert layer.content_version == before, "in-place write is invisible by design"
    layer.touch()
    assert layer.content_version == before + 1


def test_content_version_advances_on_setter():
    doc = _doc()
    layer = doc.layers.active_layer
    before = layer.content_version
    layer.pixels = np.zeros((48, 64, 4), dtype=np.float32)
    assert layer.content_version > before


def test_cache_key_tracks_position_and_channels():
    doc = _doc()
    pipe = RenderPipeline()
    layer = doc.layers.active_layer
    layer.pixels[:] = 1.0
    pipe.execute_planar(doc)

    layer.position = (5, 5)
    pipe.invalidate(layer.id)
    moved = to_interleaved(pipe.execute_planar(doc)).copy()

    layer.channel_r = False
    pipe.invalidate(layer.id)
    no_red = to_interleaved(pipe.execute_planar(doc))
    assert not np.allclose(moved, no_red), "channel toggle not reflected"


def test_cache_respects_byte_budget():
    cache = LayerRasterCache(budget_bytes=1 << 16)  # 64 KiB

    class FakeLayer:
        def __init__(self, i):
            self.id = f"L{i}"
            self._pixels = np.zeros((32, 32, 4), dtype=np.float32)
            self.styles = []
            self.position = (0, 0)
            self.channel_r = self.channel_g = self.channel_b = True
            self.channel_a = True
            self.content_version = 0

    big = np.zeros((4, 64, 64), dtype=np.float32)  # 64 KiB each
    for i in range(8):
        cache.put_prepared(FakeLayer(i), None, (big.copy(), (0, 0)))
    assert cache.stats()["bytes"] <= (1 << 16), cache.stats()


def test_cache_clear_and_invalidate():
    cache = LayerRasterCache()

    class FakeLayer:
        id = "X"
        _pixels = np.zeros((4, 4, 4), dtype=np.float32)
        styles = []
        position = (0, 0)
        channel_r = channel_g = channel_b = channel_a = True
        content_version = 0

    layer = FakeLayer()
    cache.put_prepared(layer, None, (np.zeros((4, 4, 4), np.float32), (0, 0)))
    assert cache.get_prepared(layer, None) is not None
    cache.invalidate("X")
    assert cache.get_prepared(layer, None) is None
    assert cache.stats()["bytes"] == 0


@pytest.mark.parametrize("tool_name,tool_cls_path", [
    ("brush", "photo_editor.tools.brush:BrushTool"),
    ("eraser", "photo_editor.tools.eraser:EraserTool"),
])
def test_painting_tools_touch_the_layer(tool_name, tool_cls_path):
    """A stroke must advance content_version so caches drop the layer."""
    module, cls_name = tool_cls_path.split(":")
    import importlib
    tool = getattr(importlib.import_module(module), cls_name)()
    doc = _doc()
    layer = doc.layers.active_layer
    layer.pixels[:] = 1.0
    before = layer.content_version
    tool.on_press(doc, 10, 10, 1.0)
    tool.on_move(doc, 20, 20, 1.0)
    tool.on_release(doc, 20, 20)
    assert layer.content_version > before, (
        f"{tool_name} stroke did not call Layer.touch(); "
        "the render cache would serve stale pixels"
    )
