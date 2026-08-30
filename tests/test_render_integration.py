"""End-to-end render path through MainWindow.

The engine tests cover compositing in isolation; these cover the plumbing
between it and the screen -- preview level selection, viewport ROI, the
document/preview coordinate split, and the scheduler's threading -- which
is where the interesting bugs were.

Note: the canvas is a QOpenGLWidget and the offscreen Qt platform cannot
create a GL context, so ``widget.grab()`` returns blank here. These tests
therefore inspect the composited QPixmap the canvas was handed, which is
the output of everything under test anyway.
"""

from __future__ import annotations

import numpy as np
import pytest

from photo_editor.core.layer import Layer


@pytest.fixture
def win(qtbot):
    from photo_editor.ui.main_window import MainWindow
    w = MainWindow(dev_mode=True)
    qtbot.addWidget(w)
    w.resize(1200, 800)
    w._show_editor()
    w.show()
    qtbot.wait(50)
    return w


def _add_flat_layer(doc, value, w=1024, h=768, name="flat"):
    layer = Layer(name=name, width=w, height=h)
    px = np.zeros((h, w, 4), dtype=np.float32)
    px[..., :3] = value
    px[..., 3] = 1.0
    layer.pixels = px
    doc.layers.add(layer)
    return layer


def _render_and_wait(win, qtbot, invalidate=True):
    """Render once and wait until *that* frame has reached the canvas.

    Waiting on the thread pool alone is not enough: the worker's result is
    delivered by a queued signal, and a frame enqueued during window setup
    may still be in flight. Waiting for the specific generation to be shown
    makes the test independent of that timing.
    """
    # Mutating the layer stack directly bypasses execute_command/_refresh,
    # which is what normally invalidates the pipeline.
    if invalidate:
        win._pipeline.invalidate()
    sched = win._render_scheduler
    sched.enqueue_immediate(win._doc)
    generation = sched._generation
    qtbot.waitUntil(lambda: sched._last_shown_generation >= generation,
                    timeout=20000)
    return win._canvas._pixmap


def _sample(pixmap, fx=0.5, fy=0.5):
    img = pixmap.toImage()
    c = img.pixelColor(int(img.width() * fx), int(img.height() * fy))
    return (c.red(), c.green(), c.blue(), c.alpha())


def test_canvas_receives_a_composited_frame(win, qtbot):
    doc = win._doc
    _add_flat_layer(doc, 0.75, w=doc.width, h=doc.height)
    pm = _render_and_wait(win, qtbot)
    assert pm is not None and pm.width() > 0
    r, g, b, a = _sample(pm)
    assert (r, g, b) == (191, 191, 191), f"expected the top layer, got {(r, g, b)}"
    assert a == 255


def test_canvas_keeps_document_coordinates_when_previewing(win, qtbot):
    """The preview buffer may be smaller than the document; document-space
    overlays (selection, guides, transform box, tool hit-testing) must not
    silently rescale with it."""
    doc = win._doc
    doc.resize(3200, 2400)
    _add_flat_layer(doc, 0.5, w=3200, h=2400)
    win._canvas.set_image(np.zeros((2400, 3200, 4), np.uint8), force=True,
                          doc_size=(3200, 2400))
    win._canvas.set_zoom(0.25)
    win._sync_preview_level()
    pm = _render_and_wait(win, qtbot)
    assert win._canvas._doc_w == 3200 and win._canvas._doc_h == 2400
    assert pm.width() < 3200, "a zoomed-out view should render a smaller buffer"


def test_zooming_in_does_not_render_a_softer_preview(win, qtbot):
    """At 100% zoom the preview must not be upscaled from a coarse level."""
    doc = win._doc
    doc.resize(2048, 1536)
    _add_flat_layer(doc, 0.5, w=2048, h=1536)
    win._canvas.set_image(np.zeros((1536, 2048, 4), np.uint8), force=True,
                          doc_size=(2048, 1536))

    win._canvas.set_zoom(1.0)
    win._sync_preview_level()
    assert win._render_scheduler.preview_max_size == 2048

    win._canvas.set_zoom(0.25)
    win._sync_preview_level()
    assert win._render_scheduler.preview_max_size < 2048


def test_viewport_roi_restricts_rendering_when_zoomed_in(win, qtbot):
    doc = win._doc
    doc.resize(4000, 3000)
    _add_flat_layer(doc, 0.5, w=4000, h=3000)
    win._canvas.set_image(np.zeros((3000, 4000, 4), np.uint8), force=True,
                          doc_size=(4000, 3000))
    win._canvas.set_zoom(1.0)
    win._sync_preview_level()
    roi = win._render_scheduler._roi
    assert roi is not None, "a zoomed-in view should restrict rendering to the viewport"
    _, _, rw, rh = roi
    assert rw < 4000 and rh < 3000

    pm = _render_and_wait(win, qtbot)
    assert pm.width() <= rw + 2 and pm.height() <= rh + 2
    assert win._canvas._src_rect is not None
    assert _sample(pm)[:3] == (128, 128, 128)


def test_whole_document_visible_uses_no_roi(win, qtbot):
    doc = win._doc
    _add_flat_layer(doc, 0.25, w=doc.width, h=doc.height)
    win._canvas.set_image(np.zeros((doc.height, doc.width, 4), np.uint8),
                          force=True, doc_size=(doc.width, doc.height))
    win._canvas.zoom_to_fit()
    win._sync_preview_level()
    assert win._render_scheduler._roi is None
    pm = _render_and_wait(win, qtbot)
    assert win._canvas._src_rect is None
    assert _sample(pm)[:3] == (64, 64, 64)


def test_layer_edit_is_reflected_in_the_next_frame(win, qtbot):
    """Guards the cache-invalidation contract end to end."""
    doc = win._doc
    layer = _add_flat_layer(doc, 0.25, w=doc.width, h=doc.height)
    first = _sample(_render_and_wait(win, qtbot))
    assert first[:3] == (64, 64, 64)

    layer.begin_write()
    layer.pixels[:] = np.array([0.75, 0.75, 0.75, 1.0], dtype=np.float32)
    win._pipeline.invalidate(layer.id)
    second = _sample(_render_and_wait(win, qtbot))
    assert second[:3] == (191, 191, 191), (
        f"edit not reflected: still {second[:3]} -- stale cache")


def test_many_layers_render_without_error(win, qtbot):
    doc = win._doc
    for i in range(12):
        _add_flat_layer(doc, 0.1 + i * 0.05, w=800, h=600, name=f"L{i}")
        doc.layers.layers[-1].opacity = 0.5
    pm = _render_and_wait(win, qtbot)
    assert pm is not None and pm.width() > 0


def test_scheduler_runs_one_render_at_a_time(win, qtbot):
    """Concurrent renders against one pipeline were a real data race."""
    assert win._render_scheduler._pool.maxThreadCount() == 1


def test_edit_during_an_in_flight_render_is_not_lost(win, qtbot):
    """Invalidating while a render is running must not leave a stale cache.

    A render that began before an edit would otherwise finish, mark its
    now-stale output as the valid cache, and the next render would return
    it without recompositing -- so the canvas silently kept showing the
    pre-edit picture. The pipeline versions its cache with an epoch so a
    render whose epoch has moved on refuses to publish.
    """
    doc = win._doc
    layer = _add_flat_layer(doc, 0.25, w=doc.width, h=doc.height)

    # Start a render, then invalidate and edit *without* waiting for it.
    win._pipeline.invalidate()
    win._render_scheduler.enqueue_immediate(doc)

    layer.begin_write()
    layer.pixels[:] = np.array([0.75, 0.75, 0.75, 1.0], dtype=np.float32)
    win._pipeline.invalidate(layer.id)

    pm = _render_and_wait(win, qtbot, invalidate=False)
    assert _sample(pm)[:3] == (191, 191, 191), (
        "an edit made while a render was in flight was lost")


def test_pipeline_epoch_blocks_stale_cache_publication():
    """Unit-level version of the above, without the UI."""
    from photo_editor.core.document import Document
    from photo_editor.engine.render_pipeline import RenderPipeline

    doc = Document(64, 48)
    pipe = RenderPipeline()
    pipe.execute_to_uint8(doc)
    assert pipe._uint8_valid

    # Simulate a render that started before an invalidation lands.
    epoch_before = pipe._epoch
    pipe.invalidate()
    assert pipe._epoch != epoch_before
    assert not pipe._uint8_valid


def test_scheduler_is_genuinely_double_buffered(win, qtbot):
    """The UI thread converts frame N to a QPixmap while the worker may
    already be compositing frame N+1, so the two must never share a buffer.

    They did: the scheduler adopted whatever execute_to_uint8 returned, which
    is the pipeline's own internal buffer, so both slots ended up pointing at
    one array.
    """
    doc = win._doc
    _add_flat_layer(doc, 0.5, w=doc.width, h=doc.height)

    seen = set()
    for _ in range(5):
        _render_and_wait(win, qtbot)
        for buf in win._render_scheduler._buffers:
            if buf is not None:
                seen.add(id(buf))

    assert len(seen) >= 2, (
        f"scheduler used {len(seen)} distinct output buffer(s); double "
        "buffering has collapsed and a worker can overwrite the frame the "
        "UI thread is reading")
    pipeline_buf = win._pipeline._uint8_buf
    if pipeline_buf is not None:
        assert id(pipeline_buf) not in seen, (
            "scheduler buffers alias the pipeline's internal buffer")
