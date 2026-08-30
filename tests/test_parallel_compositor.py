"""Band-parallel compositing must be identical to whole-canvas compositing.

Every compositor here passes ``min_parallel_pixels=0`` to FORCE the banded
path. Without it the reference scenes (160x120) fall below the production
threshold and each of these tests silently exercised the serial path
instead -- which is exactly how a compositor-sharing race between
concurrent bands survived a green suite.

Splitting the canvas into bands changes only which thread touches which
rows. If a band ever reads or writes outside its slice -- a placement that
forgot the origin shift, a scratch buffer shared between workers -- the
result diverges, usually as a seam at a band boundary. These tests compare
against the serial result exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from photo_editor.engine.parallel_compositor import ParallelCompositor
from photo_editor.engine.planar_compositor import PlanarCompositor
from tests.fidelity.scenes import all_scenes

SCENES = sorted(all_scenes().items())


@pytest.mark.parametrize("name,factory", SCENES, ids=[n for n, _ in SCENES])
def test_parallel_matches_serial(name, factory):
    doc = factory()
    w, h = doc.width, doc.height
    serial = PlanarCompositor().composite(doc.layers, w, h)
    par = ParallelCompositor(max_workers=4, band_height=16, min_parallel_pixels=0).composite(
        doc.layers, w, h)
    np.testing.assert_allclose(
        par, serial, atol=1e-6,
        err_msg=f"{name}: band-parallel result differs from serial",
    )


@pytest.mark.parametrize("band_height", [1, 7, 16, 64, 1000])
def test_band_height_does_not_change_result(band_height):
    """Any band height must give the same picture -- no seams, no gaps."""
    doc = all_scenes()["many_layers"]()
    w, h = doc.width, doc.height
    serial = PlanarCompositor().composite(doc.layers, w, h)
    par = ParallelCompositor(max_workers=4, band_height=band_height,
                                min_parallel_pixels=0).composite(
        doc.layers, w, h)
    np.testing.assert_allclose(par, serial, atol=1e-6)


@pytest.mark.parametrize("workers", [1, 2, 3, 8])
def test_worker_count_does_not_change_result(workers):
    doc = all_scenes()["clipping_chain"]()
    w, h = doc.width, doc.height
    serial = PlanarCompositor().composite(doc.layers, w, h)
    par = ParallelCompositor(max_workers=workers, band_height=8,
                                min_parallel_pixels=0).composite(
        doc.layers, w, h)
    np.testing.assert_allclose(par, serial, atol=1e-6)


def test_band_origin_renders_a_correct_sub_window():
    """A single band must equal the matching rows of the full composite."""
    doc = all_scenes()["group"]()
    w, h = doc.width, doc.height
    full = PlanarCompositor().composite(doc.layers, w, h)
    y0, y1 = 30, 70
    band = PlanarCompositor().composite(doc.layers, w, y1 - y0, origin=(0, y0))
    np.testing.assert_allclose(band, full[:, y0:y1, :], atol=1e-6)


def test_repeated_parallel_composites_are_stable():
    """Reused scratch pools across workers must not leak between frames."""
    doc = all_scenes()["nested_group"]()
    comp = ParallelCompositor(max_workers=4, band_height=16, min_parallel_pixels=0)
    first = comp.composite(doc.layers, doc.width, doc.height).copy()
    for _ in range(4):
        again = comp.composite(doc.layers, doc.width, doc.height)
        np.testing.assert_allclose(again, first, atol=1e-6)
    comp.shutdown()


def test_writes_into_caller_buffer():
    doc = all_scenes()["opacity"]()
    out = np.full((4, doc.height, doc.width), 7.0, dtype=np.float32)
    comp = ParallelCompositor(max_workers=2, band_height=16, min_parallel_pixels=0)
    result = comp.composite(doc.layers, doc.width, doc.height, out=out)
    assert result is out, "composite must write into the provided buffer"
    assert not np.allclose(out, 7.0), "buffer was not overwritten"
    comp.shutdown()


def test_band_parallel_actually_runs_on_multiple_threads():
    """Guard the guard: if these tests stop exercising real concurrency,
    every other assertion in this file becomes decorative."""
    import threading

    doc = all_scenes()["many_layers"]()
    comp = ParallelCompositor(max_workers=4, band_height=16,
                              min_parallel_pixels=0)
    seen = set()
    original = comp._thread_compositor

    def spy():
        seen.add(threading.current_thread().name)
        return original()

    comp._thread_compositor = spy
    try:
        comp.composite(doc.layers, doc.width, doc.height)
    finally:
        comp.shutdown()
    assert len(seen) > 1, (
        f"band-parallel compositing ran on one thread ({seen}); these tests "
        "are not exercising the concurrent path they claim to")


def test_more_bands_than_workers_do_not_share_a_compositor():
    """The race this file exists to catch: with more bands than workers,
    indexing compositors by band number handed one object to two concurrent
    bands, which raced on its origin, level and scratch pool."""
    import threading

    doc = all_scenes()["many_layers"]()
    serial = PlanarCompositor().composite(doc.layers, doc.width, doc.height)
    # 120 rows / 4-row bands = 30 bands across 4 workers.
    comp = ParallelCompositor(max_workers=4, band_height=4,
                              min_parallel_pixels=0)
    owners = {}
    original = comp._thread_compositor

    def spy():
        c = original()
        owners.setdefault(id(c), set()).add(threading.current_thread().name)
        return c

    comp._thread_compositor = spy
    try:
        out = comp.composite(doc.layers, doc.width, doc.height)
    finally:
        comp.shutdown()

    for cid, threads in owners.items():
        assert len(threads) == 1, (
            f"compositor {cid} was used by {threads} concurrently")
    np.testing.assert_allclose(out, serial, atol=1e-6)


def test_dissolve_pattern_is_independent_of_decomposition():
    """Dissolve's dither is a property of the document, not of the region.

    It is derived from absolute coordinates precisely so that a band, a
    viewport ROI and a whole-canvas render agree. When it was derived from
    the region's own size instead, band-parallel rendering produced a
    different pattern per band -- visible seams along band boundaries that
    moved as you panned.
    """
    from photo_editor.core.enums import BlendMode
    from tests.fidelity.scenes import scene_blend_mode

    doc = scene_blend_mode(BlendMode.DISSOLVE)
    w, h = doc.width, doc.height
    whole = PlanarCompositor().composite(doc.layers, w, h)

    for band_height in (4, 7, 16, 40):
        banded = ParallelCompositor(max_workers=4, band_height=band_height,
                                    min_parallel_pixels=0).composite(
            doc.layers, w, h)
        np.testing.assert_allclose(
            banded, whole, atol=1e-6,
            err_msg=f"dissolve differs at band height {band_height}")

    # A sub-window must agree with the same window of the full render.
    y0, y1 = 30, 70
    band = PlanarCompositor().composite(doc.layers, w, y1 - y0, origin=(0, y0))
    np.testing.assert_allclose(band, whole[:, y0:y1, :], atol=1e-6)
