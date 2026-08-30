"""Band-parallel compositing must be identical to whole-canvas compositing.

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
    par = ParallelCompositor(max_workers=4, band_height=16).composite(
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
    par = ParallelCompositor(max_workers=4, band_height=band_height).composite(
        doc.layers, w, h)
    np.testing.assert_allclose(par, serial, atol=1e-6)


@pytest.mark.parametrize("workers", [1, 2, 3, 8])
def test_worker_count_does_not_change_result(workers):
    doc = all_scenes()["clipping_chain"]()
    w, h = doc.width, doc.height
    serial = PlanarCompositor().composite(doc.layers, w, h)
    par = ParallelCompositor(max_workers=workers, band_height=8).composite(
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
    comp = ParallelCompositor(max_workers=4, band_height=16)
    first = comp.composite(doc.layers, doc.width, doc.height).copy()
    for _ in range(4):
        again = comp.composite(doc.layers, doc.width, doc.height)
        np.testing.assert_allclose(again, first, atol=1e-6)
    comp.shutdown()


def test_writes_into_caller_buffer():
    doc = all_scenes()["opacity"]()
    out = np.full((4, doc.height, doc.width), 7.0, dtype=np.float32)
    comp = ParallelCompositor(max_workers=2, band_height=16)
    result = comp.composite(doc.layers, doc.width, doc.height, out=out)
    assert result is out, "composite must write into the provided buffer"
    assert not np.allclose(out, 7.0), "buffer was not overwritten"
    comp.shutdown()
