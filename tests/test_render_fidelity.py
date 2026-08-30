"""Render-fidelity gate.

Every reference scene must composite to within one 8-bit level of the
golden captured from the pre-overhaul renderer. This is the hard gate that
makes rewriting the compositing core safe: performance work may change how
pixels are produced, never which pixels are produced.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.fidelity.golden import compare, render_scene
from tests.fidelity.scenes import all_scenes

SCENES = sorted(all_scenes().items())


@pytest.mark.parametrize("name,factory", SCENES, ids=[n for n, _ in SCENES])
def test_scene_matches_golden(name, factory):
    doc = factory()
    actual = render_scene(doc)
    ok, detail = compare(name, actual)
    assert ok, f"{name}: {detail}"


def test_output_contract():
    """The compositor's output contract: float32 RGBA in [0, 1]."""
    doc = all_scenes()["many_layers"]()
    out = render_scene(doc)
    assert out.dtype == np.float32
    assert out.ndim == 3 and out.shape[2] == 4
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_determinism():
    """Compositing the same document twice gives identical bytes."""
    factory = all_scenes()["clipping_chain"]
    a = render_scene(factory())
    b = render_scene(factory())
    np.testing.assert_array_equal(a, b)
