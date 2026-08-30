"""PlanarCompositor must reproduce Compositor exactly, scene for scene.

This is the equivalence proof for the compositor rewrite: the two
implementations are run side by side over every reference scene, which
between them cover all 28 blend modes, opacity, legacy masks, mask layers,
standalone and detached masks, clipping chains, clips_parent children,
groups, nested groups, root and scoped adjustment layers, filter layers
with blur padding, layer styles, channel toggles, off-canvas placement,
hidden and empty layers.
"""

from __future__ import annotations

import numpy as np
import pytest

from photo_editor.blending.planar import to_interleaved
from photo_editor.engine.compositor import Compositor
from photo_editor.engine.planar_compositor import PlanarCompositor
from tests.fidelity.scenes import all_scenes

TOL = 1.0 / 255.0
SCENES = sorted(all_scenes().items())


def _both(doc):
    w, h = doc.width, doc.height
    old = Compositor().composite(doc.layers, w, h)
    new = to_interleaved(PlanarCompositor().composite(doc.layers, w, h))
    return old, new


@pytest.mark.parametrize("name,factory", SCENES, ids=[n for n, _ in SCENES])
def test_planar_compositor_matches_original(name, factory):
    old, new = _both(factory())
    assert new.shape == old.shape, f"{name}: shape {new.shape} != {old.shape}"
    diff = np.abs(new - old)
    worst = float(diff.max())
    assert worst <= TOL, (
        f"{name}: max abs diff {worst:.6f} > {TOL:.6f}; "
        f"{int((diff > TOL).sum())} of {diff.size} components differ; "
        f"worst at {np.unravel_index(int(diff.argmax()), diff.shape)}"
    )


def test_planar_compositor_output_is_planar_contiguous():
    doc = all_scenes()["many_layers"]()
    out = PlanarCompositor().composite(doc.layers, doc.width, doc.height)
    assert out.shape == (4, doc.height, doc.width)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]


def test_planar_compositor_is_reusable():
    """Scratch-buffer reuse must not leak state between composites."""
    comp = PlanarCompositor()
    factories = [all_scenes()[n] for n in
                 ("many_layers", "clipping_chain", "group", "nested_group")]
    # Composite each scene twice through the SAME compositor instance and
    # confirm results are stable -- catches scratch buffers being handed
    # out while still referenced.
    for f in factories:
        doc = f()
        a = to_interleaved(comp.composite(doc.layers, doc.width, doc.height)).copy()
        b = to_interleaved(comp.composite(doc.layers, doc.width, doc.height)).copy()
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_interleaving_scenes_through_one_instance():
    """Compositing different documents alternately must not cross-contaminate."""
    comp = PlanarCompositor()
    scenes = all_scenes()
    d1, d2 = scenes["group"](), scenes["clipping_chain"]()
    ref1 = to_interleaved(PlanarCompositor().composite(
        d1.layers, d1.width, d1.height)).copy()
    ref2 = to_interleaved(PlanarCompositor().composite(
        d2.layers, d2.width, d2.height)).copy()
    for _ in range(3):
        a = to_interleaved(comp.composite(d1.layers, d1.width, d1.height)).copy()
        b = to_interleaved(comp.composite(d2.layers, d2.width, d2.height)).copy()
        np.testing.assert_allclose(a, ref1, atol=1e-6)
        np.testing.assert_allclose(b, ref2, atol=1e-6)


def test_group_followed_by_a_clipping_layer_renders():
    """A group with a clipping layer above it aborted every render.

    The compositor tracked borrowed scratch buffers in a list and tested
    membership with `buf in borrowed`, which on ndarrays compares
    elementwise and raises "truth value of an array is ambiguous". The
    arrangement is an ordinary Photoshop-style one, and no reference scene
    combined a group with a clipping layer above it.
    """
    from photo_editor.core.document import Document
    from photo_editor.core.enums import LayerType
    from photo_editor.core.layer import Layer

    doc = Document(64, 48)
    doc.layers.layers.clear()
    doc.layers.add(Layer(name="base", width=64, height=48))
    group = Layer(name="grp", width=64, height=48, layer_type=LayerType.GROUP)
    doc.layers.add(group)
    child = Layer(name="kid", width=32, height=24)
    child.parent_id = group.id
    doc.layers.add(child)
    clip = Layer(name="clip", width=64, height=48)
    clip.clipping_mask = True
    doc.layers.add(clip)
    doc.layers.add(Layer(name="top", width=64, height=48))

    out = PlanarCompositor().composite(doc.layers, 64, 48)
    assert out.shape == (4, 48, 64)


def test_repeated_composites_with_groups_and_clips_are_stable():
    """Scratch buffers are pooled; releasing one still in use would show as
    drift between successive renders of the same document."""
    from photo_editor.core.document import Document
    from photo_editor.core.enums import LayerType
    from photo_editor.core.layer import Layer

    doc = Document(64, 48)
    doc.layers.layers.clear()
    doc.layers.add(Layer(name="base", width=64, height=48))
    group = Layer(name="grp", width=64, height=48, layer_type=LayerType.GROUP)
    doc.layers.add(group)
    kid = Layer(name="kid", width=40, height=30)
    kid.parent_id = group.id
    doc.layers.add(kid)
    clip = Layer(name="clip", width=64, height=48)
    clip.clipping_mask = True
    doc.layers.add(clip)

    comp = PlanarCompositor()
    first = comp.composite(doc.layers, 64, 48).copy()
    for _ in range(4):
        again = comp.composite(doc.layers, 64, 48)
        np.testing.assert_allclose(again, first, atol=1e-6)
