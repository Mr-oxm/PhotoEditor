"""Reference scenes for render-fidelity testing.

Small, deterministic documents that exercise every semantically interesting
path through the compositor: all blend modes, opacity, layer masks, mask
layers, clipping chains, groups, nested groups, layer styles, adjustment and
filter layers, channel toggles, off-canvas placement, and empty layers.

Kept small (256x192) so golden buffers stay tiny and comparisons are fast.
Every scene is built from a fixed seed so results are bit-reproducible.
"""

from __future__ import annotations

import numpy as np

from photo_editor.core.document import Document
from photo_editor.core.enums import BlendMode, LayerType
from photo_editor.core.layer import Layer

W, H = 160, 120


# ---------------------------------------------------------------------------
# Deterministic pixel content
# ---------------------------------------------------------------------------

def _gradient(w: int, h: int, seed: int, alpha: float = 1.0) -> np.ndarray:
    """Structured RGBA content — gradients plus fixed noise."""
    rng = np.random.default_rng(seed)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    img = np.empty((h, w, 4), dtype=np.float32)
    img[..., 0] = xx
    img[..., 1] = yy
    img[..., 2] = (xx + yy) * 0.5
    img[..., :3] += rng.random((h, w, 3), dtype=np.float32) * 0.25
    np.clip(img[..., :3], 0.0, 1.0, out=img[..., :3])
    img[..., 3] = alpha
    return img


def _radial_alpha(w: int, h: int, seed: int) -> np.ndarray:
    """Content with a soft radial alpha falloff — exercises partial alpha."""
    img = _gradient(w, h, seed)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    r = np.sqrt(xx * xx + yy * yy)
    img[..., 3] = np.clip(1.0 - r, 0.0, 1.0)
    return img


def _soft_mask(w: int, h: int) -> np.ndarray:
    """Horizontal ramp mask in [0, 1]."""
    return np.clip(
        np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :].repeat(h, 0),
        0.0, 1.0,
    )


def _blank_doc(name: str) -> Document:
    doc = Document(W, H, name=name)
    doc.layers.layers.clear()
    return doc


def _add(doc: Document, pixels: np.ndarray, name: str, **kw) -> Layer:
    h, w = pixels.shape[:2]
    layer = Layer(name=name, width=w, height=h,
                  layer_type=kw.pop("layer_type", LayerType.RASTER))
    layer.pixels = pixels
    for key, value in kw.items():
        setattr(layer, key, value)
    doc.layers.add(layer)
    return layer


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

def scene_blend_mode(mode: BlendMode) -> Document:
    """Two overlapping layers combined with *mode*."""
    doc = _blank_doc(f"blend-{mode.name}")
    _add(doc, _gradient(W, H, 1), "base")
    _add(doc, _radial_alpha(W, H, 2), "over", blend_mode=mode, opacity=0.85)
    return doc


def scene_opacity() -> Document:
    doc = _blank_doc("opacity")
    _add(doc, _gradient(W, H, 3), "base")
    for i, op in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        _add(doc, _radial_alpha(W // 2, H // 2, 10 + i), f"o{i}",
             opacity=op, position=(i * 20, i * 15))
    return doc


def scene_legacy_mask() -> Document:
    doc = _blank_doc("legacy-mask")
    _add(doc, _gradient(W, H, 4), "base")
    top = _add(doc, _gradient(W, H, 5), "masked")
    top.mask = _soft_mask(W, H)
    return doc


def scene_mask_layer() -> Document:
    doc = _blank_doc("mask-layer")
    _add(doc, _gradient(W, H, 6), "base")
    target = _add(doc, _radial_alpha(W, H, 7), "target")
    mask_px = np.zeros((H, W, 4), dtype=np.float32)
    mask_px[..., :3] = _soft_mask(W, H)[..., None]
    mask_px[..., 3] = 1.0
    mask = _add(doc, mask_px, "mask", layer_type=LayerType.MASK,
                parent_id=target.id)
    target.mask_layers.append(mask.id)
    return doc


def scene_standalone_mask() -> Document:
    doc = _blank_doc("standalone-mask")
    _add(doc, _gradient(W, H, 8), "base")
    _add(doc, _radial_alpha(W, H, 9), "mid")
    mask_px = np.zeros((H, W, 4), dtype=np.float32)
    mask_px[..., :3] = _soft_mask(W, H)[..., None]
    mask_px[..., 3] = 1.0
    _add(doc, mask_px, "global-mask", layer_type=LayerType.MASK)
    return doc


def scene_clipping_chain() -> Document:
    doc = _blank_doc("clipping-chain")
    _add(doc, _gradient(W, H, 11), "bg")
    _add(doc, _radial_alpha(W, H, 12), "clip-base")
    _add(doc, _gradient(W, H, 13), "clipped-1", clipping_mask=True,
         blend_mode=BlendMode.MULTIPLY)
    _add(doc, _gradient(W, H, 14), "clipped-2", clipping_mask=True,
         opacity=0.6)
    return doc


def scene_group() -> Document:
    doc = _blank_doc("group")
    _add(doc, _gradient(W, H, 15), "bg")
    group = Layer(name="grp", width=W, height=H, layer_type=LayerType.GROUP)
    doc.layers.add(group)
    _add(doc, _radial_alpha(W, H, 16), "g-child-1", parent_id=group.id)
    _add(doc, _gradient(W, H, 17), "g-child-2", parent_id=group.id,
         blend_mode=BlendMode.SCREEN, opacity=0.7)
    group.opacity = 0.8
    group.blend_mode = BlendMode.OVERLAY
    return doc


def scene_nested_group() -> Document:
    """Groups inside groups, with content the blend modes act on.

    The MULTIPLY layer needs something beneath it *inside the group* --
    a group composites onto a transparent canvas, so multiplying against
    that gives uniform black and the whole scene becomes a useless
    all-zero reference that no rendering change could ever perturb.
    """
    doc = _blank_doc("nested-group")
    _add(doc, _gradient(W, H, 18), "bg")
    outer = Layer(name="outer", width=W, height=H, layer_type=LayerType.GROUP)
    doc.layers.add(outer)
    # Opaque base inside the group, so the modes above it have a substrate.
    _add(doc, _gradient(W, H, 60), "group-base", parent_id=outer.id)
    inner = Layer(name="inner", width=W, height=H, layer_type=LayerType.GROUP)
    inner.parent_id = outer.id
    doc.layers.add(inner)
    _add(doc, _radial_alpha(W, H, 19), "deep", parent_id=inner.id,
         opacity=0.8)
    _add(doc, _gradient(W, H, 20), "shallow", parent_id=outer.id,
         blend_mode=BlendMode.MULTIPLY, opacity=0.7)
    outer.opacity = 0.9
    return doc


def scene_adjustment_layer() -> Document:
    """Root-level adjustment layer applied to everything below it."""
    from photo_editor.adjustments.brightness_contrast import BrightnessContrast
    doc = _blank_doc("adjustment-root")
    _add(doc, _gradient(W, H, 21), "base")
    adj = Layer(name="bc", width=W, height=H, layer_type=LayerType.ADJUSTMENT)
    adj.adjustment = BrightnessContrast()
    adj.adjustment_params = {"brightness": 25, "contrast": 40}
    doc.layers.add(adj)
    _add(doc, _radial_alpha(W, H, 22), "above-adj")
    return doc


def scene_child_adjustment() -> Document:
    """Adjustment layer scoped to a single parent layer."""
    from photo_editor.adjustments.curves import Curves
    doc = _blank_doc("adjustment-child")
    _add(doc, _gradient(W, H, 23), "base")
    target = _add(doc, _gradient(W, H, 24), "target")
    adj = Layer(name="curves", width=W, height=H,
                layer_type=LayerType.ADJUSTMENT)
    adj.adjustment = Curves()
    adj.adjustment_params = {
        "channel": "RGB",
        "points_rgb": [[0, 20], [128, 150], [255, 240]],
        "points_red": [[0, 0], [255, 255]],
        "points_green": [[0, 0], [255, 255]],
        "points_blue": [[0, 0], [255, 255]],
    }
    adj.parent_id = target.id
    doc.layers.add(adj)
    return doc


def scene_filter_layer() -> Document:
    """Filter layer with blur padding (exercises _apply_filters_padded)."""
    from photo_editor.filters.blur.gaussian_blur import GaussianBlur
    doc = _blank_doc("filter-child")
    _add(doc, _gradient(W, H, 25), "base")
    target = _add(doc, _radial_alpha(W // 2, H // 2, 26), "target",
                  position=(60, 40))
    flt = Layer(name="blur", width=W, height=H, layer_type=LayerType.FILTER)
    flt.adjustment = GaussianBlur()
    flt.adjustment_params = {"radius": 6.0, "preserve_alpha": False}
    flt.parent_id = target.id
    doc.layers.add(flt)
    return doc


def scene_layer_styles() -> Document:
    from photo_editor.styles.drop_shadow import DropShadow
    from photo_editor.styles.stroke import Stroke
    doc = _blank_doc("styles")
    _add(doc, _gradient(W, H, 27), "base")
    styled = _add(doc, _radial_alpha(W // 2, H // 2, 28), "styled",
                  position=(50, 40))
    styled.styles.append(DropShadow())
    styled.styles.append(Stroke())
    return doc


def scene_channels() -> Document:
    doc = _blank_doc("channels")
    _add(doc, _gradient(W, H, 29), "base")
    _add(doc, _gradient(W, H, 30), "no-red", channel_r=False)
    _add(doc, _radial_alpha(W, H, 31), "no-blue-alpha",
         channel_b=False, position=(30, 20))
    return doc


def scene_offcanvas() -> Document:
    """Layers partly and wholly outside the canvas."""
    doc = _blank_doc("offcanvas")
    _add(doc, _gradient(W, H, 32), "base")
    _add(doc, _radial_alpha(120, 120, 33), "top-left", position=(-60, -60))
    _add(doc, _radial_alpha(120, 120, 34), "bottom-right",
         position=(W - 40, H - 40))
    _add(doc, _radial_alpha(64, 64, 35), "fully-outside",
         position=(W + 200, H + 200))
    return doc


def scene_hidden_and_empty() -> Document:
    doc = _blank_doc("hidden-empty")
    _add(doc, _gradient(W, H, 36), "base")
    _add(doc, _gradient(W, H, 37), "hidden", visible=False)
    _add(doc, np.zeros((H, W, 4), dtype=np.float32), "empty")
    _add(doc, _radial_alpha(W, H, 38), "visible-top", opacity=0.5)
    return doc


def scene_many_layers() -> Document:
    """Deeper stack with mixed blend modes — closer to a real project."""
    doc = _blank_doc("many-layers")
    modes = [BlendMode.NORMAL, BlendMode.MULTIPLY, BlendMode.SCREEN,
             BlendMode.OVERLAY, BlendMode.SOFT_LIGHT, BlendMode.DIFFERENCE]
    _add(doc, _gradient(W, H, 40), "base")
    for i in range(12):
        _add(doc, _radial_alpha(W - i * 8, H - i * 6, 41 + i), f"L{i}",
             blend_mode=modes[i % len(modes)],
             opacity=0.5 + (i % 5) * 0.1,
             position=(i * 5, i * 4))
    return doc


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def all_scenes() -> dict[str, callable]:
    """Name -> zero-arg factory for every reference scene."""
    scenes: dict[str, callable] = {
        "opacity": scene_opacity,
        "legacy_mask": scene_legacy_mask,
        "mask_layer": scene_mask_layer,
        "standalone_mask": scene_standalone_mask,
        "clipping_chain": scene_clipping_chain,
        "group": scene_group,
        "nested_group": scene_nested_group,
        "adjustment_root": scene_adjustment_layer,
        "adjustment_child": scene_child_adjustment,
        "filter_child": scene_filter_layer,
        "styles": scene_layer_styles,
        "channels": scene_channels,
        "offcanvas": scene_offcanvas,
        "hidden_empty": scene_hidden_and_empty,
        "many_layers": scene_many_layers,
    }
    for mode in BlendMode:
        scenes[f"blend_{mode.name}"] = (
            lambda m=mode: scene_blend_mode(m)
        )
    return scenes
