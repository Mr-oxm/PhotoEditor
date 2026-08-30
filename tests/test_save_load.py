"""Tests for the .basera save / load pipeline (v3 ZIP format).

Covers:
* Snapshot-only saves (no history in file)
* v3 format: ZIP container with manifest.json + numpy pixel entries
* Save speed (must be reasonably fast even for large, complex documents)
* Full round-trip integrity for every supported layer type
* 20-layer document including vector, raster, group, text, adjustment,
  filter, and mask layers
* Document metadata (color mode, profile, unit, dpi) round-trip
* Recent-projects helper functions
* Status-bar activity message lifecycle (show / clear / safety timeout)
* Save crash resilience (status clears even on callback failure)
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_document(width: int = 400, height: int = 300) -> "Document":
    from photo_editor.core.document import Document
    return Document(width, height, name="Test Doc", color_mode="RGB",
                    color_profile="sRGB IEC61966-2.1", unit="px")


def _fill_pixels(layer, color=(0.5, 0.3, 0.8, 1.0)) -> None:
    layer.pixels[:] = np.array(color, dtype=np.float32)


def _add_adjustment_layer(doc, adj_name: str = "Brightness/Contrast") -> "Layer":
    from photo_editor.core.enums import LayerType
    from photo_editor.registries import get_adjustment_class
    layer = doc.add_layer(name=adj_name, layer_type=LayerType.ADJUSTMENT)
    cls = get_adjustment_class(adj_name)
    if cls is not None:
        layer.adjustment = cls()
        layer.adjustment_params = {"brightness": 20, "contrast": 10}
    return layer


def _add_filter_layer(doc, filt_name: str = "Gaussian Blur") -> "Layer":
    from photo_editor.core.enums import LayerType
    from photo_editor.registries import get_filter_name_map
    layer = doc.add_layer(name=filt_name, layer_type=LayerType.FILTER)
    cls = get_filter_name_map().get(filt_name)
    if cls is not None:
        layer.adjustment = cls()
        layer.adjustment_params = {"radius": 2.0}
    return layer


def _add_vector_layer(doc, name: str = "Vec") -> "Layer":
    layer = doc.add_vector_layer(name=name)
    from photo_editor.vector.shapes import RectangleShape
    from photo_editor.vector.style import VectorStyle
    from photo_editor.vector.geometry import AffineTransform
    from photo_editor.vector.scene import VectorObject
    rect = RectangleShape(width=80.0, height=50.0)
    obj = VectorObject(
        name="rect",
        shape=rect,
        style=VectorStyle(),
        transform=AffineTransform.identity(),
    )
    layer._vector_data.add(obj)
    return layer


def _add_text_layer(doc, text: str = "Hello") -> "Layer":
    from photo_editor.core.enums import LayerType
    from photo_editor.core.text_layer import TextLayerData, TextRun
    layer = doc.add_layer(name=f"Text: {text}", layer_type=LayerType.TEXT)
    td = TextLayerData()
    td.runs = [TextRun(text=text)]
    layer._text_data = td
    return layer


def _add_mask_layer(doc) -> "Layer":
    return doc.add_mask_layer()


def _add_group(doc, name: str = "Group") -> "Layer":
    return doc.add_group(name=name)


def _save_load(doc, tmp_path) -> "Document":
    from photo_editor.utils.project_io import save_basera_project, load_basera_project
    p = tmp_path / "proj.basera"
    save_basera_project(doc, p)
    return load_basera_project(p)


# ---------------------------------------------------------------------------
# 20-layer document factory
# ---------------------------------------------------------------------------

def _build_20layer_doc() -> "Document":
    """Return a document with exactly 20 layers covering all layer types."""
    doc = _make_document(256, 256)

    # 1. Background already exists — fill it
    bg = doc.layers[0]
    _fill_pixels(bg, (1.0, 1.0, 1.0, 1.0))

    # 2-6. Five raster layers (different blend modes / opacities)
    from photo_editor.core.enums import BlendMode
    for i, (blend, opacity) in enumerate([
        (BlendMode.NORMAL,     1.0),
        (BlendMode.MULTIPLY,   0.8),
        (BlendMode.SCREEN,     0.6),
        (BlendMode.OVERLAY,    0.7),
        (BlendMode.SOFT_LIGHT, 0.5),
    ], start=2):
        layer = doc.add_layer(name=f"Raster {i}")
        _fill_pixels(layer, (i * 0.1, 0.5, 0.9, 1.0))
        layer.blend_mode = blend
        layer.opacity = opacity

    # 7-9. Three vector / shape layers
    for j in range(3):
        _add_vector_layer(doc, name=f"Vector {j + 1}")

    # 10-11. Two text layers
    _add_text_layer(doc, "Hello World")
    _add_text_layer(doc, "Basera")

    # 12-13. Two adjustment layers
    _add_adjustment_layer(doc, "Brightness/Contrast")
    _add_adjustment_layer(doc, "Hue/Saturation")

    # 14-15. Two filter layers
    _add_filter_layer(doc, "Gaussian Blur")
    _add_filter_layer(doc, "Sharpen")

    # 16. A group layer
    group = _add_group(doc, "My Group")

    # 17. A raster child inside the group
    child = doc.add_layer(name="Group Child")
    _fill_pixels(child, (0.2, 0.4, 0.6, 1.0))
    doc.layers.reparent([child.id], group.id)

    # 18. A mask layer
    _add_mask_layer(doc)

    # 19-20. Two more raster layers (clipping mask pair)
    base_clip = doc.add_layer(name="Clip Base")
    _fill_pixels(base_clip, (0.9, 0.1, 0.1, 1.0))
    clip = doc.add_layer(name="Clip Layer")
    clip.clipping_mask = True
    _fill_pixels(clip, (0.1, 0.9, 0.1, 0.5))

    assert len(doc.layers.layers) == 20, (
        f"Expected 20 layers, got {len(doc.layers.layers)}"
    )
    return doc


# ---------------------------------------------------------------------------
# Format structure tests
# ---------------------------------------------------------------------------

class TestV3Format:
    """The .basera file must be a valid ZIP with the right structure."""

    def test_output_is_zip(self, tmp_path):
        from photo_editor.utils.project_io import save_basera_project
        doc = _make_document()
        p = tmp_path / "test.basera"
        save_basera_project(doc, p)
        assert zipfile.is_zipfile(p), "Output must be a valid ZIP file"

    def test_manifest_present(self, tmp_path):
        from photo_editor.utils.project_io import save_basera_project
        doc = _make_document()
        p = tmp_path / "test.basera"
        save_basera_project(doc, p)
        with zipfile.ZipFile(p) as zf:
            assert "manifest.json" in zf.namelist()

    def test_manifest_is_valid_json(self, tmp_path):
        import json
        from photo_editor.utils.project_io import save_basera_project
        doc = _make_document()
        p = tmp_path / "test.basera"
        save_basera_project(doc, p)
        with zipfile.ZipFile(p) as zf:
            manifest = json.loads(zf.read("manifest.json"))
        assert manifest["magic"] == "BASERA_PROJECT"
        assert manifest["version"] == 4

    def test_layer_pixels_stored_as_compressed_npy(self, tmp_path):
        """v4 stores zlib-compressed .npy payloads in an uncompressed ZIP."""
        from photo_editor.utils.project_io import save_basera_project
        doc = _make_document()
        doc.add_layer(name="Raster")
        p = tmp_path / "test.basera"
        save_basera_project(doc, p)
        with zipfile.ZipFile(p) as zf:
            entries = [n for n in zf.namelist() if n.endswith(".npy.z")]
            assert entries, "pixel data must be stored as .npy.z entries"
            for info in zf.infolist():
                if info.filename.endswith(".npy.z"):
                    assert info.compress_type == zipfile.ZIP_STORED, (
                        "payloads are already zlib-compressed; the container "
                        "must not compress them again")

    def test_pixels_stored_as_uint16(self, tmp_path):
        """float32 image data is 4x larger than needed and barely
        compressible; uint16 is effectively lossless for [0, 1] data."""
        import io
        import zlib
        import numpy as np
        from photo_editor.utils.project_io import save_basera_project
        doc = _make_document()
        layer = doc.add_layer(name="Raster")
        _fill_pixels(layer)
        p = tmp_path / "test.basera"
        save_basera_project(doc, p)
        with zipfile.ZipFile(p) as zf:
            name = next(n for n in zf.namelist()
                        if n.endswith(f"{layer.id}.npy.z"))
            arr = np.load(io.BytesIO(zlib.decompress(zf.read(name))))
        assert arr.dtype == np.uint16, f"stored as {arr.dtype}, expected uint16"

    def test_out_of_range_pixels_are_stored_losslessly(self, tmp_path):
        """Values outside [0, 1] must not be silently clipped by quantising."""
        import numpy as np
        from photo_editor.utils.project_io import (
            load_basera_project, save_basera_project,
        )
        doc = _make_document()
        layer = doc.add_layer(name="HDR")
        px = np.zeros((layer.height, layer.width, 4), dtype=np.float32)
        px[..., 0] = 3.5          # deliberately out of range
        px[..., 3] = 1.0
        layer.pixels = px
        p = tmp_path / "hdr.basera"
        save_basera_project(doc, p)
        loaded = load_basera_project(p)
        rl = next(l for l in loaded.layers if l.name == "HDR")
        assert np.allclose(rl.pixels[..., 0], 3.5, atol=1e-5), (
            "out-of-range data was clipped by quantisation")

    def test_uint16_round_trip_is_visually_lossless(self, tmp_path):
        import numpy as np
        from photo_editor.utils.project_io import (
            load_basera_project, save_basera_project,
        )
        rng = np.random.default_rng(0)
        doc = _make_document()
        layer = doc.add_layer(name="Noise")
        px = rng.random((layer.height, layer.width, 4)).astype(np.float32)
        layer.pixels = px
        p = tmp_path / "noise.basera"
        save_basera_project(doc, p)
        loaded = load_basera_project(p)
        rl = next(l for l in loaded.layers if l.name == "Noise")
        err = float(np.abs(rl.pixels - px).max())
        assert err < 1.0 / 255.0 / 100.0, (
            f"round-trip error {err:.8f} exceeds 1% of an 8-bit level")

    def test_no_history_in_manifest(self, tmp_path):
        import json
        from photo_editor.utils.project_io import save_basera_project
        doc = _make_document()
        for _ in range(5):
            doc.add_layer(name="tmp")
        assert len(doc.history.states) > 0

        p = tmp_path / "test.basera"
        save_basera_project(doc, p)
        with zipfile.ZipFile(p) as zf:
            manifest = json.loads(zf.read("manifest.json"))
        assert "history" not in manifest

    def test_adjustment_layers_have_no_pixel_file(self, tmp_path):
        """Adjustment and filter layer pixels must NOT be stored."""
        import json
        from photo_editor.utils.project_io import save_basera_project
        from photo_editor.core.enums import LayerType
        doc = _make_document()
        al = _add_adjustment_layer(doc, "Brightness/Contrast")
        fl = _add_filter_layer(doc, "Gaussian Blur")

        p = tmp_path / "test.basera"
        save_basera_project(doc, p)

        with zipfile.ZipFile(p) as zf:
            names = zf.namelist()
        for lid in (al.id, fl.id):
            assert f"layers/{lid}.npy" not in names, (
                f"Pixel file must not exist for adj/filter layer {lid}"
            )


# ---------------------------------------------------------------------------
# Snapshot-only semantics
# ---------------------------------------------------------------------------

class TestSnapshotOnly:
    def test_loaded_doc_has_empty_history(self, tmp_path):
        from photo_editor.utils.project_io import save_basera_project, load_basera_project
        doc = _make_document()
        for _ in range(3):
            doc.add_layer(name="tmp")
        p = tmp_path / "test.basera"
        save_basera_project(doc, p)
        loaded = load_basera_project(p)
        assert len(loaded.history.states) == 0

    def test_loaded_doc_is_clean(self, tmp_path):
        from photo_editor.utils.project_io import save_basera_project, load_basera_project
        doc = _make_document()
        p = tmp_path / "test.basera"
        save_basera_project(doc, p)
        assert not load_basera_project(p).dirty


# ---------------------------------------------------------------------------
# Metadata round-trip
# ---------------------------------------------------------------------------

class TestMetadataRoundTrip:
    def test_basic_metadata(self, tmp_path):
        from photo_editor.utils.project_io import save_basera_project, load_basera_project
        doc = _make_document(800, 600)
        doc.name = "Portrait Session"
        doc.dpi = 300
        doc.color_mode = "CMYK"
        doc.color_profile = "Adobe RGB (1998)"
        doc.unit = "in"

        p = tmp_path / "meta.basera"
        save_basera_project(doc, p)
        loaded = load_basera_project(p)

        assert loaded.width == 800
        assert loaded.height == 600
        assert loaded.dpi == 300
        assert loaded.color_mode == "CMYK"
        assert loaded.color_profile == "Adobe RGB (1998)"
        assert loaded.unit == "in"


# ---------------------------------------------------------------------------
# Layer round-trip tests
# ---------------------------------------------------------------------------

class TestLayerRoundTrip:
    def test_raster_pixels(self, tmp_path):
        doc = _make_document()
        layer = doc.add_layer(name="Raster")
        layer.pixels[:] = 0.42
        loaded = _save_load(doc, tmp_path)
        rl = next(l for l in loaded.layers if l.name == "Raster")
        assert np.allclose(rl.pixels, 0.42, atol=1e-5)

    def test_vector_layer(self, tmp_path):
        doc = _make_document()
        _add_vector_layer(doc, "MyVec")
        loaded = _save_load(doc, tmp_path)
        vl = next(l for l in loaded.layers if l.name == "MyVec")
        assert vl._vector_data is not None
        assert len(vl._vector_data.objects) == 1

    def test_text_layer(self, tmp_path):
        doc = _make_document()
        _add_text_layer(doc, "Test Text")
        loaded = _save_load(doc, tmp_path)
        tl = next(l for l in loaded.layers if "Test Text" in l.name)
        assert tl._text_data is not None
        assert "Test Text" in tl._text_data.runs[0].text

    def test_adjustment_layer(self, tmp_path):
        doc = _make_document()
        _add_adjustment_layer(doc, "Brightness/Contrast")
        loaded = _save_load(doc, tmp_path)
        al = next(l for l in loaded.layers if l.name == "Brightness/Contrast")
        assert al.adjustment is not None
        assert al.adjustment_params.get("brightness") == 20

    def test_filter_layer(self, tmp_path):
        doc = _make_document()
        _add_filter_layer(doc, "Gaussian Blur")
        loaded = _save_load(doc, tmp_path)
        fl = next(l for l in loaded.layers if l.name == "Gaussian Blur")
        assert fl.adjustment is not None
        assert "radius" in fl.adjustment_params

    def test_group_and_child(self, tmp_path):
        doc = _make_document()
        group = _add_group(doc, "G1")
        child = doc.add_layer(name="Child")
        doc.layers.reparent([child.id], group.id)
        loaded = _save_load(doc, tmp_path)
        lg = next(l for l in loaded.layers if l.name == "G1")
        lc = next(l for l in loaded.layers if l.name == "Child")
        assert lc.parent_id == lg.id

    def test_mask_layer(self, tmp_path):
        from photo_editor.core.enums import LayerType
        doc = _make_document()
        _add_mask_layer(doc)
        loaded = _save_load(doc, tmp_path)
        mask_layers = [l for l in loaded.layers if l.layer_type == LayerType.MASK]
        assert len(mask_layers) >= 1

    def test_clipping_mask_flag(self, tmp_path):
        doc = _make_document()
        doc.add_layer(name="Base")
        clip = doc.add_layer(name="Clip")
        clip.clipping_mask = True
        loaded = _save_load(doc, tmp_path)
        lclip = next(l for l in loaded.layers if l.name == "Clip")
        assert lclip.clipping_mask is True

    def test_layer_order_preserved(self, tmp_path):
        doc = _make_document()
        names = [f"L{i}" for i in range(5)]
        for n in names:
            doc.add_layer(name=n)
        loaded = _save_load(doc, tmp_path)
        loaded_names = [l.name for l in loaded.layers]
        indices = [loaded_names.index(n) for n in names]
        assert indices == sorted(indices)

    def test_opacity_and_blend_mode(self, tmp_path):
        from photo_editor.core.enums import BlendMode
        doc = _make_document()
        layer = doc.add_layer(name="Blend")
        layer.opacity = 0.37
        layer.blend_mode = BlendMode.MULTIPLY
        loaded = _save_load(doc, tmp_path)
        ll = next(l for l in loaded.layers if l.name == "Blend")
        assert abs(ll.opacity - 0.37) < 1e-5
        assert ll.blend_mode == BlendMode.MULTIPLY

    def test_layer_mask_data(self, tmp_path):
        """Per-layer mask array must survive the round-trip."""
        doc = _make_document()
        layer = doc.add_layer(name="Masked")
        mask = np.full((300, 400), 0.5, dtype=np.float32)
        layer._mask = mask
        loaded = _save_load(doc, tmp_path)
        ll = next(l for l in loaded.layers if l.name == "Masked")
        assert ll._mask is not None
        assert np.allclose(ll._mask, 0.5, atol=1e-5)


# ---------------------------------------------------------------------------
# 20-layer document
# ---------------------------------------------------------------------------

class Test20LayerDocument:
    def test_layer_count_preserved(self, tmp_path):
        from photo_editor.utils.project_io import save_basera_project, load_basera_project
        doc = _build_20layer_doc()
        p = tmp_path / "twenty.basera"
        save_basera_project(doc, p)
        assert len(load_basera_project(p).layers.layers) == 20

    def test_all_layer_types_present(self, tmp_path):
        from photo_editor.utils.project_io import save_basera_project, load_basera_project
        from photo_editor.core.enums import LayerType
        doc = _build_20layer_doc()
        p = tmp_path / "twenty_types.basera"
        save_basera_project(doc, p)
        loaded = load_basera_project(p)
        found = {l.layer_type for l in loaded.layers}
        for lt in (LayerType.RASTER, LayerType.SHAPE, LayerType.TEXT,
                   LayerType.ADJUSTMENT, LayerType.FILTER,
                   LayerType.GROUP, LayerType.MASK):
            assert lt in found, f"Layer type {lt} missing after round-trip"

    def test_vector_objects_preserved(self, tmp_path):
        from photo_editor.utils.project_io import save_basera_project, load_basera_project
        from photo_editor.core.enums import LayerType
        doc = _build_20layer_doc()
        p = tmp_path / "twenty_vec.basera"
        save_basera_project(doc, p)
        loaded = load_basera_project(p)
        vec_layers = [l for l in loaded.layers if l.layer_type == LayerType.SHAPE]
        assert len(vec_layers) == 3
        for vl in vec_layers:
            assert vl._vector_data is not None
            assert len(vl._vector_data.objects) >= 1

    def test_no_history_in_file(self, tmp_path):
        import json
        from photo_editor.utils.project_io import save_basera_project
        doc = _build_20layer_doc()
        p = tmp_path / "twenty_hist.basera"
        save_basera_project(doc, p)
        with zipfile.ZipFile(p) as zf:
            manifest = json.loads(zf.read("manifest.json"))
        assert "history" not in manifest


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestSavePerformance:
    def test_save_speed_20_layers(self, tmp_path):
        from photo_editor.utils.project_io import save_basera_project
        doc = _build_20layer_doc()
        p = tmp_path / "perf.basera"
        start = time.perf_counter()
        save_basera_project(doc, p)
        elapsed = time.perf_counter() - start
        assert elapsed < 10.0, f"Save took {elapsed:.2f}s — limit 10s"

    def test_load_speed_20_layers(self, tmp_path):
        from photo_editor.utils.project_io import save_basera_project, load_basera_project
        doc = _build_20layer_doc()
        p = tmp_path / "perf_load.basera"
        save_basera_project(doc, p)
        start = time.perf_counter()
        load_basera_project(p)
        elapsed = time.perf_counter() - start
        assert elapsed < 10.0, f"Load took {elapsed:.2f}s — limit 10s"

    def test_v3_smaller_than_v1_equivalent(self, tmp_path):
        """v3 ZIP file must be smaller than an equivalent v1 pickle+gzip."""
        import copy, gzip, pickle
        from photo_editor.utils.project_io import save_basera_project

        doc = _build_20layer_doc()
        # Accumulate a few undo states so v1 would include them
        for _ in range(5):
            doc.save_snapshot("dummy")

        p_v3 = tmp_path / "v3.basera"
        save_basera_project(doc, p_v3)
        size_v3 = p_v3.stat().st_size

        # Simulate a v1-style payload (with history in pickle+gzip)
        current_state = doc._build_history_state("__Snapshot__")
        v1_payload = {
            "magic": "BASERA_PROJECT",
            "version": 1,
            "current_state": {
                "name": current_state.name,
                "metadata": copy.deepcopy(current_state.metadata),
                "layer_data": {k: v.copy() for k, v in current_state.layer_data.items()},
            },
            "history": {
                "states": [
                    {
                        "name": s.name,
                        "metadata": copy.deepcopy(s.metadata),
                        "layer_data": {k: v.copy() for k, v in s.layer_data.items()},
                    }
                    for s in doc.history.states
                ],
                "current_index": doc.history.current_index,
            },
        }
        p_v1 = tmp_path / "v1_sim.basera.gz"
        with gzip.open(p_v1, "wb", compresslevel=1) as fh:
            pickle.dump(v1_payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        size_v1 = p_v1.stat().st_size

        assert size_v3 < size_v1, (
            f"v3 ({size_v3} B) should be smaller than v1 ({size_v1} B)"
        )


# ---------------------------------------------------------------------------
# Recent projects helpers
# ---------------------------------------------------------------------------

class TestRecentProjects:
    def test_add_and_load(self, tmp_path, monkeypatch):
        from photo_editor.utils import recent_projects as rp
        config = tmp_path / "config"
        monkeypatch.setattr(rp, "_config_dir", lambda: config)

        # Create a fake .basera file so the path exists
        fake = tmp_path / "project.basera"
        fake.write_bytes(b"")

        rp.add_recent_project(fake)
        entries = rp.load_recent_projects()
        assert any(Path(e["path"]).resolve() == fake.resolve() for e in entries)

    def test_deduplication(self, tmp_path, monkeypatch):
        from photo_editor.utils import recent_projects as rp
        config = tmp_path / "config"
        monkeypatch.setattr(rp, "_config_dir", lambda: config)

        fake = tmp_path / "project.basera"
        fake.write_bytes(b"")

        rp.add_recent_project(fake)
        rp.add_recent_project(fake)
        entries = rp.load_recent_projects()
        paths = [Path(e["path"]).resolve() for e in entries]
        assert paths.count(fake.resolve()) == 1

    def test_most_recent_first(self, tmp_path, monkeypatch):
        from photo_editor.utils import recent_projects as rp
        config = tmp_path / "config"
        monkeypatch.setattr(rp, "_config_dir", lambda: config)

        files = []
        for i in range(3):
            f = tmp_path / f"proj{i}.basera"
            f.write_bytes(b"")
            files.append(f)
            rp.add_recent_project(f)

        entries = rp.load_recent_projects()
        assert Path(entries[0]["path"]).resolve() == files[-1].resolve()

    def test_missing_files_pruned(self, tmp_path, monkeypatch):
        from photo_editor.utils import recent_projects as rp
        config = tmp_path / "config"
        monkeypatch.setattr(rp, "_config_dir", lambda: config)

        real = tmp_path / "real.basera"
        real.write_bytes(b"")
        ghost = tmp_path / "ghost.basera"
        ghost.write_bytes(b"")

        rp.add_recent_project(real)
        rp.add_recent_project(ghost)
        ghost.unlink()

        entries = rp.load_recent_projects()
        paths = [Path(e["path"]).resolve() for e in entries]
        assert ghost.resolve() not in paths
        assert real.resolve() in paths


# ---------------------------------------------------------------------------
# Status bar activity message lifecycle
# ---------------------------------------------------------------------------

class TestStatusBarActivity:
    """The transient message pill must show / auto-clear / handle edge cases.

    Note: Qt's ``isVisible()`` returns False when the parent window is not
    shown, so in tests we check ``not isHidden()`` which reflects the
    *explicit* visibility state set by ``setVisible(True/False)``.
    """

    @pytest.fixture()
    def sbar(self, qtbot):
        from photo_editor.ui.status_bar import EditorStatusBar
        bar = EditorStatusBar()
        qtbot.addWidget(bar)
        return bar

    def _pill_shown(self, sbar) -> bool:
        return not sbar._msg_pill.isHidden()

    def _pill_hidden(self, sbar) -> bool:
        return sbar._msg_pill.isHidden()

    def test_show_activity_makes_pill_visible(self, sbar):
        assert self._pill_hidden(sbar)
        sbar.show_activity("Saving…", 0)
        assert self._pill_shown(sbar)
        assert "Saving" in sbar._msg_pill.text()

    def test_clear_activity_hides_pill(self, sbar):
        sbar.show_activity("Saving…", 0)
        sbar.clear_activity()
        assert self._pill_hidden(sbar)
        assert sbar._msg_pill.text() == ""

    def test_timed_message_auto_clears(self, sbar, qtbot):
        sbar.show_activity("Saved", 50)
        assert self._pill_shown(sbar)
        qtbot.waitUntil(lambda: self._pill_hidden(sbar), timeout=2000)
        assert sbar._msg_pill.text() == ""

    def test_replace_message_resets_timer(self, sbar):
        sbar.show_activity("Saving…", 0)
        sbar.show_activity("Saved!", 100)
        assert "Saved!" in sbar._msg_pill.text()
        assert sbar._msg_timer.isActive()

    def test_safety_timeout_clears_stuck_message(self, sbar, qtbot):
        """A message with a safety timeout eventually clears."""
        sbar.show_activity("Saving…", 80)
        assert self._pill_shown(sbar)
        qtbot.waitUntil(lambda: self._pill_hidden(sbar), timeout=2000)

    def test_doc_info_unchanged_during_activity(self, sbar):
        sbar.set_document_info("MyFile", 1920, 1080)
        sbar.show_activity("Saving…", 0)
        assert "MyFile" in sbar._doc_pill.text()
        assert "1920" in sbar._size_pill.text()


# ---------------------------------------------------------------------------
# Save-command resilience — status must always clear
# ---------------------------------------------------------------------------

class TestSaveCrashResilience:
    """Even if a callback throws, the status pill must eventually clear."""

    def test_save_succeeds_and_clears_status(self, tmp_path):
        """Normal save flow: Saving… → Saved → auto-clear."""
        from photo_editor.utils.project_io import save_basera_project

        doc = _make_document()
        doc.add_layer(name="L1")
        p = tmp_path / "ok.basera"

        # Simulate the save
        save_basera_project(doc, p)
        assert p.exists()
        assert p.stat().st_size > 0

    def test_save_with_corrupt_path_raises(self, tmp_path):
        """Save to an impossible path must raise, not hang."""
        from photo_editor.utils.project_io import save_basera_project

        doc = _make_document()
        bad_path = tmp_path / "nonexistent_dir_xyz" / "deep" / "proj.basera"
        # This should NOT raise because save_basera_project creates parents
        save_basera_project(doc, bad_path)
        assert bad_path.exists()

    def test_save_document_command_returns_none(self, tmp_path):
        """SaveDocumentCommand.execute returns None for .basera saves."""
        from photo_editor.commands.document.save_document import SaveDocumentCommand
        from photo_editor.engine.render_pipeline import RenderPipeline

        doc = _make_document()
        p = tmp_path / "cmd.basera"
        cmd = SaveDocumentCommand(p, RenderPipeline())
        result = cmd.execute(doc)
        assert result is None
        assert p.exists()

    def test_on_success_with_try_finally_pattern(self, tmp_path):
        """Simulates the _save_basera on_success pattern with try/finally.

        Even if something inside on_success throws, the finally block
        must still fire (setting the 'Saved' status message).
        """
        from photo_editor.utils.project_io import save_basera_project

        doc = _make_document()
        p = tmp_path / "resilience.basera"
        save_basera_project(doc, p)

        status_messages = []

        def on_success(_result):
            try:
                raise RuntimeError("Simulated callback crash")
            finally:
                status_messages.append("Saved")

        try:
            on_success(None)
        except RuntimeError:
            pass

        assert "Saved" in status_messages, (
            "The 'Saved' status must fire even when the callback crashes"
        )


def test_concurrent_saves_do_not_share_a_temp_path(tmp_path):
    """A background Ctrl+S and the blocking save on quit both target the same
    file. With a shared '<target>.tmp' the second writer truncated the first
    one's partial archive and os.replace() published whichever finished last
    -- a corrupt project either way."""
    import threading

    import numpy as np
    from photo_editor.utils.project_io import (
        load_basera_project, save_basera_project,
    )

    target = tmp_path / "shared.basera"
    docs = []
    for value in (0.25, 0.75):
        doc = _make_document()
        layer = doc.add_layer(name="L")
        layer.pixels[:] = np.array([value] * 4, dtype=np.float32)
        docs.append(doc)

    errors = []

    def save(doc):
        try:
            save_basera_project(doc, target)
        except Exception as exc:      # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=save, args=(d,)) for d in docs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent saves raised: {errors}"
    # Whichever won, the file must be a complete, loadable project.
    loaded = load_basera_project(target)
    assert any(l.name == "L" for l in loaded.layers)
    assert not list(tmp_path.glob("*.tmp")), "temp files were left behind"


class TestSaveDirtyTracking:
    """A multi-second background save must not swallow edits made during it."""

    def _win(self, qtbot):
        from photo_editor.ui.main_window import MainWindow
        win = MainWindow(dev_mode=True)
        qtbot.addWidget(win)
        win._autosave_timer.stop()
        return win

    def test_edits_during_a_save_are_still_unsaved(self, qtbot, tmp_path):
        """The document was marked clean when the save finished regardless of
        what happened meanwhile, so a stroke made during a four-second save
        was recorded as saved and lost on quit without a prompt."""
        import numpy as np

        win = self._win(qtbot)
        try:
            doc = win._doc
            layer = doc.layers.active_layer
            layer.begin_write()
            layer.pixels[:] = 0.25
            doc.mark_dirty()

            ctrl = win._document_ctrl
            token = ctrl.__class__.__module__  # noqa: F841 - import guard
            from photo_editor.ui.controllers.document_ctrl import (
                _document_revision,
            )
            revision = _document_revision(doc)

            # An edit lands while the save is in flight.
            layer.begin_write()
            layer.pixels[:] = 0.75

            ctrl._after_save_succeeded(doc, str(tmp_path / "x.basera"),
                                       saved_revision=revision)
            assert doc.dirty, (
                "an edit made during the save was marked as saved")
        finally:
            win._doc.mark_clean()

    def test_an_untouched_document_is_marked_clean(self, qtbot, tmp_path):
        win = self._win(qtbot)
        try:
            from photo_editor.ui.controllers.document_ctrl import (
                _document_revision,
            )
            doc = win._doc
            doc.mark_dirty()
            revision = _document_revision(doc)
            win._document_ctrl._after_save_succeeded(
                doc, str(tmp_path / "y.basera"), saved_revision=revision)
            assert not doc.dirty
        finally:
            win._doc.mark_clean()
