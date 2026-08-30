"""Basera project I/O — ZIP-based container format (v3).

Architecture
------------
A .basera file is a standard ZIP archive (inspectable with any ZIP tool):

    manifest.json           – document metadata + complete ordered layer list
    layers/{id}.npy         – pixel buffer (raster / mask / shape / group layers)
    layers/{id}_mask.npy    – per-layer single-channel mask (optional)
    layers/{id}_src.npy     – non-destructive source pixels (optional)
    layers/{id}_srcmask.npy – non-destructive source mask (optional)
    selection.npy           – canvas-level selection mask (optional)

Pixel buffers are written as raw numpy binary (.npy) and compressed by the
ZIP layer (DEFLATE level-1 — fast, modest size).  No pickle, no double-copy,
no Python-version lock-in.

Vector, text, adjustment, and filter layer data lives entirely in the
manifest as JSON.  Pixel arrays for adjustment/filter layers are never stored
(they are always re-derived during rendering).

Backward compatibility
----------------------
v1 / v2 files (pickle+gzip) are transparently upgraded on load.

Why this is fast
----------------
* No _build_history_state() call (that copies every pixel array twice).
* No pickle serialisation of numpy arrays (uses the numpy native format).
* No validation re-read (just ensure manifest is parseable JSON).
* Atomic write: temp file → os.replace() — safe on all platforms.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..core.document import Document
from ..core.enums import BlendMode, LayerType
from ..core.layer import Layer
from ..core.history import HistoryState

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAGIC = "BASERA_PROJECT"
_FORMAT_VERSION = 3

_MANIFEST = "manifest.json"
_SEL_ENTRY = "selection.npy"
_LAYERS_DIR = "layers/"

# Layer types whose pixel buffers are purely derived during rendering.
# Storing them wastes space and time; they are always initialised to zeros.
_NO_PIXEL_TYPES = frozenset({LayerType.ADJUSTMENT, LayerType.FILTER})


# ---------------------------------------------------------------------------
# Save path
# ---------------------------------------------------------------------------

def save_basera_project(document: Document, path: str | Path) -> None:
    """Write the document snapshot to *path* atomically.

    The format is a ZIP archive (v3).  No history is persisted — only the
    current document state.  Writing directly from layer objects means zero
    extra copies of pixel data.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")

    try:
        _write_zip(document, tmp)
        os.replace(tmp, target)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _write_zip(document: Document, dest: Path) -> None:
    """Write all document data into a ZIP file at *dest*."""
    manifest = _build_manifest(document)

    with zipfile.ZipFile(
        dest, "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
    ) as zf:
        # 1. Manifest first — fast JSON text
        zf.writestr(_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=None))

        # 2. Pixel buffers for each layer
        for layer in document.layers:
            _write_layer_pixels(zf, layer)

        # 3. Selection mask (if active)
        sel = document.selection
        if getattr(sel, "_mask", None) is not None:
            _write_npy(zf, _SEL_ENTRY, sel._mask)


def _build_manifest(document: Document) -> dict:
    """Return the manifest dict (serialisable to JSON)."""
    layer_list = []
    for layer in document.layers:
        layer_list.append(_layer_meta(layer))

    return {
        "magic": _MAGIC,
        "version": _FORMAT_VERSION,
        "document": {
            "name": document.name,
            "width": document.width,
            "height": document.height,
            "dpi": document.dpi,
            "color_mode": document.color_mode,
            "color_profile": document.color_profile,
            "unit": document.unit,
        },
        "active_index": document.layers.active_index,
        "has_selection": getattr(document.selection, "_mask", None) is not None,
        "layers": layer_list,
    }


def _layer_meta(layer: Layer) -> dict:
    """Return a JSON-serialisable dict of all layer metadata."""
    lid = layer.id
    skip_pixels = layer.layer_type in _NO_PIXEL_TYPES

    meta: dict = {
        "id": lid,
        "name": layer.name,
        "width": layer.width,
        "height": layer.height,
        "layer_type": layer.layer_type.name,
        "opacity": layer.opacity,
        "blend_mode": layer.blend_mode.name,
        "visible": layer.visible,
        "locked": layer.locked,
        "position": list(layer.position),
        "mask_enabled": layer.mask_enabled,
        "clipping_mask": layer.clipping_mask,
        "clips_parent": layer.clips_parent,
        "parent_id": layer.parent_id,
        "children": list(layer.children),
        "mask_layers": list(layer.mask_layers),
        "ex_parent_id": layer.ex_parent_id,
        "transform_angle": layer.transform_angle,
        "transform_scale_x": layer.transform_scale_x,
        "transform_scale_y": layer.transform_scale_y,
        "transform_base_w": layer.transform_base_w,
        "transform_base_h": layer.transform_base_h,
        # Pixel presence flags
        "has_pixels": not skip_pixels,
        "has_mask": layer._mask is not None,
        "has_source": layer._source_pixels is not None,
        "has_source_mask": layer._source_mask is not None,
    }

    # --- Inline data for non-pixel layer types ---

    td = getattr(layer, "_text_data", None)
    if td is not None:
        meta["text_data"] = td.to_dict()

    if layer.adjustment is not None:
        meta["adjustment_name"] = layer.adjustment.name
        meta["adjustment_params"] = dict(layer.adjustment_params)

    vd = getattr(layer, "_vector_data", None)
    if vd is not None and hasattr(vd, "to_dict"):
        meta["vector_data"] = vd.to_dict()

    return meta


def _write_layer_pixels(zf: zipfile.ZipFile, layer: Layer) -> None:
    """Write the pixel buffers for *layer* into the ZIP archive."""
    if layer.layer_type in _NO_PIXEL_TYPES:
        return  # pixels are always zeros for these types

    lid = layer.id
    _write_npy(zf, f"{_LAYERS_DIR}{lid}.npy", layer.pixels)

    if layer._mask is not None:
        _write_npy(zf, f"{_LAYERS_DIR}{lid}_mask.npy", layer._mask)
    if layer._source_pixels is not None:
        _write_npy(zf, f"{_LAYERS_DIR}{lid}_src.npy", layer._source_pixels)
    if layer._source_mask is not None:
        _write_npy(zf, f"{_LAYERS_DIR}{lid}_srcmask.npy", layer._source_mask)


def _write_npy(zf: zipfile.ZipFile, entry_name: str, arr: np.ndarray) -> None:
    """Write a numpy array as a .npy stream into *entry_name* of the ZIP."""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    # ZipFile already applies DEFLATE compression, so we don't double-compress.
    # writestr() with bytes is efficient: it avoids a seek-back to patch lengths
    # as long as the buffer fits in memory (which it always does for layer data).
    zf.writestr(entry_name, buf.getvalue())


# ---------------------------------------------------------------------------
# Load path
# ---------------------------------------------------------------------------

def load_basera_project(path: str | Path) -> Document:
    """Load a .basera project file and return a fully restored Document.

    Accepts v3 (ZIP), v2, and v1 (pickle+gzip) files.
    """
    source = Path(path)
    if zipfile.is_zipfile(source):
        return _load_v3(source)
    return _load_legacy(source)


def _load_v3(source: Path) -> Document:
    """Load a v3 ZIP-based project file."""
    with zipfile.ZipFile(source, "r") as zf:
        try:
            manifest_bytes = zf.read(_MANIFEST)
        except KeyError:
            raise ValueError("Invalid .basera file: manifest.json not found")

        try:
            manifest = json.loads(manifest_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid .basera file: manifest.json is corrupt: {exc}") from exc

        if manifest.get("magic") != _MAGIC:
            raise ValueError("Invalid .basera file: wrong magic value")

        doc_meta = manifest.get("document", {})
        document = Document(
            width=int(doc_meta.get("width", 1)),
            height=int(doc_meta.get("height", 1)),
            name=str(doc_meta.get("name", "Untitled")),
            color_mode=str(doc_meta.get("color_mode", "RGB")),
            color_profile=str(doc_meta.get("color_profile", "sRGB IEC61966-2.1")),
            unit=str(doc_meta.get("unit", "px")),
        )
        document.dpi = int(doc_meta.get("dpi", 72))
        document.file_path = str(source)

        # Rebuild layer stack
        from ..core.layer_stack import LayerStack
        new_stack = LayerStack()

        for lmeta in manifest.get("layers", []):
            layer = _layer_from_meta(lmeta, zf)
            new_stack.add(layer)

        new_stack.active_index = int(manifest.get("active_index", 0))
        document.layers = new_stack

        # Restore selection mask
        if manifest.get("has_selection"):
            try:
                document.selection._set_mask(_read_npy(zf, _SEL_ENTRY))
            except (KeyError, Exception):
                pass

    document.history.clear()
    document.mark_clean()
    return document


def _layer_from_meta(meta: dict, zf: zipfile.ZipFile) -> Layer:
    """Reconstruct a Layer from its manifest entry and ZIP pixel data."""
    lid = meta["id"]

    layer_type = LayerType[meta["layer_type"]]
    blend_mode = BlendMode[meta["blend_mode"]]

    layer = Layer(
        name=meta["name"],
        width=int(meta["width"]),
        height=int(meta["height"]),
        layer_type=layer_type,
        id=lid,
        opacity=float(meta.get("opacity", 1.0)),
        blend_mode=blend_mode,
        visible=bool(meta.get("visible", True)),
        locked=bool(meta.get("locked", False)),
        position=tuple(meta.get("position", [0, 0])),
        mask_enabled=bool(meta.get("mask_enabled", True)),
        clipping_mask=bool(meta.get("clipping_mask", False)),
        clips_parent=bool(meta.get("clips_parent", False)),
        parent_id=meta.get("parent_id"),
        transform_angle=float(meta.get("transform_angle", 0.0)),
        transform_scale_x=float(meta.get("transform_scale_x", 1.0)),
        transform_scale_y=float(meta.get("transform_scale_y", 1.0)),
        transform_base_w=int(meta.get("transform_base_w", 0)),
        transform_base_h=int(meta.get("transform_base_h", 0)),
    )

    layer.children = list(meta.get("children", []))
    layer.mask_layers = list(meta.get("mask_layers", []))
    layer.ex_parent_id = meta.get("ex_parent_id")

    # --- Pixel data ---
    if meta.get("has_pixels", True) and layer_type not in _NO_PIXEL_TYPES:
        entry = f"{_LAYERS_DIR}{lid}.npy"
        try:
            layer.pixels = _read_npy(zf, entry)
        except (KeyError, Exception):
            pass  # leave default zeros if missing

    if meta.get("has_mask"):
        try:
            layer._mask = _read_npy(zf, f"{_LAYERS_DIR}{lid}_mask.npy")
        except (KeyError, Exception):
            pass

    if meta.get("has_source"):
        try:
            layer._source_pixels = _read_npy(zf, f"{_LAYERS_DIR}{lid}_src.npy")
        except (KeyError, Exception):
            pass

    if meta.get("has_source_mask"):
        try:
            layer._source_mask = _read_npy(zf, f"{_LAYERS_DIR}{lid}_srcmask.npy")
        except (KeyError, Exception):
            pass

    # --- Inline data ---
    td_dict = meta.get("text_data")
    if td_dict is not None:
        try:
            from ..core.text_layer import TextLayerData
            layer._text_data = TextLayerData.from_dict(td_dict)
        except Exception:
            pass

    adj_name = meta.get("adjustment_name")
    if adj_name is not None:
        try:
            from ..registries import get_adjustment_class, get_filter_name_map
            if layer_type == LayerType.FILTER:
                cls = get_filter_name_map().get(adj_name)
            else:
                cls = get_adjustment_class(adj_name)
            if cls is not None:
                layer._adjustment = cls()
                layer._adjustment_params = dict(meta.get("adjustment_params", {}))
        except Exception:
            pass

    vd_dict = meta.get("vector_data")
    if vd_dict is not None:
        try:
            from ..vector.scene import VectorLayer as VL
            layer._vector_data = VL.from_dict(vd_dict)
        except Exception:
            pass

    return layer


def _read_npy(zf: zipfile.ZipFile, entry_name: str) -> np.ndarray:
    """Read a numpy array from a ZIP entry."""
    with zf.open(entry_name) as fh:
        return np.load(io.BytesIO(fh.read()), allow_pickle=False)


# ---------------------------------------------------------------------------
# Legacy loader (v1 / v2 — pickle+gzip)
# ---------------------------------------------------------------------------

def _load_legacy(source: Path) -> Document:
    """Load a v1 or v2 pickle+gzip project file."""
    import copy
    import gzip
    import pickle

    try:
        with gzip.open(source, "rb") as fh:
            payload = pickle.load(fh)
    except (OSError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(
            "Project file is incomplete or corrupted (legacy format). "
            "Please try saving the project again."
        ) from exc

    if not isinstance(payload, dict) or payload.get("magic") != _MAGIC:
        raise ValueError("Invalid .basera file")

    meta = payload.get("document", {})
    document = Document(
        width=int(meta.get("width", 1)),
        height=int(meta.get("height", 1)),
        name=str(meta.get("name", "Untitled")),
        color_mode=str(meta.get("color_mode", "RGB")),
        color_profile=str(meta.get("color_profile", "sRGB IEC61966-2.1")),
        unit=str(meta.get("unit", "px")),
    )
    document.dpi = int(meta.get("dpi", 72))
    document.file_path = str(source)

    current_state_data = payload.get("current_state")
    if not isinstance(current_state_data, dict):
        raise ValueError("Invalid legacy .basera file: missing current_state")

    state = HistoryState(
        name=current_state_data.get("name", "Unnamed"),
        metadata=copy.deepcopy(current_state_data.get("metadata", {})),
        layer_data={
            k: v.copy() for k, v in current_state_data.get("layer_data", {}).items()
        },
    )
    document._restore(state)
    document.history.clear()
    document.mark_clean()
    return document


# ---------------------------------------------------------------------------
# Public helpers (used by validation / tests)
# ---------------------------------------------------------------------------

def load_basera_payload(path: str | Path) -> dict:
    """Return the raw manifest dict from a .basera file (any version).

    For v3 files this is the manifest.json; for legacy files it is the
    top-level payload dict.  Used by tests and diagnostic tools.

    Raises ``ValueError`` (with a friendly message) when the file is
    truncated or otherwise unreadable.
    """
    source = Path(path)
    if zipfile.is_zipfile(source):
        try:
            with zipfile.ZipFile(source, "r") as zf:
                return json.loads(zf.read(_MANIFEST))
        except (KeyError, json.JSONDecodeError, zipfile.BadZipFile, Exception) as exc:
            raise ValueError(
                f".basera file is incomplete or corrupted: {exc}"
            ) from exc

    # Legacy pickle path (v1/v2)
    import gzip
    import pickle
    try:
        with gzip.open(source, "rb") as fh:
            return pickle.load(fh)
    except (EOFError, OSError, Exception) as exc:
        raise ValueError(
            f".basera file is incomplete or corrupted: {exc}"
        ) from exc
