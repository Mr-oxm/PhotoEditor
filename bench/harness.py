"""Reusable benchmark utilities: timing, memory, synthetic documents.

Designed to run headless (QT_QPA_PLATFORM=offscreen) so the numbers are
reproducible in CI and comparable across commits.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import resource
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict

import numpy as np


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@dataclass
class Timing:
    """Result of timing a callable over several iterations."""

    name: str
    iterations: int
    times_ms: list[float] = field(default_factory=list)

    @property
    def best_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else float("nan")

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else float("nan")

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.times_ms) if self.times_ms else float("nan")

    @property
    def fps(self) -> float:
        m = self.median_ms
        return 1000.0 / m if m and m == m and m > 0 else float("inf")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "best_ms": round(self.best_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "fps": round(self.fps, 2),
        }


def timeit(name: str, fn, iterations: int = 5, warmup: int = 1) -> Timing:
    """Time *fn* with warmup, returning per-iteration wall times in ms."""
    for _ in range(warmup):
        fn()
    gc.collect()
    t = Timing(name=name, iterations=iterations)
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        t.times_ms.append((time.perf_counter() - start) * 1000.0)
    return t


@contextmanager
def timed(label: str, out: dict | None = None):
    """Context manager timing a block; prints and optionally records."""
    start = time.perf_counter()
    yield
    ms = (time.perf_counter() - start) * 1000.0
    if out is not None:
        out[label] = round(ms, 3)
    print(f"    {label}: {ms:.1f} ms")


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def rss_mb() -> float:
    """Resident set size in MiB (macOS reports bytes, Linux KiB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def live_rss_mb() -> float:
    """Current (not peak) RSS in MiB, via psutil if present else peak."""
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return rss_mb()


def document_pixel_bytes(doc) -> int:
    """Sum the bytes held by every layer's pixel/mask/source buffers."""
    total = 0
    for layer in doc.layers:
        for attr in ("_pixels", "_mask", "_source_pixels", "_source_mask"):
            buf = getattr(layer, attr, None)
            if isinstance(buf, np.ndarray):
                total += buf.nbytes
    return total


def history_bytes(doc) -> int:
    """Sum the bytes held by the undo history."""
    total = 0
    for state in doc.history.states:
        for buf in state.layer_data.values():
            if isinstance(buf, np.ndarray):
                total += buf.nbytes
    return total


# ---------------------------------------------------------------------------
# Synthetic content
# ---------------------------------------------------------------------------

def synthetic_image(width: int, height: int, seed: int = 0) -> np.ndarray:
    """Deterministic RGBA float32 image with structure (not flat noise).

    Structure matters: flat buffers can be unrealistically cache/branch
    friendly, and fully-transparent buffers hit early-out paths that
    would not fire on real photos.
    """
    rng = np.random.default_rng(seed)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    img = np.empty((height, width, 4), dtype=np.float32)
    img[..., 0] = xx * 0.7 + yy * 0.3
    img[..., 1] = yy * 0.6 + 0.2
    img[..., 2] = (1.0 - xx) * 0.5 + 0.25
    # A little noise so compressible/constant fast paths do not skew results
    img[..., :3] += rng.random((height, width, 3), dtype=np.float32) * 0.05
    np.clip(img[..., :3], 0.0, 1.0, out=img[..., :3])
    img[..., 3] = 1.0
    return img


def make_document(
    n_layers: int,
    width: int = 3840,
    height: int = 2160,
    *,
    layer_width: int | None = None,
    layer_height: int | None = None,
    opacity: float = 1.0,
    blend_mode=None,
    stagger: bool = True,
    alpha: float = 1.0,
):
    """Build a Document with *n_layers* synthetic raster layers.

    Layers are staggered in position so they only partially overlap,
    matching real multi-layer compositions rather than perfectly
    stacked full-canvas layers.
    """
    from photo_editor.core.document import Document
    from photo_editor.core.enums import BlendMode
    from photo_editor.core.layer import Layer

    lw = layer_width or width
    lh = layer_height or height

    doc = Document(width, height, name=f"bench-{n_layers}x{width}x{height}")
    # Drop the auto-created white background so layer count is exact
    doc.layers.layers.clear()

    for i in range(n_layers):
        layer = Layer(name=f"L{i}", width=lw, height=lh)
        pix = synthetic_image(lw, lh, seed=i)
        if alpha < 1.0:
            pix[..., 3] = alpha
        layer.pixels = pix
        if stagger:
            layer.position = ((i * 37) % max(1, width // 4),
                              (i * 53) % max(1, height // 4))
        layer.opacity = opacity
        if blend_mode is not None:
            layer.blend_mode = blend_mode
        doc.layers.add(layer)
    doc.layers.active_index = max(0, n_layers - 1)
    # Clear history built during construction
    doc.history.clear()
    return doc


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def environment() -> dict:
    import numpy
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": numpy.__version__,
    }
    try:
        import cv2
        info["opencv"] = cv2.__version__
        info["cv2_threads"] = cv2.getNumThreads()
    except Exception:
        pass
    try:
        import PySide6
        info["pyside6"] = PySide6.__version__
    except Exception:
        pass
    return info


class Report:
    """Collects benchmark results and writes them as JSON."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.env = environment()
        self.results: list[dict] = []

    def add(self, timing: Timing, **extra) -> Timing:
        row = timing.to_dict()
        row.update(extra)
        self.results.append(row)
        return timing

    def add_raw(self, **row) -> None:
        self.results.append(row)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(
                {"name": self.name, "env": self.env, "results": self.results},
                fh, indent=2,
            )
        print(f"\nSaved {len(self.results)} results -> {path}")
