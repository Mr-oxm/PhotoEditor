"""Golden-buffer management for render-fidelity testing.

Goldens are captured ONCE from the pre-overhaul compositor and committed.
Every later change must reproduce them within a tight tolerance, which is
what makes aggressive rewrites of the render core safe.

Regenerate deliberately (and review the diff!) with:
    QT_QPA_PLATFORM=offscreen uv run python -m tests.fidelity.golden --write
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "goldens")

# Tolerance: goldens are float32 composites. 1/255 is one 8-bit level --
# the finest difference that can survive to the screen or an exported PNG.
TOLERANCE = 1.0 / 255.0


def golden_path(name: str) -> str:
    return os.path.join(GOLDEN_DIR, f"{name}.npz")


def render_scene(doc) -> np.ndarray:
    """Composite *doc* through the production pipeline."""
    from photo_editor.engine.render_pipeline import RenderPipeline
    pipeline = RenderPipeline()
    pipeline.invalidate()
    return np.ascontiguousarray(pipeline.execute(doc), dtype=np.float32)


def write_goldens() -> int:
    from tests.fidelity.scenes import all_scenes

    os.makedirs(GOLDEN_DIR, exist_ok=True)
    scenes = all_scenes()
    for name, factory in sorted(scenes.items()):
        doc = factory()
        result = render_scene(doc)
        # float16 keeps the committed goldens small.  Its worst-case
        # spacing near 1.0 is 4.9e-4, ~8x finer than the 1/255 tolerance,
        # so quantising cannot mask a real regression.
        np.savez_compressed(golden_path(name), data=result.astype(np.float16))
        print(f"  wrote {name:<28} {result.shape} "
              f"range [{result.min():.4f}, {result.max():.4f}]")
    print(f"\n{len(scenes)} goldens -> {GOLDEN_DIR}")
    return 0


def compare(name: str, actual: np.ndarray) -> tuple[bool, str]:
    """Compare *actual* against the stored golden for *name*."""
    path = golden_path(name)
    if not os.path.exists(path):
        return False, f"no golden for '{name}' (run tests.fidelity.golden --write)"
    with np.load(path) as npz:
        expected = npz["data"].astype(np.float32)
    if expected.shape != actual.shape:
        return False, (f"shape {actual.shape} != golden {expected.shape}")
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    max_diff = float(diff.max())
    if max_diff <= TOLERANCE:
        return True, f"max diff {max_diff:.6f}"
    n_bad = int((diff > TOLERANCE).sum())
    idx = np.unravel_index(int(diff.argmax()), diff.shape)
    return False, (
        f"max abs diff {max_diff:.6f} > tolerance {TOLERANCE:.6f}; "
        f"{n_bad} of {diff.size} components differ; "
        f"worst at {idx}: got {actual[idx]:.6f}, expected {expected[idx]:.6f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="(Re)generate goldens from the current code")
    args = ap.parse_args()
    if args.write:
        return write_goldens()
    print("Nothing to do. Pass --write to regenerate goldens.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
