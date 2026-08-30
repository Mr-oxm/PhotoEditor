"""Before/after: interleaved BlendingEngine vs planar blend_planar_region."""
from __future__ import annotations
import time
import numpy as np

from bench.harness import synthetic_image
from photo_editor.blending.blending_engine import BlendingEngine
from photo_editor.blending.planar import blend_planar_region, to_planar
from photo_editor.core.enums import BlendMode

W, H = 3840, 2160


def t(fn, n=3):
    fn()
    ts = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - s) * 1000)
    return float(np.median(ts))


def main():
    over_i = synthetic_image(W, H, seed=7)
    over_p = to_planar(over_i)
    engine = BlendingEngine()
    print(f"Single 4K layer onto a 4K canvas\n")
    print(f"  {'mode':<14} {'interleaved':>12} {'planar':>10} {'speedup':>9}")
    print("  " + "-" * 48)
    total_i = total_p = 0.0
    for mode in (BlendMode.NORMAL, BlendMode.MULTIPLY, BlendMode.SCREEN,
                 BlendMode.OVERLAY, BlendMode.SOFT_LIGHT, BlendMode.DIFFERENCE,
                 BlendMode.COLOR, BlendMode.LUMINOSITY):
        ci = np.zeros((H, W, 4), np.float32)
        mi = t(lambda: engine.blend_region_inplace(
            ci, over_i, (0, 0), mode, 1.0, None))
        del ci
        cp = np.zeros((4, H, W), np.float32)
        mp = t(lambda: blend_planar_region(cp, over_p, (0, 0), mode, 1.0, None))
        del cp
        total_i += mi
        total_p += mp
        print(f"  {mode.name:<14} {mi:9.1f} ms {mp:8.1f} ms {mi/mp:8.1f}x")
    print("  " + "-" * 48)
    print(f"  {'TOTAL':<14} {total_i:9.1f} ms {total_p:8.1f} ms "
          f"{total_i/total_p:8.1f}x")


if __name__ == "__main__":
    main()
