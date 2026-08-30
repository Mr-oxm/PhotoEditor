"""Spike: does thread-parallel tile compositing actually scale under the GIL?

NumPy releases the GIL for large ufunc loops, so band-parallel compositing
should scale. This measures it rather than assuming it.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

VIEW_W, VIEW_H = 1600, 1000
N_LAYERS = 20


def t(label, fn, n=7):
    fn()
    ts = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - s) * 1000)
    ms = float(np.median(ts))
    print(f"  {label:<52} {ms:8.2f} ms  ({1000 / ms:6.1f} fps)")
    return ms


def blend_over(dst, src):
    inv_a = np.subtract(np.float32(1.0), src[..., 3:4])
    np.multiply(dst, inv_a, out=dst)
    np.add(dst, src, out=dst)


def main():
    print(f"Compositing {N_LAYERS} layers at {VIEW_W}x{VIEW_H}, "
          f"{os.cpu_count()} cores\n")
    rng = np.random.default_rng(0)
    layers = [rng.random((VIEW_H, VIEW_W, 4), dtype=np.float32)
              for _ in range(N_LAYERS)]
    for l in layers:
        l[..., 3] *= 0.8
    canvas = np.zeros((VIEW_H, VIEW_W, 4), np.float32)

    def serial():
        canvas.fill(0)
        for l in layers:
            blend_over(canvas, l)
    base = t("serial (1 thread)", serial)

    print()
    for nw in (2, 4, 6, 8, 10, 14):
        bands = []
        step = (VIEW_H + nw - 1) // nw
        for i in range(nw):
            y0, y1 = i * step, min(VIEW_H, (i + 1) * step)
            if y1 > y0:
                bands.append((y0, y1))
        pool = ThreadPoolExecutor(max_workers=nw)

        def band_job(band):
            y0, y1 = band
            c = canvas[y0:y1]
            c.fill(0)
            for l in layers:
                blend_over(c, l[y0:y1])

        def parallel():
            list(pool.map(band_job, bands))

        ms = t(f"band-parallel, {nw:>2} threads", parallel)
        print(f"      speedup {base / ms:5.2f}x   efficiency "
              f"{base / ms / nw * 100:5.1f}%")
        pool.shutdown()

    # Also check: does the GIL hurt for SMALL tiles (more Python overhead)?
    print("\n--- tile size sensitivity (8 threads) ---")
    pool = ThreadPoolExecutor(max_workers=8)
    for tile_h in (32, 64, 125, 250, 500):
        bands = [(y, min(VIEW_H, y + tile_h)) for y in range(0, VIEW_H, tile_h)]

        def band_job(band):
            y0, y1 = band
            c = canvas[y0:y1]
            c.fill(0)
            for l in layers:
                blend_over(c, l[y0:y1])

        def parallel():
            list(pool.map(band_job, bands))

        ms = t(f"tile height {tile_h:>3} ({len(bands):>2} tiles)", parallel)
        print(f"      speedup {base / ms:5.2f}x")
    pool.shutdown()


if __name__ == "__main__":
    main()
