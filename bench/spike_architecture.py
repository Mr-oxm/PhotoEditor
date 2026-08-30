"""Architecture spike: prove the target is reachable before committing.

Models the proposed pipeline:
  * premultiplied uint8 layer storage + mip pyramid
  * composite at VIEWPORT resolution using the appropriate mip level
  * only the visible document region
  * 'sandwich' caching (under-cache + active layer + over-cache)

Compares against the current full-res float32 straight-alpha path.
"""
from __future__ import annotations

import time
import numpy as np
import cv2

DOC_W, DOC_H = 3840, 2160
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
    print(f"  {label:<56} {ms:8.2f} ms  ({1000 / ms if ms else 0:6.1f} fps)")
    return ms


def build_pyramid(rgba_u8_premul, levels=5):
    """Mip pyramid of premultiplied uint8, each level half the previous."""
    pyr = [rgba_u8_premul]
    cur = rgba_u8_premul
    for _ in range(levels - 1):
        h, w = cur.shape[:2]
        if w < 8 or h < 8:
            break
        cur = cv2.resize(cur, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        pyr.append(cur)
    return pyr


def premultiply_u8(rgba_f32):
    """straight float32 -> premultiplied uint8."""
    a = rgba_f32[..., 3:4]
    out = np.empty(rgba_f32.shape, np.uint8)
    np.multiply(rgba_f32, 255.0, out=out, casting="unsafe")
    pm = rgba_f32.copy()
    pm[..., :3] *= a
    np.multiply(pm, 255.0, out=out, casting="unsafe")
    return out


def blend_over_premul_u16(dst_u16, src_u8):
    """Premultiplied 'over' in uint16 fixed point: dst = src + dst*(1-a)/255."""
    inv = np.subtract(np.uint16(255), src_u8[..., 3:4].astype(np.uint16))
    dst_u16 *= inv
    dst_u16 //= 255
    dst_u16 += src_u8
    return dst_u16


def blend_over_premul_f32(dst, src):
    """Premultiplied 'over', contiguous float32: dst = src + dst*(1-a)."""
    inv_a = np.subtract(np.float32(1.0), src[..., 3:4])
    np.multiply(dst, inv_a, out=dst)
    np.add(dst, src, out=dst)


def main():
    print(f"Document {DOC_W}x{DOC_H}, viewport {VIEW_W}x{VIEW_H}, "
          f"{N_LAYERS} layers\n")

    # ---- Build synthetic layers -------------------------------------------
    print("Building layers...")
    rng = np.random.default_rng(0)
    layers_f32 = []
    for i in range(N_LAYERS):
        img = rng.random((DOC_H, DOC_W, 4), dtype=np.float32)
        img[..., 3] = 0.8
        layers_f32.append(img)

    print("\n=== Storage cost ===")
    f32_mb = sum(l.nbytes for l in layers_f32) / 1024 / 1024
    print(f"  current: float32 straight   {f32_mb:9.1f} MB")

    t0 = time.perf_counter()
    layers_pm = [premultiply_u8(l) for l in layers_f32]
    print(f"  premultiply+u8 conversion:  {(time.perf_counter() - t0) * 1000:.0f} ms "
          f"for {N_LAYERS} layers (one-time, per layer edit)")
    u8_mb = sum(l.nbytes for l in layers_pm) / 1024 / 1024
    print(f"  proposed: premul uint8      {u8_mb:9.1f} MB  "
          f"({f32_mb / u8_mb:.1f}x smaller)")

    t0 = time.perf_counter()
    pyramids = [build_pyramid(l) for l in layers_pm]
    build_ms = (time.perf_counter() - t0) * 1000
    pyr_mb = sum(lvl.nbytes for p in pyramids for lvl in p) / 1024 / 1024
    print(f"  proposed: + mip pyramid     {pyr_mb:9.1f} MB  "
          f"({f32_mb / pyr_mb:.1f}x smaller than today)")
    print(f"  pyramid build: {build_ms:.0f} ms for {N_LAYERS} layers "
          f"({build_ms / N_LAYERS:.1f} ms/layer, one-time)")
    for i, lvl in enumerate(pyramids[0]):
        print(f"    L{i}: {lvl.shape[1]}x{lvl.shape[0]}")

    # ---- A: current-style full-res float32 straight-alpha ------------------
    print("\n=== A. Current approach: full-res float32, strided RGB ops ===")
    from photo_editor.blending.blending_engine import _normal_inplace

    def current_full():
        canvas = np.zeros((DOC_H, DOC_W, 4), np.float32)
        for l in layers_f32[:3]:      # only 3 — 20 would take ~12 s
            _normal_inplace(canvas, l[..., :3], l[..., 3:4])
    ms3 = t("3 layers @ 4K, current blend", current_full, n=2)
    print(f"    -> extrapolated to {N_LAYERS} layers: "
          f"{ms3 / 3 * N_LAYERS:.0f} ms ({1000 / (ms3 / 3 * N_LAYERS):.2f} fps)")

    # ---- B: premultiplied float32 at full res ------------------------------
    print("\n=== B. Premultiplied float32, contiguous, still full-res ===")
    layers_pm_f32 = [(l.astype(np.float32) / 255.0) for l in layers_pm]

    def premul_full():
        canvas = np.zeros((DOC_H, DOC_W, 4), np.float32)
        for l in layers_pm_f32:
            blend_over_premul_f32(canvas, l)
    t(f"{N_LAYERS} layers @ 4K, premul f32", premul_full, n=3)

    # ---- C: viewport-resolution composite from mip level -------------------
    print("\n=== C. + viewport-resolution compositing (mip level 1, 1:2) ===")
    # At 'fit' zoom (~0.42 for 4K in a 1600x1000 view) we sample mip level 1
    # (1920x1080) and scale down; here we model compositing at viewport size.
    view_layers = [
        cv2.resize(l, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)
        for l in layers_pm
    ]
    view_f32 = [(l.astype(np.float32) / 255.0) for l in view_layers]

    def viewport_f32():
        canvas = np.zeros((VIEW_H, VIEW_W, 4), np.float32)
        for l in view_f32:
            blend_over_premul_f32(canvas, l)
    t(f"{N_LAYERS} layers @ viewport, premul f32", viewport_f32)

    def viewport_u16():
        canvas = np.zeros((VIEW_H, VIEW_W, 4), np.uint16)
        for l in view_layers:
            blend_over_premul_u16(canvas, l)
    t(f"{N_LAYERS} layers @ viewport, premul u16 fixed-point", viewport_u16)

    # ---- D: sandwich caching -----------------------------------------------
    print("\n=== D. + sandwich cache (drag one layer: under + active + over) ===")
    under = np.zeros((VIEW_H, VIEW_W, 4), np.float32)
    for l in view_f32[:10]:
        blend_over_premul_f32(under, l)
    over = np.zeros((VIEW_H, VIEW_W, 4), np.float32)
    for l in view_f32[11:]:
        blend_over_premul_f32(over, l)
    active = view_f32[10]
    scratch = np.empty_like(under)

    def sandwich():
        np.copyto(scratch, under)
        blend_over_premul_f32(scratch, active)
        blend_over_premul_f32(scratch, over)
    ms_sw = t("drag frame: 1 copy + 2 blends @ viewport", sandwich)

    # ---- E: uint8 output + QImage-ready ------------------------------------
    print("\n=== E. Full interactive frame (sandwich + uint8 output) ===")
    out_u8 = np.empty((VIEW_H, VIEW_W, 4), np.uint8)

    def full_frame():
        np.copyto(scratch, under)
        blend_over_premul_f32(scratch, active)
        blend_over_premul_f32(scratch, over)
        np.multiply(scratch, 255.0, out=out_u8, casting="unsafe")
    ms_frame = t("complete drag frame -> uint8 RGBA", full_frame)

    # ---- Summary ------------------------------------------------------------
    baseline = ms3 / 3 * N_LAYERS
    print("\n" + "=" * 72)
    print(f"  Baseline (current, {N_LAYERS}x4K):        {baseline:9.1f} ms  "
          f"({1000 / baseline:6.2f} fps)")
    print(f"  Spiked interactive drag frame:      {ms_frame:9.2f} ms  "
          f"({1000 / ms_frame:6.1f} fps)")
    print(f"  Speedup:                            {baseline / ms_frame:9.0f}x")
    print(f"  Memory: {f32_mb:.0f} MB -> {pyr_mb:.0f} MB "
          f"({f32_mb / pyr_mb:.1f}x reduction, and full res retained)")
    print("=" * 72)


if __name__ == "__main__":
    main()
