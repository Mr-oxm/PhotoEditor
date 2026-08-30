"""Micro-probe: decompose the cost of one 4K NORMAL blend.

Isolates each numpy operation in `_normal_inplace` so we can see whether
the cost is strided access, temporaries, or the division.
"""
from __future__ import annotations
import time
import numpy as np

W, H = 3840, 2160
N = 5


def t(label, fn, n=N):
    fn()  # warmup
    ts = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - s) * 1000)
    ms = float(np.median(ts))
    print(f"  {label:<52} {ms:8.2f} ms")
    return ms


def main():
    base = np.random.random((H, W, 4)).astype(np.float32)
    over = np.random.random((H, W, 4)).astype(np.float32)
    print(f"4K RGBA float32 buffer = {base.nbytes / 1024 / 1024:.1f} MB\n")

    print("--- raw numpy primitives at 4K ---")
    t("copy whole (H,W,4) contiguous", lambda: base.copy())
    t("base[..., :3] * 2.0   (strided read, alloc)", lambda: base[..., :3] * 2.0)
    t("base * 2.0            (contig read, alloc)", lambda: base * 2.0)
    t("base *= 1.0000001     (contig in-place)", lambda: base.__imul__(np.float32(1.0000001)))
    tmp3 = np.empty((H, W, 3), np.float32)
    t("np.multiply(base[...,:3], 2.0, out=contig)",
      lambda: np.multiply(base[..., :3], 2.0, out=tmp3))
    t("np.any(base[..., 3:4])  (full scan)", lambda: np.any(base[..., 3:4]))
    t("np.any(base[..., 3])    (full scan)", lambda: np.any(base[..., 3]))

    print("\n--- current _normal_inplace (as implemented) ---")
    from photo_editor.blending.blending_engine import _normal_inplace

    def cur():
        b = base.copy()
        _normal_inplace(b, over[..., :3], over[..., 3:4])
    t("copy + _normal_inplace", cur)

    b2 = base.copy()
    over_rgb = over[..., :3]
    over_a = over[..., 3:4]
    t("_normal_inplace only (no copy)", lambda: _normal_inplace(b2, over_rgb, over_a))

    print("\n--- premultiplied-alpha alternative (contiguous, no division) ---")
    # Premultiplied 'over' operator: out = over + base * (1 - over_a)
    base_pm = np.random.random((H, W, 4)).astype(np.float32)
    over_pm = np.random.random((H, W, 4)).astype(np.float32)

    def pm_naive():
        inv = 1.0 - over_pm[..., 3:4]
        return over_pm + base_pm * inv

    t("premul naive (broadcast alpha, allocs)", pm_naive)

    # Broadcast trick: view alpha as (H, W, 1) then use out= to avoid allocation
    def pm_inplace():
        inv_a = np.subtract(np.float32(1.0), over_pm[..., 3:4])
        base_pm *= inv_a
        base_pm += over_pm
    b3 = base_pm.copy()

    def pm_inplace2():
        inv_a = np.subtract(np.float32(1.0), over_pm[..., 3:4])
        np.multiply(b3, inv_a, out=b3)
        np.add(b3, over_pm, out=b3)
    t("premul in-place (2 ops, broadcast alpha)", pm_inplace2)

    print("\n--- reshaped-to-2D premul (helps SIMD?) ---")
    b4 = base_pm.copy()
    flat_b = b4.reshape(-1, 4)
    flat_o = over_pm.reshape(-1, 4)

    def pm_flat():
        inv_a = np.subtract(np.float32(1.0), flat_o[:, 3:4])
        np.multiply(flat_b, inv_a, out=flat_b)
        np.add(flat_b, flat_o, out=flat_b)
    t("premul flat (N,4)", pm_flat)

    print("\n--- uint8 storage variants ---")
    b_u8 = (np.random.random((H, W, 4)) * 255).astype(np.uint8)
    o_u8 = (np.random.random((H, W, 4)) * 255).astype(np.uint8)
    print(f"  uint8 4K RGBA = {b_u8.nbytes / 1024 / 1024:.1f} MB")
    t("uint8 -> float32 convert (alloc)", lambda: b_u8.astype(np.float32))
    out_f = np.empty((H, W, 4), np.float32)
    t("uint8 -> float32 convert (out=)", lambda: np.copyto(out_f, b_u8, casting="unsafe"))

    print("\n--- OpenCV alternatives ---")
    import cv2
    print(f"  cv2 threads: {cv2.getNumThreads()}")
    t("cv2.addWeighted float32 4ch", lambda: cv2.addWeighted(base, 0.5, over, 0.5, 0.0))
    t("cv2.multiply float32 4ch", lambda: cv2.multiply(base, over))
    dst = np.empty_like(base)
    t("cv2.multiply float32 4ch (dst=)", lambda: cv2.multiply(base, over, dst))
    t("cv2.addWeighted uint8 4ch", lambda: cv2.addWeighted(b_u8, 0.5, o_u8, 0.5, 0.0))
    t("cv2.resize 4K->1080p INTER_AREA f32",
      lambda: cv2.resize(base, (1920, 1080), interpolation=cv2.INTER_AREA))
    t("cv2.resize 4K->1080p INTER_AREA u8",
      lambda: cv2.resize(b_u8, (1920, 1080), interpolation=cv2.INTER_AREA))
    t("cv2.resize 4K->1080p INTER_NEAREST u8",
      lambda: cv2.resize(b_u8, (1920, 1080), interpolation=cv2.INTER_NEAREST))

    print("\n--- viewport-sized work (1600x1000) for reference ---")
    vw, vh = 1600, 1000
    vb = np.random.random((vh, vw, 4)).astype(np.float32)
    vo = np.random.random((vh, vw, 4)).astype(np.float32)

    def vp_premul():
        inv_a = np.subtract(np.float32(1.0), vo[..., 3:4])
        np.multiply(vb, inv_a, out=vb)
        np.add(vb, vo, out=vb)
    t("premul in-place at viewport size", vp_premul)
    print(f"  -> 20 such layers = {t('  (silent)', vp_premul, n=3) * 20:.1f} ms/frame")


if __name__ == "__main__":
    main()
