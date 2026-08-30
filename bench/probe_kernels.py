"""Probe: find the fastest formulation of the straight-alpha 'over' operator.

The blend must produce identical results to the current implementation, so
we are choosing an implementation strategy, not changing the math:

    out_a   = over_a + base_a*(1 - over_a)
    out_rgb = (over_rgb*over_a + base_rgb*base_a*(1 - over_a)) / out_a
"""
from __future__ import annotations
import time
import numpy as np
import cv2

H, W = 2160, 3840


def t(label, fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - s) * 1000)
    ms = float(np.median(ts))
    print(f"  {label:<56} {ms:8.2f} ms")
    return ms


def main():
    rng = np.random.default_rng(0)
    base0 = rng.random((H, W, 4), dtype=np.float32)
    over = rng.random((H, W, 4), dtype=np.float32)
    print(f"4K RGBA float32 = {base0.nbytes/1e6:.0f} MB\n")

    print("--- weight-array materialisation strategies ---")
    a1 = over[..., 3:4]
    w4 = np.empty((H, W, 4), np.float32)
    t("broadcast_to((h,w,1)->(h,w,4)).copy()",
      lambda: np.copyto(w4, np.broadcast_to(a1, (H, W, 4))))
    t("np.repeat(a1, 4, axis=2) (alloc)", lambda: np.repeat(a1, 4, axis=2))
    a2d = np.ascontiguousarray(over[..., 3])
    t("cv2.merge([a,a,a,a])", lambda: cv2.merge([a2d, a2d, a2d, a2d]))
    dst4 = np.empty((H, W, 4), np.float32)
    t("stride-trick view + copyto", lambda: np.copyto(
        dst4, np.lib.stride_tricks.as_strided(
            a2d, (H, W, 4), (a2d.strides[0], a2d.strides[1], 0))))

    print("\n--- multiply: broadcast (h,w,1) vs contiguous (h,w,4) ---")
    b1 = base0.copy()
    t("np.multiply(b, a1, out=b)   [broadcast (h,w,1)]",
      lambda: np.multiply(b1, a1, out=b1))
    b2 = base0.copy()
    t("np.multiply(b, w4, out=b)   [contiguous (h,w,4)]",
      lambda: np.multiply(b2, w4, out=b2))
    b3 = base0.copy()
    t("cv2.multiply(b, w4, b)      [OpenCV, 4ch]",
      lambda: cv2.multiply(b3, w4, b3))
    b4 = base0.copy()
    t("b *= 1.0001                 [scalar, contiguous]",
      lambda: b4.__imul__(np.float32(1.0001)))

    print("\n--- full 'over' operator, candidate implementations ---")
    from photo_editor.blending.blending_engine import _normal_inplace

    bA = base0.copy()
    orgb = over[..., :3]
    oa = over[..., 3:4]
    t("CURRENT _normal_inplace (strided)",
      lambda: _normal_inplace(bA, orgb, oa))

    # Candidate 1: broadcast weights, 4-channel ops
    bB = base0.copy()

    def cand_broadcast():
        base_a = bB[..., 3:4].copy()
        w2 = np.subtract(np.float32(1.0), oa)
        w2 *= base_a
        out_a = np.add(oa, w2)
        np.multiply(bB, w2, out=bB)
        np.add(bB, over * oa, out=bB)
        np.divide(bB, np.maximum(out_a, 1e-10), out=bB)
        bB[..., 3:4] = out_a
    t("cand A: broadcast (h,w,1) weights on 4ch", cand_broadcast)

    # Candidate 2: materialise weights as contiguous (h,w,4)
    bC = base0.copy()
    w2_4 = np.empty((H, W, 4), np.float32)
    oa4 = np.empty((H, W, 4), np.float32)
    outa4 = np.empty((H, W, 4), np.float32)
    tmp = np.empty((H, W, 4), np.float32)

    def cand_contig():
        np.copyto(oa4, np.broadcast_to(oa, (H, W, 4)))
        np.copyto(w2_4, np.broadcast_to(bC[..., 3:4], (H, W, 4)))
        np.subtract(np.float32(1.0), oa4, out=tmp)
        np.multiply(w2_4, tmp, out=w2_4)
        np.add(oa4, w2_4, out=outa4)
        np.multiply(bC, w2_4, out=bC)
        np.multiply(over, oa4, out=tmp)
        np.add(bC, tmp, out=bC)
        np.maximum(outa4, 1e-10, out=tmp)
        np.divide(bC, tmp, out=bC)
        bC[..., 3:4] = outa4[..., 3:4]
    t("cand B: materialised contiguous (h,w,4) weights", cand_contig)

    # Candidate 3: premultiplied accumulation (canvas kept premultiplied)
    bD = base0.copy()
    over_pm = over.copy()
    over_pm[..., :3] *= over[..., 3:4]
    inv4 = np.empty((H, W, 4), np.float32)

    def cand_premul():
        np.subtract(np.float32(1.0), np.broadcast_to(oa, (H, W, 4)), out=inv4)
        np.multiply(bD, inv4, out=bD)
        np.add(bD, over_pm, out=bD)
    t("cand C: PREMULTIPLIED canvas (2 ops + weight)", cand_premul)

    # Candidate 4: premultiplied with OpenCV
    bE = base0.copy()

    def cand_premul_cv():
        np.subtract(np.float32(1.0), np.broadcast_to(oa, (H, W, 4)), out=inv4)
        cv2.multiply(bE, inv4, bE)
        cv2.add(bE, over_pm, bE)
    t("cand D: premultiplied via OpenCV (threaded)", cand_premul_cv)

    # Candidate 5: premultiplied, uint8 source -> float32 canvas via cv2
    print("\n--- premultiplied with cached uint8 layer source ---")
    over_pm_u8 = np.empty((H, W, 4), np.uint8)
    np.multiply(over_pm, 255.0, out=over_pm_u8, casting="unsafe")
    bF = base0.copy()
    srcf = np.empty((H, W, 4), np.float32)

    def cand_u8_src():
        np.multiply(over_pm_u8, np.float32(1 / 255.0), out=srcf,
                    casting="unsafe")
        np.subtract(np.float32(1.0), srcf[..., 3:4], out=inv4[..., :1])
        np.copyto(inv4, np.broadcast_to(inv4[..., :1], (H, W, 4)))
        cv2.multiply(bF, inv4, bF)
        cv2.add(bF, srcf, bF)
    t("cand E: uint8 src -> f32 + premul blend", cand_u8_src)


if __name__ == "__main__":
    main()
