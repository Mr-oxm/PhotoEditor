"""Probe: planar (4,H,W) vs interleaved (H,W,4) for the 'over' operator.

Planar layout makes the alpha plane contiguous, so broadcasting it across
the colour planes has a contiguous inner loop -- no weight materialisation
needed at all.
"""
from __future__ import annotations
import time
import numpy as np
import cv2

H, W = 2160, 3840


def t(label, fn, n=7):
    fn()
    ts = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - s) * 1000)
    ms = float(np.median(ts))
    print(f"  {label:<58} {ms:8.2f} ms")
    return ms


def main():
    rng = np.random.default_rng(0)
    inter_b = rng.random((H, W, 4), dtype=np.float32)
    inter_o = rng.random((H, W, 4), dtype=np.float32)
    plan_b = np.ascontiguousarray(inter_b.transpose(2, 0, 1))
    plan_o = np.ascontiguousarray(inter_o.transpose(2, 0, 1))
    print(f"4K RGBA float32 = {inter_b.nbytes/1e6:.0f} MB\n")

    print("--- planar alpha access ---")
    t("planar: src[3] is a contiguous (H,W) view (free)",
      lambda: plan_o[3])
    t("planar: np.subtract(1.0, src[3])  [contiguous alloc]",
      lambda: np.subtract(np.float32(1.0), plan_o[3]))
    inv = np.empty((H, W), np.float32)
    t("planar: np.subtract(1.0, src[3], out=inv)",
      lambda: np.subtract(np.float32(1.0), plan_o[3], out=inv))
    t("interleaved: ascontiguousarray(src[...,3])  [gather]",
      lambda: np.ascontiguousarray(inter_o[..., 3]))

    print("\n--- premultiplied 'over': dst = src + dst*(1-a) ---")
    pb = plan_b.copy()

    def planar_over():
        np.subtract(np.float32(1.0), plan_o[3], out=inv)
        np.multiply(pb, inv, out=pb)      # (4,H,W) * (H,W) broadcast
        np.add(pb, plan_o, out=pb)
    ms_planar = t("PLANAR premul over (no weight materialisation)", planar_over)

    ib = inter_b.copy()
    inv4 = np.empty((H, W, 4), np.float32)

    def inter_over():
        np.subtract(np.float32(1.0),
                    np.broadcast_to(inter_o[..., 3:4], (H, W, 4)), out=inv4)
        np.multiply(ib, inv4, out=ib)
        np.add(ib, inter_o, out=ib)
    ms_inter = t("INTERLEAVED premul over (numpy broadcast weight)", inter_over)

    ib2 = inter_b.copy()
    a_c = np.ascontiguousarray(inter_o[..., 3])

    def inter_over_cv():
        i = cv2.merge([a_c, a_c, a_c, a_c])
        np.subtract(np.float32(1.0), i, out=i)
        cv2.multiply(ib2, i, ib2)
        cv2.add(ib2, inter_o, ib2)
    ms_cv = t("INTERLEAVED premul over (cv2.merge weight, alpha precached)",
              inter_over_cv)

    print("\n--- per-plane loop (explicit, avoids broadcast entirely) ---")
    pb2 = plan_b.copy()

    def planar_loop():
        np.subtract(np.float32(1.0), plan_o[3], out=inv)
        for c in range(4):
            np.multiply(pb2[c], inv, out=pb2[c])
            np.add(pb2[c], plan_o[c], out=pb2[c])
    t("PLANAR per-plane loop", planar_loop)

    print("\n--- layout conversion cost (paid once per layer, cached) ---")
    t("interleaved -> planar (transpose + copy)",
      lambda: np.ascontiguousarray(inter_b.transpose(2, 0, 1)))
    t("cv2.split (interleaved -> 4 planes)", lambda: cv2.split(inter_b))
    t("cv2.merge (4 planes -> interleaved)",
      lambda: cv2.merge([plan_b[0], plan_b[1], plan_b[2], plan_b[3]]))

    print("\n--- generic blend mode (MULTIPLY) in each layout ---")
    pb3 = plan_b.copy()

    def planar_multiply():
        np.subtract(np.float32(1.0), plan_o[3], out=inv)
        for c in range(3):
            np.multiply(pb3[c], plan_o[c], out=pb3[c])
    t("PLANAR multiply on 3 colour planes", planar_multiply)

    ib3 = inter_b.copy()

    def inter_multiply():
        np.multiply(ib3[..., :3], inter_o[..., :3], out=ib3[..., :3])
    t("INTERLEAVED multiply on [..., :3] (strided)", inter_multiply)

    print("\n" + "=" * 70)
    print(f"  planar      : {ms_planar:6.2f} ms")
    print(f"  interleaved : {ms_inter:6.2f} ms  ({ms_inter/ms_planar:.1f}x slower)")
    print(f"  cv2 weight  : {ms_cv:6.2f} ms  ({ms_cv/ms_planar:.1f}x slower)")
    print(f"  current impl: 230.00 ms  ({230/ms_planar:.0f}x slower than planar)")
    print("=" * 70)


if __name__ == "__main__":
    main()
