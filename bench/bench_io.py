"""Project save/load benchmark, and a comparison of pixel encodings.

The .basera format stores raw float32 pixels and DEFLATEs them on the UI
thread. Float32 image data has noisy mantissa bits, so it is both 4x larger
than it needs to be and close to incompressible -- the worst combination
for a codec.
"""
from __future__ import annotations

import io
import os
import time
import zipfile

import numpy as np

from bench.harness import synthetic_image


def t(label, fn, n=3):
    fn()
    ts = []
    for _ in range(n):
        s = time.perf_counter()
        r = fn()
        ts.append((time.perf_counter() - s) * 1000)
    return float(np.median(ts)), r


def main():
    W, H = 3840, 2160
    img = synthetic_image(W, H, seed=1)
    print(f"One 4K RGBA layer: float32 = {img.nbytes/1e6:.0f} MB\n")

    print(f"  {'encoding':<40} {'time':>9} {'size':>9}  {'vs f32':>7}")
    print("  " + "-" * 70)

    def enc_f32_deflate():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            b = io.BytesIO(); np.save(b, img, allow_pickle=False)
            z.writestr("a", b.getvalue())
        return len(buf.getvalue())
    ms, size = t("f32", enc_f32_deflate, n=1)
    base = size
    print(f"  {'float32 + DEFLATE-1 (current)':<40} {ms:7.0f} ms {size/1e6:7.1f} MB {1.0:6.2f}x")

    def enc_f32_stored():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            b = io.BytesIO(); np.save(b, img, allow_pickle=False)
            z.writestr("a", b.getvalue())
        return len(buf.getvalue())
    ms, size = t("f32s", enc_f32_stored, n=1)
    print(f"  {'float32, no compression':<40} {ms:7.0f} ms {size/1e6:7.1f} MB {size/base:6.2f}x")

    u16 = np.empty(img.shape, np.uint16)
    def to_u16():
        np.multiply(img, 65535.0, out=u16, casting="unsafe")
    ms_conv, _ = t("conv", to_u16, n=3)
    print(f"  {'  (float32 -> uint16 conversion)':<40} {ms_conv:7.0f} ms")

    def enc_u16_deflate():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            b = io.BytesIO(); np.save(b, u16, allow_pickle=False)
            z.writestr("a", b.getvalue())
        return len(buf.getvalue())
    ms, size = t("u16", enc_u16_deflate, n=1)
    print(f"  {'uint16 + DEFLATE-1':<40} {ms:7.0f} ms {size/1e6:7.1f} MB {size/base:6.2f}x")

    def enc_u16_stored():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            b = io.BytesIO(); np.save(b, u16, allow_pickle=False)
            z.writestr("a", b.getvalue())
        return len(buf.getvalue())
    ms, size = t("u16s", enc_u16_stored, n=1)
    print(f"  {'uint16, no compression':<40} {ms:7.0f} ms {size/1e6:7.1f} MB {size/base:6.2f}x")

    u8 = np.empty(img.shape, np.uint8)
    np.multiply(img, 255.0, out=u8, casting="unsafe")
    def enc_u8_deflate():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            b = io.BytesIO(); np.save(b, u8, allow_pickle=False)
            z.writestr("a", b.getvalue())
        return len(buf.getvalue())
    ms, size = t("u8", enc_u8_deflate, n=1)
    print(f"  {'uint8 + DEFLATE-1':<40} {ms:7.0f} ms {size/1e6:7.1f} MB {size/base:6.2f}x")

    # PNG via OpenCV -- a real image codec on image data
    import cv2
    def enc_png():
        ok, arr = cv2.imencode(".png", u16[..., [2, 1, 0, 3]],
                               [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return arr.nbytes
    ms, size = t("png", enc_png, n=1)
    print(f"  {'uint16 PNG (cv2, level 1)':<40} {ms:7.0f} ms {size/1e6:7.1f} MB {size/base:6.2f}x")

    print("\n  Fidelity of uint16 round-trip:")
    back = u16.astype(np.float32) / 65535.0
    print(f"    max abs error = {np.abs(back - img).max():.8f} "
          f"({np.abs(back - img).max() * 255:.5f} of an 8-bit level)")
    back8 = u8.astype(np.float32) / 255.0
    print(f"    uint8 for comparison = {np.abs(back8 - img).max():.8f} "
          f"({np.abs(back8 - img).max() * 255:.3f} of an 8-bit level)")

    print("\n  Projected for a 20-layer 4K project:")
    for label, per_ms, per_mb in (
        ("float32 + DEFLATE-1 (current)", 0, 0),
    ):
        pass


if __name__ == "__main__":
    main()
