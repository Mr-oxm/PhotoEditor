"""The headline benchmark: what does an interactive frame actually cost?

Models the real UI condition -- a 4K document displayed in a ~1600x1000
viewport -- rather than compositing at document resolution, which is what
the original renderer did regardless of zoom.
"""
from __future__ import annotations

import argparse
import gc
import time

import numpy as np

from bench.harness import Report, live_rss_mb, make_document, timeit
from photo_editor.engine.render_pipeline import RenderPipeline, level_size

MB = 1 << 20
VIEWPORT = 2048   # longest side the canvas needs for a ~1600x1000 widget


def run(report: Report, counts, iterations: int) -> None:
    print(f"\n=== Interactive preview: 4K document, preview_max_size={VIEWPORT} ===")
    print(f"  {'layers':>6} {'full-res':>11} {'preview':>11} {'speedup':>8} "
          f"{'cache':>10} {'level':>6}")
    print("  " + "-" * 60)
    for n in counts:
        doc = make_document(n, 3840, 2160)

        pipe = RenderPipeline(cache_budget_mb=1024)
        def full():
            # Force a recomposite while KEEPING the layer cache warm --
            # that is the real interactive condition (one layer changed,
            # the rest are unchanged and cached).
            pipe._planar_valid = False
            pipe._uint8_valid = False
            pipe.execute_to_uint8(doc, level=0)
        pipe.execute_to_uint8(doc, level=0)
        t_full = timeit(f"full_{n}L", full, iterations=iterations)
        pipe.shutdown()
        del pipe
        gc.collect()

        pipe = RenderPipeline(cache_budget_mb=1024)
        level = pipe.preview_level(doc, VIEWPORT)
        def preview():
            pipe._planar_valid = False
            pipe._uint8_valid = False
            pipe.execute_to_uint8(doc, level=level)
        pipe.execute_to_uint8(doc, level=level)
        pipe.layer_cache.reset_stats()
        t_prev = timeit(f"preview_{n}L", preview, iterations=iterations)
        stats = pipe.layer_cache.stats()
        pipe.shutdown()

        report.add(t_prev, layers=n, mode="preview", level=level,
                   cache_mb=stats["mb"], hit_rate=stats["hit_rate"])
        report.add(t_full, layers=n, mode="full")
        ow, oh = level_size(3840, 2160, level)
        print(f"  {n:>6} {t_full.median_ms:9.1f} ms {t_prev.median_ms:9.1f} ms "
              f"{t_full.median_ms / t_prev.median_ms:7.1f}x "
              f"{stats['mb']:8.0f} MB {level:>4} ({ow}x{oh})")
        del doc, pipe
        gc.collect()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/preview.json")
    ap.add_argument("--iterations", type=int, default=5)
    args = ap.parse_args()
    report = Report("preview")
    run(report, (1, 5, 10, 20, 30), args.iterations)
    report.save(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
