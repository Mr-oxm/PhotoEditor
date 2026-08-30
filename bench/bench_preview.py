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
# The document's on-screen size: a 4K document at fit zoom in a ~1600x1000
# widget is displayed about 1536 px across, which is what the renderer
# should target. 3840 would be 100% zoom.
VIEWPORT = 1536


def run(report: Report, counts, iterations: int) -> None:
    print(f"\n=== 4K document, preview_max_size={VIEWPORT} ===")
    print(f"  {'layers':>6} {'full-res':>11} {'rebuild':>11} {'drag':>11} "
          f"{'fps':>6} {'cache':>10} {'lvl':>4}")
    print("  " + "-" * 68)
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
        del pipe
        gc.collect()

        # Interactive: dragging the top layer, with the sandwich cache on.
        pipe = RenderPipeline(cache_budget_mb=1024)
        focus = doc.layers.layers[-1]
        pipe.begin_interaction(focus.id)
        def drag():
            focus.position = (focus.position[0] + 1, focus.position[1])
            pipe.invalidate(focus.id)
            pipe.execute_to_uint8(doc, level=level)
        drag(); drag()
        pipe.sandwich.reset_stats()
        t_drag = timeit(f"drag_{n}L", drag, iterations=iterations)
        sw = pipe.sandwich.stats()
        pipe.shutdown()

        report.add(t_prev, layers=n, mode="preview", level=level,
                   cache_mb=stats["mb"], hit_rate=stats["hit_rate"])
        report.add(t_drag, layers=n, mode="drag",
                   sandwich_hit_rate=sw["hit_rate"], sandwich_mb=sw["mb"])
        report.add(t_full, layers=n, mode="full")
        ow, oh = level_size(3840, 2160, level)
        print(f"  {n:>6} {t_full.median_ms:9.1f} ms {t_prev.median_ms:9.1f} ms "
              f"{t_drag.median_ms:8.1f} ms {1000/t_drag.median_ms:6.0f} "
              f"{stats['mb']:7.0f} MB {level:>3}")
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
