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
    print(f"\n=== 4K document, preview target {VIEWPORT}px (all times in ms) ===")
    print(f"  {'layers':>6} {'full-res':>9} {'rebuild':>8} "
          f"{'drag-top':>7} {'drag-mid':>7} {'drag-bot':>8} "
          f"{'mousedown':>8} {'cache':>7}")
    print("  " + "-" * 72)
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

        # Interactive drag, measured at three depths in the stack. The
        # sandwich cache covers everything BELOW the focus, so dragging the
        # top layer is its best case and dragging the bottom one its worst.
        # Reporting only the top layer would flatter the result.
        drag_times = {}
        prime_ms = None
        for label, index in (("top", -1), ("mid", n // 2), ("bottom", 0)):
            pipe = RenderPipeline(cache_budget_mb=1024)
            focus = doc.layers.layers[index]

            # Cost of entering the interaction: the under-half is composited
            # once at mouse-down. This is a real hitch the user feels.
            pipe.begin_interaction(focus.id)
            pipe.invalidate(focus.id)
            t0 = time.perf_counter()
            pipe.execute_to_uint8(doc, level=level)
            first = (time.perf_counter() - t0) * 1000
            if label == "mid":
                prime_ms = first

            def drag(_p=pipe, _f=focus):
                _f.position = (_f.position[0] + 1, _f.position[1])
                _p.invalidate(_f.id)
                _p.execute_to_uint8(doc, level=level)
            drag(); drag()
            t = timeit(f"drag_{label}_{n}L", drag, iterations=iterations)
            drag_times[label] = t.median_ms
            report.add(t, layers=n, mode=f"drag-{label}",
                       first_frame_ms=round(first, 1))
            pipe.shutdown()
            del pipe
            gc.collect()

        t_drag = type("T", (), {"median_ms": drag_times["mid"]})()
        sw = {"hit_rate": 0.0, "mb": 0.0}

        report.add(t_prev, layers=n, mode="preview", level=level,
                   cache_mb=stats["mb"], hit_rate=stats["hit_rate"])
        report.add(t_full, layers=n, mode="full")
        print(f"  {n:>6} {t_full.median_ms:9.1f} {t_prev.median_ms:8.1f} "
              f"{drag_times['top']:7.1f} {drag_times['mid']:7.1f} "
              f"{drag_times['bottom']:8.1f} {prime_ms:8.1f} "
              f"{stats['mb']:7.0f}")
        del doc
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
