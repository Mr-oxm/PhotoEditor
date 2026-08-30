"""Composite-path benchmark: the headline 'can it do 20x 4K layers' number.

Run:  QT_QPA_PLATFORM=offscreen uv run python -m bench.bench_composite
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import numpy as np

from bench.harness import (
    Report, document_pixel_bytes, history_bytes, live_rss_mb,
    make_document, synthetic_image, timeit,
)

MB = 1024 * 1024


def bench_composite_scaling(report: Report, sizes, counts, iterations: int) -> None:
    """Full-document composite time as layer count grows."""
    from photo_editor.engine.render_pipeline import RenderPipeline

    print("\n=== Composite scaling (full document, float32 pipeline) ===")
    for (w, h) in sizes:
        for n in counts:
            gc.collect()
            rss_before = live_rss_mb()
            doc = make_document(n, w, h)
            pix_mb = document_pixel_bytes(doc) / MB
            rss_after_build = live_rss_mb()

            pipeline = RenderPipeline()

            def run():
                pipeline.invalidate()
                pipeline.execute(doc)

            t = timeit(f"composite_{n}L_{w}x{h}", run, iterations=iterations)
            rss_peak = live_rss_mb()
            report.add(
                t,
                layers=n, width=w, height=h,
                layer_pixel_mb=round(pix_mb, 1),
                rss_build_mb=round(rss_after_build - rss_before, 1),
                rss_render_mb=round(rss_peak - rss_after_build, 1),
            )
            print(f"  {n:>3} layers @ {w}x{h}: {t.median_ms:8.1f} ms "
                  f"({t.fps:6.2f} fps)  layer data {pix_mb:7.1f} MB  "
                  f"rss +{rss_peak - rss_before:7.1f} MB")
            del doc, pipeline
            gc.collect()


def bench_composite_to_uint8(report: Report, iterations: int) -> None:
    """The path the UI actually uses: composite + float->uint8 conversion."""
    from photo_editor.engine.render_pipeline import RenderPipeline

    print("\n=== Composite -> uint8 (UI display path) ===")
    for n in (1, 5, 20):
        doc = make_document(n, 3840, 2160)
        pipeline = RenderPipeline()

        def run():
            pipeline.invalidate()
            pipeline.execute_to_uint8(doc)

        t = timeit(f"composite_uint8_{n}L_4K", run, iterations=iterations)
        report.add(t, layers=n, width=3840, height=2160)
        print(f"  {n:>3} layers: {t.median_ms:8.1f} ms ({t.fps:6.2f} fps)")
        del doc, pipeline
        gc.collect()


def bench_blend_modes(report: Report, iterations: int) -> None:
    """Cost of the NORMAL fast path vs the generic blend path."""
    from photo_editor.blending.blending_engine import BlendingEngine
    from photo_editor.core.enums import BlendMode

    print("\n=== Blend modes (single 4K layer onto 4K canvas) ===")
    w, h = 3840, 2160
    over = synthetic_image(w, h, seed=7)
    engine = BlendingEngine()

    for mode in (BlendMode.NORMAL, BlendMode.MULTIPLY, BlendMode.OVERLAY,
                 BlendMode.SOFT_LIGHT, BlendMode.COLOR):
        canvas = np.zeros((h, w, 4), dtype=np.float32)

        def run():
            engine.blend_region_inplace(canvas, over, (0, 0), mode, 1.0, None)

        t = timeit(f"blend_{mode.name}_4K", run, iterations=iterations)
        report.add(t, blend_mode=mode.name, width=w, height=h)
        print(f"  {mode.name:<12}: {t.median_ms:8.1f} ms ({t.fps:6.2f} fps)")
        del canvas
        gc.collect()


def bench_history(report: Report) -> None:
    """Undo cost under a realistic editing session.

    The interesting number is not one snapshot of a static document -- it is
    what a sequence of single-layer edits costs, because that is what
    painting actually does. Structural sharing should make the cost scale
    with the *changed* layer, not the whole stack.
    """
    print("\n=== History: 30 single-layer edits (realistic session) ===")
    for n in (1, 5, 20):
        doc = make_document(n, 3840, 2160)
        gc.collect()
        rss_before = live_rss_mb()

        target = doc.layers.layers[n // 2]
        counter = {"i": 0}

        def one_edit():
            # Snapshot pre-edit (as tools do), then modify one layer.
            doc.save_snapshot(f"edit{counter['i']}")
            counter["i"] += 1
            target.begin_write()
            target.pixels[100:200, 100:200] = 0.5

        t = timeit(f"history_edit_{n}L_4K", one_edit, iterations=30, warmup=0)
        stats = doc.history.stats(doc.live_buffer_ids())
        retained = stats["owned_mb"]
        rss_after = live_rss_mb()
        one_layer_mb = target.pixels.nbytes / MB
        report.add(
            t, layers=n,
            history_owned_mb=round(retained, 1),
            history_referenced_mb=stats["mb"],
            states=len(doc.history.states),
            one_layer_mb=round(one_layer_mb, 1),
            budget_mb=round(doc.history._budget / MB, 1),
            rss_delta_mb=round(rss_after - rss_before, 1),
        )
        print(f"  {n:>3} layers: {t.median_ms:7.1f} ms/snapshot  "
              f"{len(doc.history.states):>3} states  "
              f"{retained:7.1f} MB owned by history "
              f"({retained / one_layer_mb:.1f} layers' worth)")
        del doc
        gc.collect()


def bench_layer_ops(report: Report, iterations: int) -> None:
    """Interactive operations: move, scale, rotate on a 4K layer."""
    from photo_editor.core.layer import Layer

    print("\n=== Non-destructive transform on one 4K layer ===")
    layer = Layer(name="t", width=3840, height=2160)
    layer.pixels = synthetic_image(3840, 2160, seed=3)
    layer.init_non_destructive()

    t = timeit("transform_scale_fast",
               lambda: layer.compute_display(scale_x=1.5, scale_y=1.5, angle=0.0, fast=True),
               iterations=iterations)
    report.add(t, op="scale", quality="fast")
    print(f"  scale 1.5x (fast)   : {t.median_ms:8.1f} ms ({t.fps:6.2f} fps)")

    t = timeit("transform_scale_quality",
               lambda: layer.compute_display(scale_x=1.5, scale_y=1.5, angle=0.0, fast=False),
               iterations=iterations)
    report.add(t, op="scale", quality="high")
    print(f"  scale 1.5x (quality): {t.median_ms:8.1f} ms ({t.fps:6.2f} fps)")

    t = timeit("transform_rotate_fast",
               lambda: layer.compute_display(scale_x=1.0, scale_y=1.0, angle=17.0, fast=True),
               iterations=iterations)
    report.add(t, op="rotate", quality="fast")
    print(f"  rotate 17deg (fast) : {t.median_ms:8.1f} ms ({t.fps:6.2f} fps)")


def bench_memory_ceiling(report: Report) -> None:
    """How much memory does a 20x4K project cost before any editing?"""
    print("\n=== Memory footprint ===")
    gc.collect()
    base = live_rss_mb()
    doc = make_document(20, 3840, 2160)
    pix_mb = document_pixel_bytes(doc) / MB
    rss = live_rss_mb()
    theoretical = 20 * 3840 * 2160 * 4 * 4 / MB
    report.add_raw(
        name="memory_20L_4K",
        layer_pixel_mb=round(pix_mb, 1),
        rss_delta_mb=round(rss - base, 1),
        theoretical_float32_mb=round(theoretical, 1),
    )
    print(f"  20x 4K layers: {pix_mb:.1f} MB of layer pixels "
          f"(RSS +{rss - base:.1f} MB)")
    print(f"  float32 RGBA is {3840 * 2160 * 4 * 4 / MB:.1f} MB per 4K layer")
    del doc
    gc.collect()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/baseline_composite.json")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--quick", action="store_true",
                    help="Skip the heaviest layer counts")
    ap.add_argument("--only", default=None,
                    help="Comma-separated subset: scaling,uint8,blend,history,transform,memory")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None

    def want(name: str) -> bool:
        return only is None or name in only

    report = Report("baseline-composite")
    print("Environment:")
    for k, v in report.env.items():
        print(f"  {k}: {v}")

    counts = (1, 2, 5, 10) if args.quick else (1, 2, 5, 10, 20)
    sizes = [(1920, 1080), (3840, 2160)]

    if want("scaling"):
        bench_composite_scaling(report, sizes, counts, args.iterations)
    if want("uint8"):
        bench_composite_to_uint8(report, args.iterations)
    if want("blend"):
        bench_blend_modes(report, args.iterations)
    if want("history"):
        bench_history(report)
    if want("transform"):
        bench_layer_ops(report, args.iterations)
    if want("memory"):
        bench_memory_ceiling(report)

    report.save(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
