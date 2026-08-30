"""Interactive tool latency: cost per mouse-move event on a 4K layer.

Mouse-move events are not coalesced before reaching the tools, so every
cost here is paid 60-120 times a second during a stroke.
"""
from __future__ import annotations

import time

import numpy as np

from bench.harness import Report, make_document, synthetic_image, timeit


def _doc_with_selection(active: bool):
    doc = make_document(1, 3840, 2160)
    doc.layers.active_index = 0
    if active:
        doc.selection.select_rect(500, 500, 2000, 1200)
    return doc


def bench_stroke(report: Report, tool_cls, label: str, iterations: int = 30):
    for sel in (False, True):
        doc = _doc_with_selection(sel)
        tool = tool_cls()
        tool.on_press(doc, 800, 700, 1.0)
        state = {"x": 800, "y": 700}

        def move():
            state["x"] += 7
            state["y"] += 3
            tool.on_move(doc, state["x"], state["y"], 1.0)

        t = timeit(f"{label}_move_sel{int(sel)}", move,
                   iterations=iterations, warmup=3)
        tool.on_release(doc, state["x"], state["y"])
        report.add(t, tool=label, selection=sel)
        print(f"  {label:<14} selection={'yes' if sel else 'no ':<3} "
              f"{t.median_ms:7.2f} ms/event  "
              f"({1000 / t.median_ms:6.0f} events/s sustainable)")
        del doc, tool


def bench_transform(report: Report):
    from photo_editor.tools.transform_tool import TransformTool
    doc = make_document(1, 3840, 2160)
    doc.layers.active_index = 0
    tool = TransformTool()
    tool.on_press(doc, 100, 100, 1.0)
    state = {"x": 100}

    def move():
        state["x"] += 5
        tool.on_move(doc, state["x"], 400, 1.0)

    t = timeit("transform_move", move, iterations=10, warmup=2)
    report.add(t, tool="transform")
    print(f"  {'transform':<14} {'':<12} {t.median_ms:7.2f} ms/event  "
          f"({1000 / t.median_ms:6.0f} events/s sustainable)")


def main() -> int:
    from photo_editor.tools.brush import BrushTool
    from photo_editor.tools.eraser import EraserTool

    report = Report("tools")
    print("\n=== Per-mouse-move cost on a 4K layer ===")
    bench_stroke(report, BrushTool, "brush")
    bench_stroke(report, EraserTool, "eraser")
    bench_transform(report)
    report.save("bench/results/tools.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
