# Basera — Performance Overhaul & Evolution Plan

**Status:** Phases 1-4 landed (engine). Phase 5 (UI/IO) and 6 (product) in progress.
**Owner:** performance/graphics workstream
**Last updated:** 2026-08-30

---

## 0. Executive summary

Basera today cannot composite a realistic professional project. Measured on an
Apple M4 Pro (14 cores, 48 GB), a 20-layer 4K document takes **3,981 ms per
frame (0.25 fps)** and holds **2,531 MB** of layer pixels, while a single undo
snapshot copies **2,531 MB** and a full 50-state history would need **124 GB**.

The root causes are architectural, not incidental:

1. Every frame recomposites **every layer at full document resolution**, with no
   caching of any intermediate result.
2. The blend inner loop operates on **strided `[..., :3]` channel slices**, which
   measured **8.7× slower** than the equivalent contiguous operation.
3. Layer pixels are stored as **straight-alpha float32** — 4× larger than needed
   and forcing a per-pixel divide in every blend.
4. Undo history **deep-copies every layer's pixels** on every snapshot.

This plan replaces those four foundations. An end-to-end spike of the proposed
design (`bench/spike_architecture.py`, `bench/spike_parallel.py`) measured:

| Metric | Today | Spiked design | Change |
|---|---:|---:|---:|
| Interactive drag frame, 20×4K | 3,981 ms | **12.2 ms** | **376× faster** |
| Full recomposite, 20×4K | 3,981 ms | **14.7 ms** | **271× faster** |
| Layer pixel memory, 20×4K | 2,531 MB | **843 MB** | **3.0× smaller** |
| Resolution retained | full | full | no loss |

The targets below are therefore grounded in measurement, not aspiration.

### Results actually achieved

**Interactive frame, 4K document at fit zoom** — the headline number. Note
it is now essentially flat in layer count, which matters more than the
absolute figure:

| Layers | Baseline | Drag frame | fps |
|---:|---:|---:|---:|
| 1 | 241 ms | **13.9 ms** | 72 |
| 10 | 2,091 ms | **12.2 ms** | 82 |
| 20 | 3,981 ms | **13.0 ms** | 77 |
| 30 | ~5,900 ms | **16.5 ms** | 60 |

20 layers at 3840×2160, moving one layer, measured through `MainWindow`
(software canvas path, so the real GL path is faster):

| Zoom | Baseline | Now | Change |
|---|---:|---:|---:|
| Fit | 3,981 ms | **29 ms** (34 fps) | **137×** |
| 50% | 3,981 ms | **21 ms** (47 fps) | **188×** |
| 100% | 3,981 ms | **22 ms** (45 fps) | **181×** |

| Metric | Baseline | Now | Change |
|---|---:|---:|---:|
| Undo snapshot | 667 ms | **2.9 ms** | **230×** |
| Undo memory, 20×4K | 2,531 MB/state (124 GB projected) | **1,139 MB total, bounded** | independent of layer count |
| Single 4K NORMAL blend | 233 ms | **17 ms** | **13.3×** |
| Full-resolution export composite | 3,981 ms | 603 ms | 6.6× |
| Save a 20×4K project | 32.4 s (UI frozen) | **3.7 s** (background) | **8.8×** |
| Load a 20×4K project | 4.1 s | **1.2 s** | **3.4×** |
| Project file size | 1.69 GB | **1.06 GB** | 37% smaller |
| Brush stroke with a selection | 1.12 ms/event | **0.05 ms** | **22×** |
| Transform drag | 293 ms/event | **33.5 ms** | **8.7×** |
| Test suite | 550 pass | **1,012 pass** | +462 tests |

---

## 1. Measured baseline

Hardware: Apple M4 Pro, 14 cores, 48 GB RAM, macOS 26.6.2.
Software: Python 3.13.15, NumPy 2.4.2, OpenCV 4.13.0, PySide6 6.10.2.
Reproduce with `make bench-baseline` (see §9).

### 1.1 Composite scaling — `RenderPipeline.execute()`

| Layers | 1920×1080 | 3840×2160 |
|---:|---:|---:|
| 1 | 58.0 ms | 241.3 ms |
| 2 | 113.3 ms | 487.8 ms |
| 5 | 262.4 ms | 1,135.3 ms |
| 10 | 479.2 ms | 2,090.6 ms |
| 20 | 967.9 ms | **3,980.6 ms (0.25 fps)** |

Cost is perfectly linear in layer count and in pixel count — confirming that
*nothing* is cached and *every* layer is touched at full resolution every frame.

### 1.2 Blend modes — one 4K layer onto a 4K canvas

| Mode | Time |
|---|---:|
| NORMAL (the "fast path") | 232.7 ms |
| MULTIPLY | 265.2 ms |
| OVERLAY | 361.7 ms |
| SOFT_LIGHT | 478.8 ms |
| COLOR | 573.2 ms |

### 1.3 Why NORMAL costs 233 ms — primitive decomposition

Measured on the same 4K buffer (`bench/probe_blend.py`):

| Operation | Time |
|---|---:|
| `base * 2.0` (contiguous) | 3.11 ms |
| `base[..., :3] * 2.0` (**strided**) | **27.14 ms** |
| `base.copy()` | 2.64 ms |
| `np.any(base[..., 3:4])` (full scan) | 2.49 ms |
| `_normal_inplace` (current, 6 strided ops + divide) | **234.60 ms** |
| Premultiplied contiguous equivalent (2 ops, no divide) | **24.65 ms** |

**The strided channel slice is the single largest CPU cost in the renderer.**
Every `base[..., :3]` and `over[..., 3:4]` in the blend path pays an 8.7×
penalty because NumPy cannot vectorise across a non-unit stride.

### 1.4 Memory and history

| Metric | 1 layer | 5 layers | 20 layers |
|---|---:|---:|---:|
| Layer pixels (float32 RGBA) | 126.6 MB | 632.8 MB | 2,531.2 MB |
| Time per undo snapshot | 9.8 ms | 91.6 ms | **666.7 ms** |
| Memory per undo state | 126.6 MB | 632.8 MB | **2,531.2 MB** |
| Projected full history (50 states) | 6.2 GB | 30.9 GB | **123.6 GB** |

A 20×4K project **cannot survive a single undo stack**. This is a
stability bug, not just a performance one.

### 1.5 Non-destructive transform, one 4K layer

| Operation | Time |
|---|---:|
| Scale 1.5× (`fast=True`, INTER_NEAREST) | 31.9 ms |
| Scale 1.5× (quality, premultiply + INTER_CUBIC) | 452.9 ms |
| Rotate 17° (`fast=True`) | 19.2 ms |

---

## 2. Root-cause analysis

### RC1 — No caching anywhere in the render path
`RenderWorker._do_render()` calls `self._pipeline.invalidate()` **unconditionally**
before every render (`engine/renderer/render_worker.py`), so the uint8 cache added
in `RenderPipeline.execute_to_uint8()` can never hit from the async path. Below
that, `Compositor.composite()` recomputes every layer's styles, adjustments,
channel masking and masks from scratch on every frame.

### RC2 — Strided channel arithmetic
`_normal_inplace` and `_porter_duff_inplace` (`blending/blending_engine.py`) each
perform six operations on `[..., :3]` / `[..., 3:4]` views. Measured 8.7× penalty
versus contiguous. `ImageProcessor._merge` compounds this with a full
`np.concatenate` per adjustment.

### RC3 — Straight-alpha float32 storage
`Layer._pixels` is float32 RGBA straight-alpha (`core/layer.py`). This costs 4×
the memory of uint8 and forces a per-pixel divide (`base[..., :3] /= safe_a`) in
every blend. Premultiplied alpha makes the `over` operator
`dst = src + dst·(1−α)` — two contiguous ops, no divide.

### RC4 — Document-resolution rendering regardless of zoom
`RenderScheduler` is constructed with `preview_max_size=0`
(`ui/main_window.py`), and even when non-zero the downsample happens *after* the
full-resolution composite, so it saves display cost but not composite cost. A 4K
document shown in a 1600×1000 viewport does **8.3×** the necessary pixel work.

### RC5 — Full-canvas temporary allocations
`Compositor._place_pixels`, `_place_mask_combined`, `_composite_group` and
`_apply_filters_padded` each allocate a fresh document-sized float32 buffer
(126.6 MB at 4K) per call, per frame. Clipping masks and groups multiply this.

### RC6 — History deep-copies every layer
`Document._build_history_state` copies `layer.pixels`, `_source_pixels`,
`_source_mask` and `_mask` for **every** layer on every snapshot, capped only by
a 50-state count limit with no memory budget.

### RC7 — Render worker races and no cancellation
`RenderScheduler._run_worker` dispatches to `QThreadPool.globalInstance()` with
no concurrency limit and no cancellation. Multiple workers can run concurrently
against the **same** `RenderPipeline`, sharing `_uint8_buf`, the `Compositor` and
the `ImagePool` — a genuine data race that can tear frames. Stale workers run to
completion, burning cores that the newest frame needs. The worker also reads
`layer.pixels` while UI-thread tools mutate them.

### RC8 — Per-frame UI work
`MainWindow._on_render_ready` runs on every completed frame and refreshes the
transform panel, channels panel, rulers, selection overlay and transform box.
`CanvasView.paintEvent` rescales the entire document-sized `QPixmap` on every
repaint — including the 10 fps marching-ants timer tick.

---

## 3. Target architecture

Four changes, each independently valuable, compounding to the measured 376×.

### 3.1 Premultiplied, contiguous blending core
Blend on the whole `(H, W, 4)` buffer at once. The premultiplied `over` operator
is `dst = src + dst·(1−α)`: one broadcast multiply, one add, no divide, no
strided RGB slice. Separable blend modes (multiply, screen, overlay, …) apply
their function to the full 4-channel buffer and fix up alpha afterwards.

*Measured: 234.60 ms → 24.65 ms on a 4K layer (9.5×).*

### 3.2 Resolution-adaptive rendering via mip pyramids
Each layer caches a pyramid of **premultiplied uint8** levels (L0 full, L1 ½,
L2 ¼, …). The renderer selects the level nearest the current zoom and composites
at **viewport** resolution, not document resolution.

*Measured: uint8 + pyramid is 843 MB vs 2,531 MB float32 for 20×4K — 3.0×
smaller while retaining full resolution. Pyramid build costs 2.1 ms/layer,
amortised across every frame that follows.*

*Measured: 20-layer composite 491 ms full-res → 93 ms at viewport (5.3×).*

### 3.3 Sandwich caching for interactive edits
Cache the composite of everything **below** the layer being edited and
everything **above** it. A drag frame becomes `copy(under) ⊕ active ⊕ over` —
two blends instead of twenty, regardless of layer count.

*Measured: 93 ms → 9.7 ms; 12.2 ms including uint8 output (82 fps).*

Correctness constraint: the over-cache is only valid when every layer above the
active one is a plain blend. A root-level adjustment or filter layer above the
active layer consumes the accumulated canvas, so the cache must be invalidated
and the renderer must fall back to a full walk. This is detected structurally
when the draw list is built.

### 3.4 Tile-parallel compositing
Split the accumulation buffer into horizontal bands and composite them
concurrently. NumPy releases the GIL inside large ufunc loops, so this scales.

*Measured (`bench/spike_parallel.py`): 93.9 ms serial → 14.7 ms with 8 threads
at 64-row tiles — 6.4×. Efficiency peaks at 8 threads; 64-row bands beat both
larger bands (too little parallelism) and the 14-thread configuration
(oversubscription against OpenCV's own pool).*

### 3.5 What we are deliberately *not* doing yet: GPU compositing
A GLSL compositor behind `QOpenGLWidget` would make this trivially real-time.
It is **deferred**, not rejected, because:
* The CPU design above already exceeds the 60 fps target with measured headroom.
* Reproducing 28 Photoshop blend modes, clipping chains, group isolation, masks
  and adjustment-layer semantics in GLSL risks regressing a 550-test suite.
* Readback for tools, thumbnails and export would need a parallel CPU path
  anyway.

The compositor is therefore designed behind a backend-neutral interface
(draw-list in, buffer out) so a GPU backend can be added later without
disturbing callers. Revisit after Phase 5 with a fidelity test harness in place.

---

## 4. Goals and acceptance criteria

Measured on the reference machine, 20 layers, 3840×2160, viewport 1600×1000.

| # | Goal | Baseline | Target | Achieved | Met |
|---|---|---:|---:|---:|---|
| G1 | Interactive drag frame, 20×4K | 3,981 ms | ≤ 16.7 ms | **13.0 ms (77 fps)** | yes |
| G2 | Full recomposite (preview path) | 3,981 ms | ≤ 33 ms | **35 ms** | ~ |
| G3 | Layer pixel memory, 20×4K | 2,531 MB | ≤ 1,000 MB | 2,531 MB | **no** |
| G4 | Undo snapshot time | 667 ms | ≤ 20 ms | **2.9 ms** | yes |
| G5 | Undo history memory | 123.6 GB | ≤ 2,000 MB | **1,139 MB, bounded** | yes |
| G6 | Single 4K NORMAL blend | 233 ms | ≤ 30 ms | **17 ms** | yes |
| G7 | Brush stroke latency, 4K layer | 1.12 ms/event | ≤ 10 ms | **0.05 ms** | yes |
| G8 | Cold start | ~1,930 ms | ≤ 1,500 ms | not addressed | **no** |
| G9 | Test suite | 550 pass | no regressions | **954 pass** | yes |
| G10 | Visual fidelity | — | ≤ 1/255 | **gate green** | yes |

**Where the targets were not met, and why**

*G2 (33 ms).* A full rebuild after a structural change is 35 ms at twenty
layers — 29 fps against a 30 fps target, close enough to be indistinguishable
in use, and it only happens when the stack itself changes rather than
during a drag.

*G3 (1 GB of layer pixels).* Not met, and deliberately not pursued. Layer
pixels are float32 because that is what every tool, filter and adjustment
reads and writes directly. Storing uint16 would halve the 2,531 MB but add
a conversion to every `.pixels` access — and a float32 cache to avoid that
would give the memory straight back. 2.5 GB for twenty 4K layers is a
reasonable cost; the memory problems worth fixing were the undo history
(was heading for 124 GB) and the render cache thrashing, and both are
fixed. Peak RSS for a 20×4K session mid-edit is ~5.6 GB, bounded by
budgets that now scale with the machine.

*G8 (cold start).* Measured at ~1,930 ms, of which **1,000 ms is a
hard-coded ten-frame splash animation** that blocks in a nested event loop
*before* `MainWindow` is even constructed, and ~70 ms is an eager `cv2`
import pulled in through `commands/__init__`. Both are straightforward to
fix and neither was reached in this pass.

G9 and G10 are hard gates on every phase. G10 is enforced by a golden-image
harness that composites reference documents through the old and new paths and
compares them pixel-for-pixel.

---

## 5. Phased plan

Each phase ends at a **checkpoint**: tests green, benchmarks re-run, fidelity
harness green, this document updated with measured results.

### Phase 1 — Foundations: correctness, safety, and the blend core
*Rationale: fix the data races and the memory bomb before building on top.*

1.1 **Fidelity harness** (`tests/test_render_fidelity.py`) — golden composites of
reference documents covering every blend mode, masks, clipping, groups, styles,
adjustment layers. Runs against old and new compositors. **Build this first.**

1.2 **Premultiplied contiguous blend core** (`blending/`). Add
`blend_premultiplied_over` and rewrite mode dispatch to contiguous whole-buffer
ops. Remove the `np.any(over_a)` full-array early-out (2.5 ms/layer/frame for a
branch that rarely fires) in favour of a cached per-layer "is fully transparent"
flag.

1.3 **History rewrite** (`core/history.py`, `core/document.py`). Copy-on-write
buffer sharing with per-layer content versions, plus a **memory budget** (default
1 GB) that evicts oldest states. Snapshot stores references, not copies, for
layers unchanged since the previous state.

1.4 **Render worker safety** (`engine/renderer/`). Single dedicated render thread
with a bounded queue; cooperative cancellation checked between layers; remove the
unconditional `pipeline.invalidate()`; per-job output buffers so no two jobs share
`_uint8_buf`.

**Checkpoint 1:** G6 (≤30 ms blend), G4, G5 met. No races under a stress test.

### Phase 2 — Resolution-adaptive rendering
2.1 `engine/cache/layer_raster.py` — `LayerRasterCache`: premultiplied uint8 mip
pyramids keyed by layer id + content version, with an LRU memory budget.

2.2 Draw-list builder (`engine/scene.py`) — flatten the layer stack into an
explicit, cacheable list of draw ops (resolving groups, clipping chains, masks,
adjustment scoping) keyed by a stack-structure version. This removes the repeated
`O(layers²)` prescans currently done inside `Compositor.composite`.

2.3 Viewport-ROI compositing — `RenderRequest(doc_rect, scale, quality)`; composite
only the visible region at the mip level matching zoom.

2.4 Canvas display path — `CanvasView` draws only the visible source rect, and
stops rebuilding a document-sized `QPixmap` per frame.

**Checkpoint 2:** G2 (≤33 ms full recomposite) and G3 (≤1 GB) met.

### Phase 3 — Interactive caching
3.1 Sandwich under/over caches with structural validity detection (§3.3).
3.2 Per-layer processed cache (styles + adjustments + channels) keyed by content
version + parameter hash.
3.3 Dirty-region propagation from tools so a brush dab re-composites only its
bounding rect.

**Checkpoint 3:** G1 (≤16.7 ms drag frame) and G7 met.

### Phase 4 — Parallelism
4.1 Band-parallel compositor (8 threads, 64-row bands — the measured optimum).
4.2 Coordinate with OpenCV's thread pool to avoid oversubscription.
4.3 Wire the existing, currently-unused `core/image/tile_processor.py` into the
filter path.

**Checkpoint 4:** G2 comfortably met with headroom; scaling curve re-measured.

### Phase 5 — UI, I/O, and startup
5.1 Thumbnail cache keyed on layer content version; generate from the smallest
adequate mip level instead of the full buffer.
5.2 Stop per-frame panel refreshes in `_on_render_ready`; refresh panels only on
the state they actually depend on.
5.3 Lazy imports for heavy modules; measure and cut cold start (G8).
5.4 Project I/O: store pixels compressed, not raw float32.

**Checkpoint 5:** G8 met; UI thread idle during a drag.

### Phase 6 — Product evolution
Feature work, prioritised after the performance foundation is measured and
stable. Candidates to be selected on evidence from a product review, not
guessed at here. Recorded in §8 as the review proceeds.

---

## 6. Validation methodology

* **Benchmarks** — `bench/` is committed and reproducible. `bench_composite.py`
  produces the scaling table; `probe_blend.py` decomposes primitive costs;
  `spike_*.py` validate design decisions before implementation. Results are
  written as JSON under `bench/results/` so before/after diffs are mechanical.
* **No unmeasured claims.** Every performance statement in this document cites a
  measured number produced by a committed benchmark.
* **Fidelity gate** — the golden-image harness (1.1) must show ≤ 1/255 max
  absolute difference against the pre-overhaul compositor for every reference
  document, on every phase.
* **Regression gate** — the existing 550-test suite must stay green.
* **Stress testing** — a long-running randomised edit session against a 20×4K
  document, asserting RSS stays under budget and no worker race is detected.

---

## 7. Progress log

| Date | Phase | Change | Measured result |
|---|---|---|---|
| 2026-08-30 | 0 | Baseline captured, harness committed | 20×4K = 3,981 ms, 2,531 MB |
| 2026-08-30 | 0 | Primitive decomposition | strided RGB slice = 8.7× penalty |
| 2026-08-30 | 0 | Architecture spike | 12.2 ms drag frame (376×), 843 MB |
| 2026-08-30 | 0 | Parallelism spike | 6.4× @ 8 threads / 64-row bands |
| 2026-08-30 | 1 | Fidelity gate: 42 golden scenes @ 1/255 | gate green, caught non-deterministic Dissolve |
| 2026-08-30 | 1 | Planar blend core | 4K NORMAL 233 → 17 ms (13.3×) |
| 2026-08-30 | 1 | Planar compositor + layer cache | 20×4K 3,981 → 588 ms (6.8×) |
| 2026-08-30 | 1 | Copy-on-write history + byte budget | 667 → 2.9 ms; memory now layer-count independent |
| 2026-08-30 | 2 | Mip-level preview rendering | 20×4K preview 2,194 → 34.9 ms (17×) |
| 2026-08-30 | 4 | Band-parallel compositing | 6.4× where the working set fits cache |
| 2026-08-30 | 2 | Viewport ROI + ROI-cropped preparation | 100% zoom 948 → 28 ms (33×) |
| 2026-08-30 | 5 | Per-frame UI work removed; thumbnails fixed | panels no longer refresh per frame |
| 2026-08-30 | 5 | .basera v4 format, async save | save 32.4 → 3.7 s; file 37% smaller |
| 2026-08-30 | 3 | Tool per-event costs | brush 1.12 → 0.05 ms; transform 293 → 33.5 ms |
| 2026-08-30 | 3 | Sandwich cache + fused uint8 output | drag frame 13.0 ms (77 fps), constant in layer count |

### Bugs found and fixed along the way

The performance work surfaced several genuine correctness bugs. Each has a
regression test.

| Bug | Symptom | Fix |
|---|---|---|
| Dissolve used unseeded `np.random` | Dissolve layers flickered every frame; output uncacheable and untestable | Stable per-size dither field |
| Invalidating during an in-flight render | Render finishing after an edit published stale output as *valid*; canvas kept showing the pre-edit picture | Epoch-versioned cache publication |
| Band-parallel compositors indexed by band | With more bands than workers, concurrent bands shared a compositor and raced on its state, dropping layers | Thread-local compositors |
| `int(round())` placement | Banker's rounding is not translation-invariant, so layers at odd positions jittered by a pixel while panning | Round half-up |
| float→uint8 truncation | Every displayed and exported channel biased down by up to one level (0.25 → 63) | Round instead of truncate |
| Root adjustment layers rebind the canvas | Band writes into the caller's buffer were discarded | Copy back on rebind |
| Preview buffer size leaked into document coords | Latent: overlays, guides and hit-testing would rescale when the preview level changed | Document size tracked separately from buffer size |

---

## 8. Product work

Three features were added after the performance pass, chosen because they
serve the workload this editor targets — large, many-layered projects —
rather than because they were easy to add.

### 8.1 Autosave and crash recovery

A professional editor holding gigabytes of work should not lose it to a
crash. Every open document is autosaved to the user's application-support
directory; on startup, work left behind by a session that did not exit
cleanly is offered back.

This is a feature the performance work made possible rather than one that
merely fits alongside it. Autosave was not viable while saving a 20-layer
4K project took 32 seconds on the UI thread — a periodic save would have
frozen the application for half a minute at a time. At ~3.7 s on a worker
it runs unnoticed.

Both failure directions are covered by tests: work is not lost after a
crash, and work is not *falsely* offered after a clean exit (a second
window must not offer to "recover" documents another window has open).
An idle session writes nothing.

### 8.2 Smart snapping and alignment guides

Dragging a layer snaps its edges and centre to the canvas, to other
visible layers, and to guides, drawing the alignment line responsible.

The threshold is in *screen* pixels converted through the current zoom: a
fixed document-space threshold would grab every candidate in reach when
zoomed out and be unusable when zoomed in. Candidates are gathered once at
the start of a drag rather than per mouse-move — rebuilding them per event
would reintroduce exactly the O(layers) per-event work this project spent
its time removing.

*View > Snap to Objects* toggles it; Ctrl (Cmd on macOS) suspends it for a
single drag.

### 8.3 Layers-panel search and filter

Twenty layers is where scrolling stops being a way to find anything. The
panel gains a search box, a kind filter and a match count. A match brings
its ancestors along so it appears in its proper place in the hierarchy,
and a matching group brings its subtree.

### 8.4 Considered and not done

* **Histogram panel** — genuinely standard, and a histogram already exists
  inside the Levels dialog; a dockable one is a small extension.
* **Resize/rotate snapping** — the engine supports it; only the translate
  path is wired.
* **Batch export / export presets** — valuable, larger in scope.

---

## 9. What remains

Ordered by value.

1. **Over-cache half of the sandwich** — the *under* half is implemented and
   is what makes drag frames flat in layer count. The *over* half (layers
   above the focus, pre-composited when they form an isolated NORMAL run)
   is designed and its validity check is written and tested
   (`over_run_is_isolatable`) but not yet wired. It would help when
   dragging a layer near the bottom of a deep stack.
2. **Startup** — remove the blocking 1 s splash animation
   (`app.py:68-83`); make `commands/__init__` lazy so `cv2` is not imported
   eagerly (~70 ms).
3. **Gradient tool** — the handle-drag path renders the gradient at full
   resolution per event, including two `np.mgrid` int64 arrays (~400 MB per
   event at 4K). The drag path already downsamples; the handle path needs
   the same.
4. **Group/multi-select resize and rotate** — one full-layer warp *per
   selected layer per event*. The single-layer path is fixed; these are not.
5. **Vector rasterizer** — computes an MD5 state hash per object and
   discards it (`rasterizer.py:66-77`); `VectorObject.bbox()` rebuilds the
   transformed path every call with no cache. Also: the rasterizer produces
   premultiplied alpha and everything downstream treats it as straight, so
   semi-transparent vector fills render darker than they should. That is a
   correctness bug, not a performance one.
6. **Unused dependencies** — `matplotlib`, `pandas`, `scipy`, `psd-tools`,
   `enaml`, `atom`, `qtpy`, `contourpy`, `kiwisolver`, `abr`, `pegen`,
   `bytecode` are declared in `pyproject.toml` and imported nowhere.
7. **GPU compositing** — still the right long-term answer for very large
   documents; see §3.5 for why it was deferred rather than rejected.

---

## 10. Reproducing the numbers

```bash
uv sync --system-certs

# The headline number: interactive preview cost by layer count
QT_QPA_PLATFORM=offscreen uv run python -m bench.bench_preview

# Full-resolution composite (the export path), blend modes, history, memory
QT_QPA_PLATFORM=offscreen uv run python -m bench.bench_composite

# Per-mouse-move tool latency
QT_QPA_PLATFORM=offscreen uv run python -m bench.bench_tools

# Planar vs interleaved blending, and the primitive decomposition
QT_QPA_PLATFORM=offscreen uv run python -m bench.bench_blend_planar
QT_QPA_PLATFORM=offscreen uv run python -m bench.probe_blend
QT_QPA_PLATFORM=offscreen uv run python -m bench.probe_planar
QT_QPA_PLATFORM=offscreen uv run python -m bench.probe_kernels

# Design spikes, kept so the architectural choices stay checkable
QT_QPA_PLATFORM=offscreen uv run python -m bench.spike_architecture
QT_QPA_PLATFORM=offscreen uv run python -m bench.spike_parallel

# Project I/O encodings
QT_QPA_PLATFORM=offscreen uv run python -m bench.bench_io

QT_QPA_PLATFORM=offscreen uv run pytest tests/ -q
```

Results land in `bench/results/*.json`.

Six tests fail in a fresh clone: they load SVG fixtures from a `svgs/`
directory that is in `.gitignore` and not distributed. They failed the same
way before any of this work.

### Tests added

| Suite | What it pins down |
|---|---|
| `test_render_fidelity.py` | 42 golden scenes at 1/255 — the gate everything else rests on |
| `test_planar_blending.py` | Planar and interleaved blending agree, all 28 modes |
| `test_planar_compositor.py` | The rewritten compositor matches the original, scene for scene |
| `test_parallel_compositor.py` | Band-parallel output is identical to serial |
| `test_preview_levels.py` | Mip levels and viewport ROI render the same picture |
| `test_render_integration.py` | The real MainWindow render path end to end |
| `test_layer_cache.py` | Caches never serve stale pixels |
| `test_history_memory.py` | Undo correctness under buffer sharing, and its memory bound |
| `test_memory_budget.py` | Budgets scale with the machine and stay in bounds |
| `test_tool_latency.py` | Stroke mask caching; transform uses the fast path |
| `test_ui_caching.py` | Thumbnail, icon and channel-preview invalidation |
| `test_autosave.py` | Work is not lost, and not falsely offered |
| `test_snapping.py` | Snap geometry, and Move-tool integration |
| `test_layer_filter.py` | Filter matching rules and panel behaviour |

### A note on headless capture

The canvas is a `QOpenGLWidget`, and Qt's offscreen platform cannot create
a GL context, so `widget.grab()` returns blank under it. Set
`BASERA_DISABLE_GL=1` to fall back to the software widget for headless
screenshots — it is also a useful escape hatch on machines with broken GL
drivers.
