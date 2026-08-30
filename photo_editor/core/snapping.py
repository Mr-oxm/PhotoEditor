"""Smart snapping for the Move tool.

While a layer is being dragged, its edges and centre are compared against
the same features of the canvas, of the other visible layers, and of any
guides. When one comes within a threshold, the drag offset is nudged so
they line up exactly, and the matching alignment line is reported so the
canvas can draw it.

Design notes
------------
* The threshold is expressed in **screen** pixels and converted to document
  space by the caller, so snapping feels the same at every zoom level.
  A fixed document-space threshold would be unusable when zoomed out (every
  candidate within reach) and useless when zoomed in.
* Each axis snaps independently: a layer can align its left edge to one
  target while its vertical centre aligns to another.
* Snapping resolves to the *nearest* candidate, and equal distances prefer
  the canvas over layers over guides, so the result is stable rather than
  flipping between coincident targets mid-drag.
* This module is pure geometry with no Qt and no document mutation, so the
  behaviour can be tested exhaustively without an event loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

Rect = tuple[float, float, float, float]      # (x, y, w, h)


class SnapSource(IntEnum):
    """Priority order when two candidates are equidistant."""

    CANVAS = 0
    LAYER = 1
    GUIDE = 2


@dataclass(frozen=True)
class SnapCandidate:
    """One position on one axis that a dragged edge can align to."""

    position: float
    source: SnapSource
    # Extent of the thing that produced it, used to draw a line only as long
    # as the relationship it represents. None means "spans the canvas".
    span: tuple[float, float] | None = None


@dataclass
class SnapLine:
    """An alignment line to draw, in document coordinates."""

    vertical: bool
    position: float
    start: float
    end: float
    source: SnapSource


@dataclass
class SnapResult:
    """Outcome of a snap query."""

    dx: float = 0.0
    dy: float = 0.0
    lines: list[SnapLine] = field(default_factory=list)

    @property
    def snapped(self) -> bool:
        return bool(self.lines)


def _edges(rect: Rect) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return ((left, centre_x, right), (top, centre_y, bottom))."""
    x, y, w, h = rect
    return ((x, x + w / 2.0, x + w), (y, y + h / 2.0, y + h))


def collect_candidates(
    doc,
    moving_ids: set[str],
    guides=None,
    include_layers: bool = True,
) -> tuple[list[SnapCandidate], list[SnapCandidate]]:
    """Build the vertical and horizontal snap candidates for *doc*.

    *moving_ids* are excluded: a layer must not snap to itself. *guides* is
    passed in rather than read off the document because guides are view
    state and live on the window.
    """
    from .enums import LayerType

    vertical: list[SnapCandidate] = []      # x positions
    horizontal: list[SnapCandidate] = []    # y positions

    w, h = float(doc.width), float(doc.height)
    for pos in (0.0, w / 2.0, w):
        vertical.append(SnapCandidate(pos, SnapSource.CANVAS))
    for pos in (0.0, h / 2.0, h):
        horizontal.append(SnapCandidate(pos, SnapSource.CANVAS))

    if include_layers:
        for layer in doc.layers:
            if layer.id in moving_ids or not layer.visible:
                continue
            if layer.layer_type in (LayerType.ADJUSTMENT, LayerType.FILTER,
                                    LayerType.MASK):
                continue
            lx, ly = layer.position
            lw, lh = float(layer.width), float(layer.height)
            if lw <= 0 or lh <= 0:
                continue
            xs, ys = _edges((float(lx), float(ly), lw, lh))
            y_span = (float(ly), float(ly) + lh)
            x_span = (float(lx), float(lx) + lw)
            for pos in xs:
                vertical.append(SnapCandidate(pos, SnapSource.LAYER, y_span))
            for pos in ys:
                horizontal.append(SnapCandidate(pos, SnapSource.LAYER, x_span))

    if guides is None:
        guides = getattr(doc, "guides", ()) or ()
    for guide in guides:
        position = float(getattr(guide, "position", 0.0))
        if _guide_is_horizontal(getattr(guide, "orientation", None)):
            horizontal.append(SnapCandidate(position, SnapSource.GUIDE))
        else:
            vertical.append(SnapCandidate(position, SnapSource.GUIDE))

    return vertical, horizontal


def _guide_is_horizontal(orientation) -> bool:
    """True for Qt.Orientation.Horizontal (value 1).

    Qt.Orientation is a *flag* enum in PySide6: ``int(o)`` raises TypeError
    and ``o == 1`` is False, so only ``.value`` is reliable. Plain ints and
    strings are tolerated so the engine stays testable without Qt.
    """
    if orientation is None:
        return False
    if isinstance(orientation, str):
        return orientation.lower().startswith("h")
    value = getattr(orientation, "value", orientation)
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False


def _best_on_axis(
    moving: tuple[float, float, float],
    candidates: list[SnapCandidate],
    threshold: float,
) -> tuple[float, SnapCandidate, float] | None:
    """Nearest (offset, candidate, aligned position) within *threshold*."""
    best: tuple[float, SnapCandidate, float] | None = None
    best_distance = threshold
    best_source = SnapSource.GUIDE
    for edge in moving:
        for candidate in candidates:
            distance = abs(candidate.position - edge)
            if distance > threshold:
                continue
            better = (
                distance < best_distance - 1e-9
                or (abs(distance - best_distance) <= 1e-9
                    and candidate.source < best_source)
            )
            if better:
                best_distance = distance
                best_source = candidate.source
                best = (candidate.position - edge, candidate, candidate.position)
    return best


def snap_rect(
    rect: Rect,
    vertical: list[SnapCandidate],
    horizontal: list[SnapCandidate],
    threshold: float,
) -> SnapResult:
    """Nudge *rect* so its nearest edge or centre aligns, per axis."""
    result = SnapResult()
    if threshold <= 0:
        return result

    xs, ys = _edges(rect)
    x, y, w, h = rect

    vbest = _best_on_axis(xs, vertical, threshold)
    if vbest is not None:
        offset, candidate, position = vbest
        result.dx = offset
        span = candidate.span or (0.0, 0.0)
        start = min(span[0], y) if candidate.span else y
        end = max(span[1], y + h) if candidate.span else y + h
        result.lines.append(SnapLine(
            vertical=True, position=position,
            start=start, end=end, source=candidate.source))

    hbest = _best_on_axis(ys, horizontal, threshold)
    if hbest is not None:
        offset, candidate, position = hbest
        result.dy = offset
        span = candidate.span or (0.0, 0.0)
        start = min(span[0], x) if candidate.span else x
        end = max(span[1], x + w) if candidate.span else x + w
        result.lines.append(SnapLine(
            vertical=False, position=position,
            start=start, end=end, source=candidate.source))

    return result


class SnapEngine:
    """Holds snap settings and the candidates gathered for a drag.

    Candidates are collected once when a drag begins, not per mouse-move:
    the other layers cannot move while this one is being dragged, and
    rebuilding the list per event would reintroduce exactly the kind of
    per-event O(layers) work the performance pass removed.
    """

    def __init__(self, enabled: bool = True, threshold_px: float = 8.0) -> None:
        self.enabled = enabled
        self.threshold_px = threshold_px
        self._vertical: list[SnapCandidate] = []
        self._horizontal: list[SnapCandidate] = []
        self._active = False

    def begin(self, doc, moving_ids: set[str], guides=None) -> None:
        if not self.enabled or doc is None:
            self._active = False
            return
        self._vertical, self._horizontal = collect_candidates(
            doc, moving_ids, guides=guides)
        self._active = True

    def end(self) -> None:
        self._active = False
        self._vertical = []
        self._horizontal = []

    @property
    def active(self) -> bool:
        return self._active and self.enabled

    def snap(self, rect: Rect, zoom: float = 1.0) -> SnapResult:
        """Snap *rect*, with the threshold expressed in screen pixels."""
        if not self.active:
            return SnapResult()
        threshold = self.threshold_px / max(zoom, 1e-6)
        return snap_rect(rect, self._vertical, self._horizontal, threshold)
