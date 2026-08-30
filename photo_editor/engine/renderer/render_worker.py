"""Background render worker -- runs compositing off the UI thread.

Threading contract
------------------
Exactly one worker runs at a time, on a dedicated single-slot thread pool.
The previous implementation dispatched to ``QThreadPool.globalInstance()``
with no concurrency limit, so several workers could run against the *same*
``RenderPipeline`` at once, all mutating its output buffer, tile cache and
compositor scratch. That is a data race that tears frames; it is now
structurally impossible.

Stale jobs are cancelled cooperatively rather than run to completion, so a
fast-moving drag does not leave cores busy producing frames nobody will see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

if TYPE_CHECKING:
    from ..render_pipeline import RenderPipeline
    from ...core.document import Document


@dataclass
class RenderCommand:
    """Immutable render request."""

    document_width: int
    document_height: int
    preview_max_size: int   # longest side of the preview, 0 = full resolution
    full_resolution: bool   # True = export, False = interactive preview
    # Document-space rectangle to composite, or None for the whole document.
    roi: tuple[int, int, int, int] | None = None


class _RenderWorkerSignals(QObject):
    """Signals emitted when a render completes (delivered on the UI thread)."""

    # (uint8_rgba, generation_id, full_refresh, doc_width, doc_height, src_rect)
    finished = Signal(object, int, bool, int, int, object)
    error = Signal(str)


class RenderWorker(QRunnable):
    """Runs a single render job."""

    def __init__(
        self,
        pipeline: RenderPipeline,
        document: Document,
        command: RenderCommand,
        generation_id: int,
        full_refresh: bool = False,
        is_current=None,
        out: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._document = document
        self._command = command
        self._generation_id = generation_id
        self._full_refresh = full_refresh
        # Callable returning False once a newer request has superseded this
        # one, checked before the expensive work starts.
        self._is_current = is_current
        self._out = out
        self.signals = _RenderWorkerSignals()
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            if self._is_current is not None and not self._is_current(
                    self._generation_id):
                return  # Superseded before we started -- drop it.
            result, src_rect = self._do_render()
            if self._is_current is not None and not self._is_current(
                    self._generation_id):
                return  # Superseded while rendering -- do not paint a stale frame.
            self.signals.finished.emit(
                result, self._generation_id, self._full_refresh,
                self._command.document_width, self._command.document_height,
                src_rect,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            self.signals.error.emit(str(exc))

    def _do_render(self):
        """Composite at the level appropriate for this request.

        Returns (rgba, src_rect) where src_rect is the document-space
        rectangle the buffer covers, or None when it covers the whole
        document. The canvas needs it to place the frame correctly.
        """
        from ..render_pipeline import level_roi_to_document
        if self._command.full_resolution or self._command.preview_max_size <= 0:
            level = 0
        else:
            level = self._pipeline.preview_level(
                self._document, self._command.preview_max_size)
        roi = None if self._command.full_resolution else self._command.roi
        rgba = self._pipeline.execute_to_uint8(
            self._document, level=level, out=self._out, roi=roi)
        lroi = self._pipeline._result_roi
        src = level_roi_to_document(lroi, level) if lroi is not None else None
        return rgba, src
