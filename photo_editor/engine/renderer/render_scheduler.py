"""Render scheduler -- coalesces requests, cancels stale jobs, caps FPS.

Without this, every mouse-move would enqueue a render and the app would
spend all its time compositing frames nobody sees. The scheduler keeps only
the newest request, runs at most one render at a time, and drops results
that a newer request has already superseded.

Output buffers are double-buffered: the UI thread converts the frame it was
just handed into a QPixmap while the worker may already be compositing the
next one, so the two must never share a buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, QThreadPool, QTimer, Signal

from .render_worker import RenderCommand, RenderWorker

if TYPE_CHECKING:
    from ..render_pipeline import RenderPipeline
    from ...core.document import Document


@dataclass
class _PendingJob:
    document: Document
    command: RenderCommand
    generation_id: int
    full_refresh: bool = False


class RenderScheduler(QObject):
    """Debounces render requests and runs them one at a time."""

    # (uint8_rgba, generation_id, full_refresh, doc_width, doc_height, src_rect)
    render_ready = Signal(object, int, bool, int, int, object)
    render_error = Signal(str)

    def __init__(
        self,
        pipeline: RenderPipeline,
        interval_ms: int = 16,          # ~60 fps ceiling
        preview_max_size: int = 2048,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._interval_ms = interval_ms
        self._preview_max_size = preview_max_size
        self._generation = 0
        self._last_shown_generation = 0
        self._pending: _PendingJob | None = None
        self._in_flight = False

        # A dedicated single-slot pool. The global pool is shared with save,
        # export and other Worker.run_async jobs; renders must not queue
        # behind a 30-second export, nor run concurrently with each other
        # against the one non-thread-safe pipeline.
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(1)

        # Two output buffers, handed out alternately, so a frame being
        # converted to a QPixmap on the UI thread is never the buffer the
        # worker is writing.
        self._buffers: list[np.ndarray | None] = [None, None]
        self._buffer_index = 0
        # Document-space rectangle to render, set by the canvas as the view
        # changes. None means the whole document.
        self._roi: tuple[int, int, int, int] | None = None

        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._execute_pending)

    # ---- Public API --------------------------------------------------------

    def enqueue_render(
        self,
        document: Document,
        full_resolution: bool = False,
        full_refresh: bool = False,
    ) -> None:
        """Request a render, coalescing with any already-pending request."""
        if document is None:
            return
        self._pending = self._make_job(document, full_resolution, full_refresh)
        if not self._timer.isActive():
            self._timer.start()

    def enqueue_immediate(
        self,
        document: Document,
        full_resolution: bool = False,
        full_refresh: bool = False,
    ) -> None:
        """Request a render and start it without waiting for the timer."""
        if document is None:
            return
        job = self._make_job(document, full_resolution, full_refresh)
        self._pending = None
        self._timer.stop()
        self._run_worker(job)

    def wait_for_idle(self, timeout_ms: int = 5000) -> bool:
        """Block until no render is running. For tests and shutdown."""
        return self._pool.waitForDone(timeout_ms)

    @property
    def preview_max_size(self) -> int:
        return self._preview_max_size

    def set_preview_max_size(self, value: int) -> None:
        self._preview_max_size = max(0, int(value))

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        """Restrict rendering to a document-space rectangle (or None)."""
        self._roi = roi

    # ---- Internals ---------------------------------------------------------

    def _make_job(self, document, full_resolution, full_refresh) -> _PendingJob:
        self._generation += 1
        cmd = RenderCommand(
            document_width=document.width,
            document_height=document.height,
            preview_max_size=0 if full_resolution else self._preview_max_size,
            full_resolution=full_resolution,
            roi=None if full_resolution else self._roi,
        )
        return _PendingJob(
            document=document, command=cmd,
            generation_id=self._generation, full_refresh=full_refresh,
        )

    def _execute_pending(self) -> None:
        job = self._pending
        self._pending = None
        if job is not None:
            self._run_worker(job)

    def _is_current(self, generation_id: int) -> bool:
        """False once a newer request has arrived -- lets a worker bail out."""
        return generation_id >= self._generation

    def _next_buffer(self) -> np.ndarray | None:
        self._buffer_index ^= 1
        return self._buffers[self._buffer_index]

    def _run_worker(self, job: _PendingJob) -> None:
        self._in_flight = True
        worker = RenderWorker(
            pipeline=self._pipeline,
            document=job.document,
            command=job.command,
            generation_id=job.generation_id,
            full_refresh=job.full_refresh,
            is_current=self._is_current,
            out=self._next_buffer(),
        )
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        self._pool.start(worker)

    def _on_finished(self, rgba, generation_id: int, full_refresh: bool,
                     doc_w: int, doc_h: int, src_rect=None) -> None:
        """A render completed -- show it unless we have shown a newer one.

        Results can arrive out of order in principle, and showing an older
        frame after a newer one makes the canvas visibly rewind during a
        drag, so older results are dropped rather than painted.
        """
        self._in_flight = False
        # Keep the buffer we were handed so it can be reused next time round.
        if isinstance(rgba, np.ndarray):
            self._buffers[self._buffer_index] = rgba
        if generation_id >= self._last_shown_generation:
            self._last_shown_generation = generation_id
            self.render_ready.emit(rgba, generation_id, full_refresh,
                                   doc_w, doc_h, src_rect)
        # A request that arrived while this one was running is still pending.
        if self._pending is not None and not self._timer.isActive():
            self._timer.start()

    def _on_error(self, message: str) -> None:
        self._in_flight = False
        self.render_error.emit(message)
