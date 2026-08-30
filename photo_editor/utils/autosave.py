"""Autosave and crash recovery.

Why this exists now and not before
----------------------------------
Autosave was not viable while saving a 20-layer 4K project took 32 seconds
on the UI thread -- a periodic save would have frozen the application for
half a minute at a time, which is worse than the problem it solves. With
the v4 format that same save takes ~3.7 s on a worker thread, so it can run
quietly in the background.

How it works
------------
Every open document is autosaved to its own file under the user's
application-support directory, alongside a small JSON sidecar recording
where the document came from and when it was written. A document is only
written when it is actually dirty and has changed since the last autosave,
so an idle session does no work at all.

On startup, any sidecar whose owning session did not shut down cleanly is
offered back to the user as a recoverable document. Clean shutdown and
explicit saves remove the corresponding files, so the recovery list only
ever contains work that was genuinely lost.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

_APP_NAME = "Basera"
_SIDECAR_SUFFIX = ".recovery.json"
_DOC_SUFFIX = ".autosave.basera"

# A session marks itself live by touching this file; a stale one means the
# process died without cleaning up.
_LIVE_SUFFIX = ".live"
_STALE_AFTER_SECONDS = 60 * 60 * 6


def _config_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif os.uname().sysname == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / _APP_NAME


def recovery_dir() -> Path:
    """Directory holding autosaved documents. Created on demand."""
    override = os.environ.get("BASERA_RECOVERY_DIR")
    path = Path(override) if override else _config_dir() / "recovery"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class RecoveryEntry:
    """One recoverable document found on disk."""

    doc_path: Path          # the autosaved .basera
    sidecar: Path           # its metadata file
    name: str               # display name
    original_path: str | None   # where the user's file lives, if it had one
    saved_at: float         # unix timestamp of the autosave

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.saved_at)

    def describe_age(self) -> str:
        secs = int(self.age_seconds)
        if secs < 90:
            return "moments ago"
        if secs < 3600:
            return f"{secs // 60} minutes ago"
        if secs < 86400:
            return f"{secs // 3600} hours ago"
        return f"{secs // 86400} days ago"

    def discard(self) -> None:
        for path in (self.doc_path, self.sidecar):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


class AutosaveManager:
    """Periodically writes dirty documents to the recovery directory.

    The manager is deliberately ignorant of Qt: it exposes ``maybe_save``
    for a timer to call and ``release`` for teardown, so it can be tested
    without an event loop.
    """

    def __init__(self, session_id: str | None = None,
                 min_interval_seconds: float = 120.0) -> None:
        self._session = session_id or f"{os.getpid()}-{int(time.time())}"
        self._min_interval = min_interval_seconds
        self._last_saved: dict[str, tuple[float, int]] = {}
        self._live_marker = recovery_dir() / f"{self._session}{_LIVE_SUFFIX}"
        self._touch_live()

    # ---- Session liveness --------------------------------------------------

    def _touch_live(self) -> None:
        try:
            self._live_marker.write_text(str(time.time()))
        except OSError:
            pass

    def release(self) -> None:
        """Clean shutdown: drop this session's autosaves and marker."""
        directory = recovery_dir()
        for path in directory.glob(f"{self._session}__*"):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            self._live_marker.unlink(missing_ok=True)
        except OSError:
            pass

    # ---- Saving ------------------------------------------------------------

    def _paths_for(self, doc_key: str) -> tuple[Path, Path]:
        stem = f"{self._session}__{doc_key}"
        directory = recovery_dir()
        return (directory / f"{stem}{_DOC_SUFFIX}",
                directory / f"{stem}{_SIDECAR_SUFFIX}")

    @staticmethod
    def _fingerprint(document) -> tuple:
        """A value that changes whenever the document's content changes.

        Layer content versions are process-wide monotonic and move on every
        edit. History *depth* is not usable for this: it stops growing once
        the byte budget starts evicting states, which on a large project
        happens within a few strokes -- and autosave would then decide
        nothing had changed and never fire again for the rest of the
        session, exactly when it matters most.
        """
        versions = [getattr(l, "content_version", 0) for l in document.layers]
        return (max(versions) if versions else 0, len(list(document.layers)))

    def should_save(self, doc_key: str, document) -> bool:
        """True when *document* has unsaved changes worth autosaving.

        An idle session -- or one where the user only panned and zoomed --
        writes nothing.
        """
        if document is None or not getattr(document, "dirty", False):
            return False
        stamp = self._last_saved.get(doc_key)
        fingerprint = self._fingerprint(document)
        if stamp is None:
            return True
        last_time, last_fingerprint = stamp
        if fingerprint == last_fingerprint:
            return False
        return (time.time() - last_time) >= self._min_interval

    def save(self, doc_key: str, document) -> Path | None:
        """Write *document* to its recovery file. Returns the path written."""
        from .project_io import save_basera_project

        doc_path, sidecar = self._paths_for(doc_key)
        try:
            save_basera_project(document, doc_path)
            sidecar.write_text(json.dumps({
                "name": getattr(document, "name", "Untitled"),
                "original_path": getattr(document, "file_path", None),
                "saved_at": time.time(),
                "session": self._session,
                "doc": doc_path.name,
            }))
        except Exception:
            return None
        self._last_saved[doc_key] = (time.time(), self._fingerprint(document))
        self._touch_live()
        return doc_path

    def maybe_save(self, doc_key: str, document) -> Path | None:
        """Save only if the document warrants it."""
        if not self.should_save(doc_key, document):
            return None
        return self.save(doc_key, document)

    def forget(self, doc_key: str) -> None:
        """Drop a document's autosave, e.g. after the user saves it properly."""
        doc_path, sidecar = self._paths_for(doc_key)
        for path in (doc_path, sidecar):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self._last_saved.pop(doc_key, None)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def _session_is_live(session: str, exclude: str | None) -> bool:
    """True when another running process still owns *session*."""
    if exclude is not None and session == exclude:
        return True
    marker = recovery_dir() / f"{session}{_LIVE_SUFFIX}"
    if not marker.exists():
        return False
    try:
        age = time.time() - marker.stat().st_mtime
    except OSError:
        return False
    return age < _STALE_AFTER_SECONDS


def find_recoverable(exclude_session: str | None = None) -> list[RecoveryEntry]:
    """Autosaves left behind by sessions that did not shut down cleanly.

    Sessions still marked live are skipped, so a second window running
    alongside this one does not offer to 'recover' its own open documents.
    """
    entries: list[RecoveryEntry] = []
    directory = recovery_dir()
    for sidecar in sorted(directory.glob(f"*{_SIDECAR_SUFFIX}")):
        try:
            meta = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
        session = str(meta.get("session", ""))
        if _session_is_live(session, exclude_session):
            continue
        doc_path = directory / str(meta.get("doc", ""))
        if not doc_path.exists():
            try:
                sidecar.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        entries.append(RecoveryEntry(
            doc_path=doc_path,
            sidecar=sidecar,
            name=str(meta.get("name", "Untitled")),
            original_path=meta.get("original_path"),
            saved_at=float(meta.get("saved_at", 0.0)),
        ))
    entries.sort(key=lambda e: e.saved_at, reverse=True)
    return entries


def purge_stale_markers() -> None:
    """Remove liveness markers from long-dead sessions."""
    for marker in recovery_dir().glob(f"*{_LIVE_SUFFIX}"):
        try:
            if time.time() - marker.stat().st_mtime > _STALE_AFTER_SECONDS:
                marker.unlink(missing_ok=True)
        except OSError:
            pass
