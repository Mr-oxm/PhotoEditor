"""Memory budgets scaled to the machine, not to fixed constants.

The render cache and the undo history both trade memory for speed, and the
right trade depends entirely on the machine. Fixed constants are wrong in
both directions: 1 GB of undo history is generous on a 48 GB workstation
and ruinous on an 8 GB laptop, where it would push the app into swap and
make every operation slower than having no cache at all.

Budgets are therefore expressed as a share of total system memory, with
floors that keep a small machine usable and caps that stop a large one
hoarding memory it will never need.
"""

from __future__ import annotations

import os

MB = 1 << 20
GB = 1 << 30

# Fractions of total RAM. They sum to well under half: the layer data
# itself, the OS, and everything else still need room.
_RENDER_CACHE_SHARE = 0.10
_HISTORY_SHARE = 0.08

_RENDER_CACHE_MIN = 192 * MB
_RENDER_CACHE_MAX = 2 * GB
_HISTORY_MIN = 128 * MB
# Capped at the 2 GB the performance plan targets. 8% of a 48 GB
# workstation is nearly 4 GB of undo history, which is more than the
# feature is worth and more than the plan promised.
_HISTORY_MAX = 2 * GB

_DEFAULT_TOTAL = 8 * GB   # assumed when the platform will not say


def total_system_memory() -> int:
    """Total physical memory in bytes, or a conservative default."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        pass
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().total)
    except Exception:
        return _DEFAULT_TOTAL


def _clamp(value: float, low: int, high: int) -> int:
    return int(max(low, min(value, high)))


def render_cache_budget() -> int:
    """Bytes the layer raster cache may hold.

    Undersizing this is worse than it looks: when a frame's working set
    does not fit, every layer is re-prepared on every frame, which measured
    421 ms/frame against 35 ms with the set resident.
    """
    env = _from_env("BASERA_RENDER_CACHE_MB")
    if env is not None:
        return env
    return _clamp(total_system_memory() * _RENDER_CACHE_SHARE,
                  _RENDER_CACHE_MIN, _RENDER_CACHE_MAX)


def history_budget() -> int:
    """Bytes the undo history may own beyond the live document."""
    env = _from_env("BASERA_HISTORY_MB")
    if env is not None:
        return env
    return _clamp(total_system_memory() * _HISTORY_SHARE,
                  _HISTORY_MIN, _HISTORY_MAX)


def _from_env(name: str) -> int | None:
    """Allow an explicit override in MB, for testing and for power users."""
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value * MB if value > 0 else None


def describe() -> dict:
    """Human-readable summary, for the status bar or a diagnostics view."""
    total = total_system_memory()
    return {
        "system_mb": round(total / MB),
        "render_cache_mb": round(render_cache_budget() / MB),
        "history_mb": round(history_budget() / MB),
    }
