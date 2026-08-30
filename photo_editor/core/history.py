"""Undo / redo history with structural sharing and a memory budget.

Why this is not a naive snapshot stack
--------------------------------------
The original implementation deep-copied every layer's pixel, mask and
source buffers into every snapshot, bounded only by a 50-state count. For a
20-layer 4K project that measured **2,531 MB per snapshot** and would need
**124 GB** for a full history -- the app ran out of memory long before it
ran out of states.

Two changes fix that:

* **Structural sharing.** A snapshot copies a layer's buffer only when that
  layer's ``content_version`` differs from the previous snapshot's. An edit
  normally touches one layer, so the other nineteen are shared by
  reference. Sharing is safe because history-owned buffers are private
  copies that nothing else mutates, and ``_restore`` copies on the way out.

* **A byte budget.** States are evicted oldest-first once the total unique
  bytes exceed the budget, so history has a hard memory ceiling instead of
  a state count that means nothing at 4K.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Default ceiling for retained undo data. Deliberately expressed in bytes:
# "50 states" means 6 MB for a small document and 124 GB for a large one.
DEFAULT_BUDGET_BYTES = 1 << 30  # 1 GiB


@dataclass
class HistoryState:
    """Single snapshot in the undo stack.

    ``layer_data`` values may be *shared* with neighbouring states. Treat
    every array here as immutable; copy before handing one to a layer.
    """

    name: str
    layer_data: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # layer id -> content_version at capture time, used for sharing.
    layer_versions: dict[str, int] = field(default_factory=dict)

    def unique_arrays(self) -> dict[int, np.ndarray]:
        """Distinct arrays held by this state, keyed by object identity."""
        return {id(a): a for a in self.layer_data.values()
                if isinstance(a, np.ndarray)}


class HistoryManager:
    """Linear undo/redo stack bounded by memory, not state count."""

    def __init__(
        self,
        max_states: int = 200,
        budget_bytes: int = DEFAULT_BUDGET_BYTES,
    ) -> None:
        self._states: list[HistoryState] = []
        self._index: int = -1
        self._max = max_states
        self._budget = budget_bytes

    # ---- Query --------------------------------------------------------------

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return self._index < len(self._states) - 1

    @property
    def current_index(self) -> int:
        if not self._states:
            return 0
        # At the end with a committed (non-live) tail, the UI shows an extra
        # row for the live document, so the index is one past the array.
        if self._index == len(self._states) - 1 and self._states[-1].name != "__Live__":
            return self._index + 1
        return self._index

    @property
    def states(self) -> list[HistoryState]:
        return self._states

    def latest(self) -> HistoryState | None:
        """Most recently pushed state, for structural sharing."""
        return self._states[-1] if self._states else None

    # ---- Mutation -----------------------------------------------------------

    def push(self, state: HistoryState, live_ids: set[int] | None = None) -> None:
        self._states = self._states[: self.current_index]
        self._states.append(state)
        if len(self._states) > self._max:
            self._states.pop(0)
        self._enforce_budget(live_ids)
        self._index = len(self._states) - 1

    def _enforce_budget(self, live_ids: set[int] | None = None) -> None:
        """Drop oldest states until history-owned bytes fit the budget.

        Always keeps at least two states so undo remains possible even when
        a single snapshot exceeds the budget on its own.
        """
        while len(self._states) > 2 and self.owned_bytes(live_ids) > self._budget:
            self._states.pop(0)

    def total_bytes(self) -> int:
        """Unique bytes referenced by all states (shared arrays counted once).

        Includes buffers that are *also* the live document's, so this is an
        upper bound on the memory history is responsible for, not the
        incremental cost. Use :meth:`owned_bytes` for that.
        """
        seen: dict[int, int] = {}
        for state in self._states:
            for arr in state.layer_data.values():
                if isinstance(arr, np.ndarray):
                    seen[id(arr)] = arr.nbytes
        return sum(seen.values())

    def owned_bytes(self, live_ids: set[int] | None = None) -> int:
        """Bytes history keeps alive that the live document does not.

        Snapshots reference the document's own buffers until a layer
        copy-on-writes away from them, so those bytes cost nothing extra.
        Only what history alone still holds counts against the budget.
        """
        if not live_ids:
            return self.total_bytes()
        seen: dict[int, int] = {}
        for state in self._states:
            for arr in state.layer_data.values():
                if isinstance(arr, np.ndarray) and id(arr) not in live_ids:
                    seen[id(arr)] = arr.nbytes
        return sum(seen.values())

    def undo(self) -> HistoryState | None:
        if self.can_undo:
            self._index -= 1
            return self._states[self._index]
        return None

    def redo(self) -> HistoryState | None:
        if self.can_redo:
            self._index += 1
            return self._states[self._index]
        return None

    def current(self) -> HistoryState | None:
        if 0 <= self._index < len(self._states):
            return self._states[self._index]
        return None

    def clear(self) -> None:
        self._states.clear()
        self._index = -1

    def names(self) -> list[str]:
        if not self._states:
            return []
        res = ["Open Document"]
        for s in self._states:
            res.append(s.name)
        if res[-1] == "__Live__":
            res.pop()
        return res

    # ---- Introspection ------------------------------------------------------

    def stats(self, live_ids: set[int] | None = None) -> dict:
        total = self.total_bytes()
        owned = self.owned_bytes(live_ids)
        return {
            "states": len(self._states),
            "index": self._index,
            "bytes": total,
            "mb": round(total / (1 << 20), 1),
            "owned_bytes": owned,
            "owned_mb": round(owned / (1 << 20), 1),
            "budget_mb": round(self._budget / (1 << 20), 1),
        }
