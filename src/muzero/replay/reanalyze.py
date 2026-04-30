"""Reanalyze queue — placeholder for EfficientZero reanalysis.

Stores references to replay buffer items for later re-processing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class ReplayItemRef:
    """Reference to a specific position in the replay buffer.

    Attributes:
        game_index: Index of the game in the replay buffer.
        start_index: Starting position within the game.
    """

    game_index: int
    start_index: int


class ReanalyzeQueue:
    """Queue of replay items waiting for reanalysis.

    Used by EfficientZero to re-compute targets with a fresh network.
    First version is a simple FIFO queue.
    """

    def __init__(self) -> None:
        self._queue: deque[ReplayItemRef] = deque()

    def enqueue(self, refs: list[ReplayItemRef]) -> None:
        """Add items to the reanalyze queue.

        Args:
            refs: List of replay item references.
        """
        self._queue.extend(refs)

    def dequeue(self, batch_size: int) -> list[ReplayItemRef]:
        """Remove and return items from the queue.

        Args:
            batch_size: Maximum number of items to return.

        Returns:
            List of up to batch_size items.
        """
        result: list[ReplayItemRef] = []
        for _ in range(min(batch_size, len(self._queue))):
            result.append(self._queue.popleft())
        return result

    def __len__(self) -> int:
        """Return the number of queued items."""
        return len(self._queue)
