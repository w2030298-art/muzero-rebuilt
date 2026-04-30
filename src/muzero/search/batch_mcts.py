"""Batch MCTS — runs multiple MCTS searches sharing an InferenceBatcher.

v1: Sequential MCTS with shared batcher. Each MCTS instance uses the
same batcher, so inference requests from different trees are combined
into single GPU calls.
"""

from __future__ import annotations

from collections.abc import Callable

from muzero.search.inference_batcher import InferenceBatcher
from muzero.search.mcts import MCTS, SearchRequest, SearchResult


class BatchMCTS:
    """Batch MCTS that shares an InferenceBatcher across multiple searches.

    Args:
        mcts_factory: Factory function that creates an MCTS instance.
        batcher: Shared inference batcher for all MCTS instances.
    """

    def __init__(
        self,
        mcts_factory: Callable[[], MCTS],
        batcher: InferenceBatcher,
    ) -> None:
        self._mcts_factory = mcts_factory
        self._batcher = batcher

    def run_batch(self, requests: list[SearchRequest]) -> list[SearchResult]:
        """Run MCTS for multiple root states.

        Each request creates its own MCTS tree, but all share the
        batcher for inference. MCTS instances are created via the factory.

        Args:
            requests: List of SearchRequest objects.

        Returns:
            List of SearchResult, one per request.
        """
        # Create MCTS instances (they share the batcher)
        searchers: list[MCTS] = []
        for _req in requests:
            mcts = self._mcts_factory()
            searchers.append(mcts)

        results: list[SearchResult] = []

        for i, req in enumerate(requests):
            result = searchers[i].run(
                root_observation=req.observation,
                legal_actions=req.legal_actions,
                to_play=req.to_play,
            )
            results.append(result)

        return results

    def flush_inference(self) -> None:
        """Flush any remaining inference requests in the batcher."""
        self._batcher.flush()
