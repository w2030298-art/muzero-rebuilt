"""Prioritized experience replay buffer.

Extends ReplayBuffer with priority-based sampling and importance weighting.
Uses a simple array-based priority storage (not sum-tree).
"""

from __future__ import annotations

import numpy as np

from muzero.config.schema import ReplayConfig
from muzero.core.game_history import GameHistory
from muzero.core.types import TrainingBatch
from muzero.replay.buffer import ReplayBuffer
from muzero.replay.target_builder import TargetBuilder


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with priority-based sampling.

    Priorities are stored as a flat array (one per game).
    Sampling probability is proportional to ``priority ** alpha``.
    Importance weights correct for the sampling bias.

    Args:
        config: Replay configuration (capacity, alpha, beta, etc.).
        target_builder: TargetBuilder for constructing training targets.
    """

    def __init__(self, config: ReplayConfig, target_builder: TargetBuilder) -> None:
        super().__init__(config, target_builder)
        self._alpha = config.alpha
        self._beta = config.beta
        self._priorities: np.ndarray = np.ones(config.capacity, dtype=np.float32)

    def add_game(self, history: GameHistory) -> None:
        """Add a game and assign maximum priority.

        Args:
            history: Completed game history.
        """
        idx = len(self._games) if len(self._games) < self._capacity else self._head
        super().add_game(history)
        if idx < len(self._priorities):
            self._priorities[idx] = float(self._priorities.max()) if len(self._games) > 0 else 1.0

    def sample_batch(
        self,
        batch_size: int,
        num_unroll_steps: int,
        td_steps: int,
    ) -> TrainingBatch:
        """Sample using priority-based probabilities.

        Args:
            batch_size: Number of sequences to sample.
            num_unroll_steps: Number of unroll steps.
            td_steps: Number of TD steps.

        Returns:
            TrainingBatch with importance weights.
        """
        if len(self) == 0:
            raise RuntimeError("PrioritizedReplayBuffer is empty")

        n = len(self._games)
        indices, probs = self._sample_indices(batch_size)
        imp_weights = self._compute_importance_weights(probs, n)

        targets = []
        for game_idx in indices:
            game = self._games[int(game_idx)]
            max_start = max(0, len(game) - num_unroll_steps)
            start_idx = 0 if max_start == 0 else int(np.random.randint(0, max_start + 1))
            target = self._target_builder.build_targets(
                history=game,
                start_index=start_idx,
                num_unroll_steps=num_unroll_steps,
                td_steps=td_steps,
            )
            targets.append(target)

        batch = self._stack_targets(targets, batch_size)
        batch.importance_weights = imp_weights
        batch.indices = indices
        return batch

    def _sample_indices(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample game indices according to priority-based probabilities.

        Returns:
            Tuple of (indices, probabilities).
        """
        n = len(self._games)
        priorities = self._priorities[:n]
        probs = priorities**self._alpha
        probs /= probs.sum()

        indices = np.random.choice(n, size=batch_size, p=probs, replace=True)
        return indices, probs[indices]

    def _compute_importance_weights(self, probs: np.ndarray, n: int) -> np.ndarray:
        """Compute importance sampling weights.

        ``w_i = (N * p_i) ** (-beta)``, then normalized by max weight.

        Args:
            probs: Sampling probabilities for each selected index.
            n: Total number of items.

        Returns:
            Importance weight array, normalized to max=1.
        """
        weights = (n * probs) ** (-self._beta)
        max_w = float(weights.max())
        if max_w > 0:
            weights /= max_w
        return weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for specified indices.

        Args:
            indices: Game indices to update.
            priorities: New priority values.
        """
        for idx, prio in zip(indices, priorities, strict=False):
            if int(idx) < len(self._priorities):
                self._priorities[int(idx)] = float(prio)
