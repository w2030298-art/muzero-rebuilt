"""Replay buffer for storing and sampling game histories."""

from __future__ import annotations

import numpy as np

from muzero.config.schema import ReplayConfig
from muzero.core.game_history import GameHistory
from muzero.core.types import TargetSequence, TrainingBatch
from muzero.replay.target_builder import TargetBuilder


class ReplayBuffer:
    """Ring buffer storing GameHistory objects and sampling training batches.

    Args:
        config: Replay buffer configuration (capacity, etc.).
        target_builder: TargetBuilder for constructing training targets.
    """

    def __init__(self, config: ReplayConfig, target_builder: TargetBuilder) -> None:
        self._capacity = config.capacity
        self._target_builder = target_builder
        self._games: list[GameHistory] = []
        self._head = 0

    def add_game(self, history: GameHistory) -> None:
        """Add a completed game to the buffer.

        If the buffer is at capacity, the oldest game is overwritten.

        Args:
            history: Completed game history.
        """
        if len(self._games) < self._capacity:
            self._games.append(history)
        else:
            self._games[self._head] = history
            self._head = (self._head + 1) % self._capacity

    def sample_batch(
        self,
        batch_size: int,
        num_unroll_steps: int,
        td_steps: int,
    ) -> TrainingBatch:
        """Sample a batch of training sequences from the buffer.

        Args:
            batch_size: Number of sequences to sample.
            num_unroll_steps: Number of unroll steps per sequence.
            td_steps: Number of TD steps for value target bootstrapping.

        Returns:
            TrainingBatch with stacked observations, actions, targets.

        Raises:
            RuntimeError: If the buffer is empty.
        """
        if len(self) == 0:
            raise RuntimeError("ReplayBuffer is empty")

        indices = np.random.randint(0, len(self._games), size=batch_size)

        targets: list[TargetSequence] = []
        for game_idx in indices:
            game = self._games[int(game_idx)]
            # Pick random start index
            max_start = max(0, len(game) - num_unroll_steps)
            start_idx = 0 if max_start == 0 else int(np.random.randint(0, max_start + 1))

            target = self._target_builder.build_targets(
                history=game,
                start_index=start_idx,
                num_unroll_steps=num_unroll_steps,
                td_steps=td_steps,
            )
            targets.append(target)

        # Stack targets into batch
        return self._stack_targets(targets, batch_size)

    def _stack_targets(self, targets: list[TargetSequence], batch_size: int) -> TrainingBatch:
        """Stack a list of TargetSequences into a single TrainingBatch.

        Args:
            targets: List of target sequences.
            batch_size: Expected batch size.

        Returns:
            TrainingBatch with stacked arrays.
        """
        observations = np.stack([t.observations for t in targets])
        actions = np.stack([t.actions for t in targets])
        target_values = np.stack([t.target_values for t in targets])
        target_rewards = np.stack([t.target_rewards for t in targets])
        target_policies = np.stack([t.target_policies for t in targets])

        masks = None
        if all(t.masks is not None for t in targets):
            masks = np.stack([t.masks for t in targets])  # type: ignore[arg-type]

        target_value_prefixes = None
        if all(t.target_value_prefixes is not None for t in targets):
            target_value_prefixes = np.stack(
                [t.target_value_prefixes for t in targets]  # type: ignore[arg-type]
            )

        return TrainingBatch(
            observations=observations,
            actions=actions,
            target_values=target_values,
            target_rewards=target_rewards,
            target_policies=target_policies,
            masks=masks,
            target_value_prefixes=target_value_prefixes,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for specified indices (no-op in base buffer).

        Args:
            indices: Indices to update.
            priorities: New priority values.
        """
        pass

    def __len__(self) -> int:
        """Return the number of stored games."""
        return len(self._games)
