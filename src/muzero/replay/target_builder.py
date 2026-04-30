"""Target builder: constructs training targets from game histories."""

from __future__ import annotations

import numpy as np

from muzero.core.game_history import GameHistory
from muzero.core.types import TargetSequence


class TargetBuilder:
    """Builds value, reward, and policy training targets from game histories.

    Args:
        discount: Discount factor for n-step value bootstrapping.
    """

    def __init__(self, discount: float) -> None:
        self._discount = discount

    def build_targets(
        self,
        history: GameHistory,
        start_index: int,
        num_unroll_steps: int,
        td_steps: int,
    ) -> TargetSequence:
        """Build a full TargetSequence from a game history.

        Args:
            history: The game history to build targets from.
            start_index: Starting position in the history.
            num_unroll_steps: Number of unroll steps.
            td_steps: Number of TD steps for value bootstrapping.

        Returns:
            TargetSequence with observations, actions, values, rewards, policies.
        """
        episode_len = len(history)
        K = num_unroll_steps

        # Observations: K+1 frames (initial + after each unroll step)
        observations = np.zeros((K + 1,) + history.observations[0].shape, dtype=np.float32)
        # Actions: K actions
        actions = np.zeros((K,), dtype=np.int64)
        # Target values: K+1 n-step bootstrapped values
        target_values = np.zeros((K + 1,) + self._value_shape(history), dtype=np.float32)
        # Target rewards: K immediate rewards
        target_rewards = np.zeros((K,) + self._reward_shape(history), dtype=np.float32)
        # Target policies: K+1 policy targets
        action_dim = self._policy_dim(history)
        target_policies = np.zeros((K + 1, action_dim), dtype=np.float32)
        # Masks: K+1 valid position mask
        masks = np.zeros(K + 1, dtype=np.float32)

        for k in range(K + 1):
            idx = start_index + k
            if idx <= episode_len:
                masks[k] = 1.0
            if idx < episode_len:
                observations[k] = history.observations[idx]
            elif episode_len > 0:
                # Beyond episode: repeat last observation or zeros
                observations[k] = history.observations[-1]

            # Build n-step value target
            target_values[k] = self._build_n_step_value(history, idx, td_steps)

            # Build policy target
            target_policies[k] = self._build_policy_target(history, idx)

        for k in range(K):
            idx = start_index + k
            if idx < episode_len:
                actions[k] = int(history.actions[idx])  # type: ignore[arg-type]
                target_rewards[k] = self._extract_reward(history, idx)
            else:
                actions[k] = 0

        return TargetSequence(
            observations=observations,
            actions=actions,
            target_values=target_values,
            target_rewards=target_rewards,
            target_policies=target_policies,
            masks=masks,
        )

    def _build_n_step_value(
        self,
        history: GameHistory,
        start_index: int,
        td_steps: int,
    ) -> np.ndarray | float:
        """Compute n-step bootstrapped value target.

        Accumulates discounted rewards over td_steps, then bootstraps
        from the root_value at the bootstrap index if not terminal.

        Args:
            history: Game history.
            start_index: Starting index.
            td_steps: Number of TD steps.

        Returns:
            n-step value target (scalar or vector).
        """
        episode_len = len(history)

        if start_index >= episode_len:
            return 0.0

        value = (
            np.zeros(self._value_shape(history), dtype=np.float32)
            if self._is_vector(history)
            else 0.0
        )

        for t in range(td_steps):
            idx = start_index + t
            if idx >= episode_len:
                break

            reward = self._extract_reward(history, idx)
            value = value + (self._discount**t) * reward

            # Check if episode terminated at this step
            if t > 0 and hasattr(history, "search_metadata") and idx < episode_len:
                pass  # We don't store terminated info per-step in current GameHistory

        # Bootstrap from root value if possible
        bootstrap_idx = start_index + td_steps
        if bootstrap_idx < episode_len:
            root_val = self._extract_root_value(history, bootstrap_idx)
            value = value + (self._discount**td_steps) * root_val

        return value

    def _build_policy_target(self, history: GameHistory, index: int) -> np.ndarray:
        """Build policy target from stored visit distributions.

        Args:
            history: Game history.
            index: Position index.

        Returns:
            Policy target array.
        """
        episode_len = len(history)
        if index < episode_len and index < len(history.child_visit_distributions):
            return history.child_visit_distributions[index].copy()
        # Return uniform distribution
        dim = self._policy_dim(history)
        return np.ones(dim, dtype=np.float32) / dim

    def _extract_reward(self, history: GameHistory, index: int) -> np.ndarray | float:
        """Extract reward at a given index.

        Args:
            history: Game history.
            index: Position index.

        Returns:
            Reward value.
        """
        if index < len(history):
            return history.rewards[index]  # type: ignore[return-value]
        return 0.0

    def _extract_root_value(self, history: GameHistory, index: int) -> np.ndarray | float:
        """Extract root value at a given index for bootstrapping.

        Args:
            history: Game history.
            index: Position index.

        Returns:
            Root value.
        """
        if index < len(history.root_values):
            return history.root_values[index]  # type: ignore[return-value]
        return 0.0

    def _value_shape(self, history: GameHistory) -> tuple[int, ...]:
        """Return the shape of a value target."""
        if history.root_values and isinstance(history.root_values[0], np.ndarray):
            return history.root_values[0].shape
        return ()

    def _reward_shape(self, history: GameHistory) -> tuple[int, ...]:
        """Return the shape of a reward target."""
        if history.rewards and isinstance(history.rewards[0], np.ndarray):
            return history.rewards[0].shape
        return ()

    def _is_vector(self, history: GameHistory) -> bool:
        """Check if values/rewards are vectors (multi-player)."""
        return bool(history.root_values and isinstance(history.root_values[0], np.ndarray))

    def _policy_dim(self, history: GameHistory) -> int:
        """Return the policy dimension."""
        if history.child_visit_distributions:
            return int(history.child_visit_distributions[0].shape[0])
        return 1
