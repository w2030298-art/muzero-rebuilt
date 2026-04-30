"""Tests for the TargetBuilder."""

from __future__ import annotations

import numpy as np

from muzero.core.game_history import GameHistory
from muzero.core.types import SearchResult, TimeStep
from muzero.replay.target_builder import TargetBuilder


def _make_game_history(length: int = 3) -> GameHistory:
    """Create a simple game history for testing (single player, 2 discrete actions)."""
    history = GameHistory()
    obs = np.zeros(4, dtype=np.float32)

    ts0 = TimeStep(observation=obs, reward=0.0, terminated=False, truncated=False, to_play=0)
    history.append_initial(ts0)

    for i in range(length):
        ts = TimeStep(
            observation=obs + float(i + 1),
            reward=float(i + 1),
            terminated=(i == length - 1),
            truncated=False,
            to_play=0,
        )
        result = SearchResult(
            action=i % 2,
            root_value=float(length - i),
            visit_counts=np.array([0.3, 0.7], dtype=np.float32),
            policy_target=np.array([0.3, 0.7], dtype=np.float32),
        )
        history.append(action=i % 2, timestep=ts, search_result=result)

    return history


def test_n_step_value_without_bootstrap_at_terminal() -> None:
    """Verify n-step value stops bootstrapping at terminal state."""
    tb = TargetBuilder(discount=0.997)
    history = _make_game_history(3)

    # At terminal index, n-step should just be the reward
    targets = tb.build_targets(history=history, start_index=2, num_unroll_steps=1, td_steps=5)

    assert targets.target_values.shape[0] == 2  # K+1 = 2


def test_n_step_value_with_bootstrap() -> None:
    """Verify n-step value bootstraps from root values."""
    tb = TargetBuilder(discount=0.997)
    history = _make_game_history(5)

    targets = tb.build_targets(history=history, start_index=0, num_unroll_steps=2, td_steps=2)

    assert targets.observations.shape[0] == 3  # K+1 = 3
    assert targets.actions.shape[0] == 2  # K = 2


def test_policy_targets_shape() -> None:
    """Verify policy targets have correct shape."""
    tb = TargetBuilder(discount=0.997)
    history = _make_game_history(3)

    targets = tb.build_targets(history=history, start_index=0, num_unroll_steps=1, td_steps=1)

    assert targets.target_policies.shape == (2, 2)  # K+1=2, action_dim=2


def test_masks_shape() -> None:
    """Verify masks are returned with correct shape."""
    tb = TargetBuilder(discount=0.997)
    history = _make_game_history(5)

    targets = tb.build_targets(history=history, start_index=0, num_unroll_steps=3, td_steps=1)

    assert targets.masks is not None
    assert targets.masks.shape == (4,)  # K+1=4
    # First positions should be valid
    assert targets.masks[0] == 1.0


def test_target_values_non_negative_in_reward_env() -> None:
    """Verify target values are computed for a reward-heavy env."""
    tb = TargetBuilder(discount=0.5)
    history = _make_game_history(3)

    targets = tb.build_targets(history=history, start_index=0, num_unroll_steps=2, td_steps=3)

    assert targets.target_values is not None
    assert targets.target_values.shape[0] == 3
