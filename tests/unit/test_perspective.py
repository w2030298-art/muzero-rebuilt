"""Tests for the PlayerPerspective projection."""

from __future__ import annotations

import torch

from muzero.core.perspective import PlayerPerspective


def test_single_player_projection_identity() -> None:
    """Single player value should be unchanged (identity)."""
    value = torch.tensor([1.0, 2.0, 3.0])
    result = PlayerPerspective.project_value(value, to_play=0, num_players=1)
    assert torch.equal(result, value)


def test_single_player_scalar() -> None:
    """Single player scalar value should be identity."""
    value = torch.tensor(5.0)
    result = PlayerPerspective.project_value(value, to_play=0, num_players=1)
    assert float(result) == 5.0


def test_two_player_zero_sum_projection_player0() -> None:
    """Two-player zero-sum: value[0] - value[1]."""
    value = torch.tensor([0.8, -0.5])
    result = PlayerPerspective.project_value(value, to_play=0, num_players=2)
    expected = 0.8 - (-0.5)  # 1.3
    assert abs(float(result) - expected) < 1e-6


def test_two_player_zero_sum_projection_player1() -> None:
    """Two-player zero-sum: value[1] - value[0]."""
    value = torch.tensor([0.8, -0.5])
    result = PlayerPerspective.project_value(value, to_play=1, num_players=2)
    expected = -0.5 - 0.8  # -1.3
    assert abs(float(result) - expected) < 1e-6


def test_two_player_batch() -> None:
    """Two-player projection works with batch dimension."""
    value = torch.tensor([[1.0, -1.0], [0.5, 0.5]])
    result = PlayerPerspective.project_value(value, to_play=0, num_players=2)
    expected = torch.tensor([2.0, 0.0])
    assert torch.allclose(result, expected)


def test_three_player_projection() -> None:
    """Three-player: value[to_play] - mean(value[others])."""
    value = torch.tensor([1.0, 0.5, 0.0])
    result = PlayerPerspective.project_value(value, to_play=0, num_players=3)
    # others = [0.5, 0.0], mean = 0.25
    expected = 1.0 - 0.25  # 0.75
    assert abs(float(result) - expected) < 1e-6


def test_three_player_middle() -> None:
    """Three-player projection for middle player."""
    value = torch.tensor([1.0, 0.5, 0.0])
    result = PlayerPerspective.project_value(value, to_play=1, num_players=3)
    # others = [1.0, 0.0], mean = 0.5
    expected = 0.5 - 0.5  # 0.0
    assert abs(float(result) - expected) < 1e-6


def test_project_reward_same_as_value() -> None:
    """project_reward should use the same logic as project_value."""
    reward = torch.tensor([1.0, -1.0])
    val_result = PlayerPerspective.project_value(reward, to_play=0, num_players=2)
    rew_result = PlayerPerspective.project_reward(reward, to_play=0, num_players=2)
    assert float(val_result) == float(rew_result)


def test_scalar_input_returns_unchanged() -> None:
    """If input is already a scalar 1D tensor, return as-is for multi-player."""
    value = torch.tensor([1.0, 2.0, 3.0])  # 3 values, not a [B,3] vector
    # This is ambiguous but we treat it as a scalar batch
    result = PlayerPerspective.project_value(value, to_play=0, num_players=2)
    assert result.shape == value.shape
