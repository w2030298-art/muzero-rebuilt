"""Tests for the GymnasiumAdapter."""

from __future__ import annotations

import numpy as np

from muzero.core.types import TimeStep
from muzero.envs.gymnasium_adapter import GymnasiumAdapter


def test_cartpole_reset_returns_timestep() -> None:
    """Verify CartPole reset returns a valid TimeStep."""
    adapter = GymnasiumAdapter("CartPole-v1")
    ts = adapter.reset(seed=0)

    assert isinstance(ts, TimeStep)
    assert ts.observation is not None
    assert ts.observation.shape == (4,)
    assert ts.reward == 0.0
    assert ts.terminated is False
    assert ts.truncated is False
    assert ts.to_play == 0
    adapter.close()


def test_cartpole_step_preserves_terminated_truncated() -> None:
    """Verify that step returns proper terminated/truncated fields."""
    adapter = GymnasiumAdapter("CartPole-v1")
    adapter.reset(seed=0)

    # Take a step — should not terminate immediately
    ts = adapter.step(0)

    assert isinstance(ts, TimeStep)
    assert ts.observation is not None
    # terminated/truncated may or may not be True depending on the step
    assert isinstance(ts.terminated, bool)
    assert isinstance(ts.truncated, bool)
    adapter.close()


def test_cartpole_action_space_discrete() -> None:
    """Verify CartPole action space is discrete with 2 actions."""
    adapter = GymnasiumAdapter("CartPole-v1")
    spec = adapter.action_space_spec()

    assert spec.type == "discrete"
    assert spec.n == 2
    adapter.close()


def test_pendulum_action_space_continuous() -> None:
    """Verify Pendulum action space is continuous."""
    adapter = GymnasiumAdapter("Pendulum-v1")
    spec = adapter.action_space_spec()

    assert spec.type == "continuous"
    assert spec.shape is not None
    assert spec.low is not None
    assert spec.high is not None
    adapter.close()


def test_gymnasium_adapter_num_players() -> None:
    """Verify GymnasiumAdapter reports 1 player."""
    adapter = GymnasiumAdapter("CartPole-v1")
    assert adapter.num_players() == 1
    assert adapter.current_player() == 0
    adapter.close()


def test_gymnasium_adapter_legal_actions_is_none() -> None:
    """Verify legal_actions returns None (no legal action mask)."""
    adapter = GymnasiumAdapter("CartPole-v1")
    adapter.reset(seed=0)
    assert adapter.legal_actions() is None
    adapter.close()


def test_observation_space_spec() -> None:
    """Verify observation space spec is correct."""
    adapter = GymnasiumAdapter("CartPole-v1")
    spec = adapter.observation_space_spec()

    assert spec.shape == (4,)
    adapter.close()


def test_pendulum_reset_step() -> None:
    """Verify Pendulum reset and step work correctly."""
    adapter = GymnasiumAdapter("Pendulum-v1")
    ts = adapter.reset(seed=0)

    assert ts.observation.shape == (3,)
    assert ts.reward == 0.0
    assert ts.to_play == 0

    # Take a step — check action is within bounds
    ts2 = adapter.step(np.array([0.0], dtype=np.float32))
    assert ts2.observation.shape == (3,)
    adapter.close()
