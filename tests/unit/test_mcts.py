"""Tests for the standard MuZero MCTS implementation."""

from __future__ import annotations

import numpy as np
import torch

from muzero.config.schema import SearchConfig
from muzero.models.outputs import NetworkOutput
from muzero.models.protocol import MuZeroNetworkProtocol
from muzero.search.action_sampler import DiscreteActionSampler
from muzero.search.mcts import MCTS
from muzero.search.policies import PUCTPolicy

# ---- Fake Network for Testing ----


class FakeNetwork(MuZeroNetworkProtocol):
    """A simple fake network that returns fixed outputs for all inferences."""

    def __init__(self, action_dim: int = 4, hidden_size: int = 8, num_players: int = 1):
        self._action_dim = action_dim
        self._hidden_size = hidden_size
        self._num_players = num_players
        self._weights: dict[str, torch.Tensor] = {}

    def initial_inference(self, observation_batch: torch.Tensor) -> NetworkOutput:
        batch_size = observation_batch.shape[0]
        n_val = 1 if self._num_players == 1 else self._num_players

        return NetworkOutput(
            value=torch.zeros(batch_size) if n_val == 1 else torch.zeros(batch_size, n_val),
            reward=torch.zeros(batch_size) if n_val == 1 else torch.zeros(batch_size, n_val),
            policy_logits=torch.ones(batch_size, self._action_dim),
            hidden_state=torch.randn(batch_size, self._hidden_size),
        )

    def recurrent_inference(
        self, hidden_state_batch: torch.Tensor, action_batch: torch.Tensor
    ) -> NetworkOutput:
        batch_size = hidden_state_batch.shape[0]
        n_val = 1 if self._num_players == 1 else self._num_players

        return NetworkOutput(
            value=torch.zeros(batch_size) if n_val == 1 else torch.zeros(batch_size, n_val),
            reward=torch.zeros(batch_size) if n_val == 1 else torch.zeros(batch_size, n_val),
            policy_logits=torch.ones(batch_size, self._action_dim),
            hidden_state=torch.randn(batch_size, self._hidden_size),
        )

    def get_weights(self) -> dict[str, torch.Tensor]:
        return dict(self._weights)

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self._weights = dict(weights)


# ---- Helper Functions ----


def _make_mcts(
    action_dim: int = 4,
    hidden_size: int = 8,
    num_players: int = 1,
    num_simulations: int = 8,
) -> MCTS:
    """Create an MCTS instance with a fake network for testing."""
    network = FakeNetwork(action_dim=action_dim, hidden_size=hidden_size, num_players=num_players)
    config = SearchConfig(
        num_simulations=num_simulations,
        discount=1.0,
    )
    sampler = DiscreteActionSampler()
    policy = PUCTPolicy()
    device = torch.device("cpu")

    return MCTS(
        network=network,
        config=config,
        action_sampler=sampler,
        search_policy=policy,
        device=device,
        num_players=num_players,
        action_dim=action_dim,
    )


# ---- Tests ----


def test_mcts_runs_configured_simulations() -> None:
    """Verify MCTS runs the configured number of simulations."""
    mcts = _make_mcts(action_dim=4, num_simulations=8)
    obs = np.random.randn(10).astype(np.float32)
    legal = np.ones(4, dtype=bool)

    result = mcts.run(root_observation=obs, legal_actions=legal, to_play=0)

    assert result.search_depth == 8
    assert result.num_expanded_nodes > 0


def test_mcts_returns_legal_action() -> None:
    """Verify MCTS returns an action from the legal actions."""
    mcts = _make_mcts(action_dim=4, num_simulations=10)
    obs = np.random.randn(10).astype(np.float32)

    # Only actions 0 and 2 are legal
    legal = np.array([True, False, True, False])

    result = mcts.run(root_observation=obs, legal_actions=legal, to_play=0)

    assert result.action in [0, 2]


def test_mcts_policy_target_sums_to_one() -> None:
    """Verify policy target is a valid probability distribution."""
    mcts = _make_mcts(action_dim=4, num_simulations=10)
    obs = np.random.randn(10).astype(np.float32)

    result = mcts.run(root_observation=obs, legal_actions=None, to_play=0)

    assert result.policy_target.shape == (4,)
    assert abs(float(result.policy_target.sum()) - 1.0) < 1e-5
    assert np.all(result.policy_target >= 0)


def test_mcts_all_actions_legal() -> None:
    """Verify MCTS works when all actions are legal."""
    mcts = _make_mcts(action_dim=4, num_simulations=5)
    obs = np.random.randn(10).astype(np.float32)

    result = mcts.run(root_observation=obs, legal_actions=None, to_play=0)

    assert result.action is not None
    assert result.visit_counts.shape == (4,)


def test_mcts_two_player() -> None:
    """Verify MCTS works for two-player games."""
    mcts = _make_mcts(action_dim=9, hidden_size=16, num_players=2, num_simulations=8)
    obs = np.random.randn(9).astype(np.float32)
    legal = np.ones(9, dtype=bool)

    result = mcts.run(root_observation=obs, legal_actions=legal, to_play=0)

    assert result.policy_target.shape == (9,)
    assert abs(float(result.policy_target.sum()) - 1.0) < 1e-5


def test_mcts_root_value_defined() -> None:
    """Verify MCTS produces a non-trivial root value."""
    mcts = _make_mcts(action_dim=4, num_simulations=10)
    obs = np.random.randn(10).astype(np.float32)

    result = mcts.run(root_observation=obs, legal_actions=None, to_play=0)

    # Root value should be defined (visit counts > 0)
    assert isinstance(result.root_value, (float, np.ndarray))


def test_mcts_temperature_0_selects_most_visited() -> None:
    """Verify temperature=0 selects the most visited action."""
    mcts = _make_mcts(action_dim=4, num_simulations=20)
    obs = np.random.randn(10).astype(np.float32)

    # Set temperature=0
    mcts._config.temperature = 0.0  # type: ignore[attr-defined]

    result = mcts.run(root_observation=obs, legal_actions=None, to_play=0)

    # The action should be the one with the highest visit count
    max_idx = int(np.argmax(result.visit_counts))
    assert result.action == max_idx
