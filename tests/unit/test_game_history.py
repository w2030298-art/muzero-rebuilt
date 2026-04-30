"""Tests for the GameHistory class."""

from __future__ import annotations

import numpy as np

from muzero.core.game_history import GameHistory
from muzero.core.types import SearchResult, TimeStep


def _make_timestep(obs: np.ndarray, reward: float, to_play: int = 0) -> TimeStep:
    """Helper to create a TimeStep for testing."""
    return TimeStep(
        observation=obs,
        reward=reward,
        terminated=False,
        truncated=False,
        to_play=to_play,
    )


def _make_search_result(action: int, root_value: float) -> SearchResult:
    """Helper to create a SearchResult for testing."""
    return SearchResult(
        action=action,
        root_value=root_value,
        visit_counts=np.array([0.5, 0.3, 0.2]),
        policy_target=np.array([0.5, 0.3, 0.2]),
        search_depth=1,
        num_expanded_nodes=3,
    )


def test_game_history_append_initial_and_step() -> None:
    """Verify that appending initial and steps creates correct history."""
    history = GameHistory()

    # Initial state
    ts0 = _make_timestep(np.array([0.0]), reward=0.0)
    history.append_initial(ts0)

    # Step 1
    ts1 = _make_timestep(np.array([1.0]), reward=1.0)
    result1 = _make_search_result(action=0, root_value=0.5)
    history.append(action=0, timestep=ts1, search_result=result1)

    assert len(history) == 1
    assert history.observations[0] is ts0.observation  # initial obs
    assert history.observations[1] is ts1.observation  # after step 1
    assert history.actions[0] == 0
    assert float(history.rewards[0]) == 1.0  # type: ignore[arg-type]


def test_game_history_length() -> None:
    """Verify __len__ returns the number of actions taken."""
    history = GameHistory()

    ts0 = _make_timestep(np.array([0.0]), reward=0.0)
    history.append_initial(ts0)

    assert len(history) == 0

    ts1 = _make_timestep(np.array([1.0]), reward=0.0)
    history.append(action=1, timestep=ts1, search_result=_make_search_result(1, 0.0))
    assert len(history) == 1

    ts2 = _make_timestep(np.array([2.0]), reward=0.0)
    history.append(action=2, timestep=ts2, search_result=_make_search_result(2, 0.0))
    assert len(history) == 2


def test_game_history_stores_root_values() -> None:
    """Verify root values from search results are stored."""
    history = GameHistory()

    ts0 = _make_timestep(np.array([0.0]), reward=0.0)
    history.append_initial(ts0)

    ts1 = _make_timestep(np.array([1.0]), reward=1.0)
    history.append(
        action=0,
        timestep=ts1,
        search_result=_make_search_result(action=0, root_value=0.75),
    )

    assert len(history.root_values) == 1
    assert float(history.root_values[0]) == 0.75  # type: ignore[arg-type]


def test_game_history_stores_visit_distributions() -> None:
    """Verify child visit distributions are stored."""
    history = GameHistory()

    ts0 = _make_timestep(np.array([0.0]), reward=0.0)
    history.append_initial(ts0)

    ts1 = _make_timestep(np.array([1.0]), reward=0.0)
    result = _make_search_result(action=1, root_value=0.0)
    history.append(action=1, timestep=ts1, search_result=result)

    assert len(history.child_visit_distributions) == 1
    assert np.allclose(history.child_visit_distributions[0], result.policy_target)


def test_game_history_stores_legal_action_masks() -> None:
    """Verify legal action masks are stored from timesteps."""
    history = GameHistory()

    ts0 = _make_timestep(np.array([0.0]), reward=0.0)
    history.append_initial(ts0)

    mask = np.array([1, 1, 0], dtype=bool)
    ts1 = TimeStep(
        observation=np.array([1.0]),
        reward=0.0,
        terminated=False,
        truncated=False,
        to_play=0,
        legal_actions=mask,
    )
    history.append(action=0, timestep=ts1, search_result=_make_search_result(0, 0.0))

    assert history.legal_action_masks[0] is mask


def test_game_history_player_tracking() -> None:
    """Verify player indices are tracked correctly."""
    history = GameHistory()

    ts0 = _make_timestep(np.array([0.0]), reward=0.0, to_play=0)
    history.append_initial(ts0)

    ts1 = _make_timestep(np.array([1.0]), reward=0.0, to_play=1)
    history.append(action=0, timestep=ts1, search_result=_make_search_result(0, 0.0))

    assert history.players[0] == 0
    assert history.players[1] == 1
