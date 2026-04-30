"""Smoke test: Three-player toy env with value vectors."""

from __future__ import annotations

from muzero.envs.board_games.three_player_toy import ThreePlayerToyEnv


def test_three_player_env_reset() -> None:
    """Verify three-player env reset state."""
    env = ThreePlayerToyEnv()
    ts = env.reset(seed=0)

    assert env.num_players() == 3
    assert env.current_player() == 0
    assert ts.reward is not None
    assert hasattr(ts.reward, "__len__")
    assert len(ts.reward) == 3  # type: ignore[arg-type]


def test_three_player_env_step() -> None:
    """Verify stepping through three-player env."""
    env = ThreePlayerToyEnv()
    env.reset()

    # Player 0: action=1 → gets +1
    ts1 = env.step(1)
    assert ts1.reward[0] == 1.0  # type: ignore[index]
    assert env.current_player() == 1

    # Player 1: action=0 → gets 0
    ts2 = env.step(0)
    assert ts2.reward[1] == 0.0  # type: ignore[index]
    assert env.current_player() == 2

    # Player 2: action=1 → gets +1
    ts3 = env.step(1)
    assert ts3.terminated
    assert ts3.reward[2] == 1.0  # type: ignore[index]


def test_three_player_legal_actions() -> None:
    """Verify all actions are always legal."""
    env = ThreePlayerToyEnv()
    env.reset()
    legal = env.legal_actions()
    assert legal is not None
    assert legal.sum() == 2
