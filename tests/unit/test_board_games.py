"""Tests for board game environments (TicTacToe, Connect4)."""

from __future__ import annotations

import numpy as np
import pytest

from muzero.envs.board_games.connect4 import Connect4Env
from muzero.envs.board_games.tictactoe import TicTacToeEnv

# ---- TicTacToe Tests ----


def test_tictactoe_legal_actions_initial() -> None:
    """Verify that all 9 actions are legal at the start of TicTacToe."""
    env = TicTacToeEnv()
    env.reset(seed=0)

    legal = env.legal_actions()
    assert legal is not None
    assert np.all(legal)  # All actions legal initially
    assert len(legal) == 9


def test_tictactoe_win_reward_vector() -> None:
    """Verify that winning produces the correct reward vector [1, -1]."""
    env = TicTacToeEnv()
    env.reset(seed=0)

    # Play a game where player 0 wins: top-left, center, top-mid, right-center, top-right
    # Actually let's use a known sequence: X in (0,1), (1,0), (2,2); O in (0,0), (1,1)
    #    0  1  2
    # 0  O  .  .
    # 1  .  O  .
    # 2  .  .  X
    # That's draw, let me use a winning sequence instead
    # Player 0 (X): 0, 4, 8 (diagonal)
    # Player 1 (O): 1, 5

    moves = [0, 1, 4, 5, 8]
    winner_found = False
    reward_at_win = None

    for move in moves:
        ts = env.step(move)
        if ts.terminated:
            winner_found = True
            reward_at_win = ts.reward
            break

    assert winner_found, "Expected player 0 to win with diagonal 0-4-8"
    assert reward_at_win is not None
    assert isinstance(reward_at_win, np.ndarray)
    assert reward_at_win[0] == 1.0  # Player 0 (last to move) wins
    assert reward_at_win[1] == -1.0  # Player 1 loses


def test_tictactoe_illegal_action_raises() -> None:
    """Verify that playing on an occupied cell raises ValueError."""
    env = TicTacToeEnv()
    env.reset(seed=0)

    env.step(0)  # Player 0 occupies (0,0)
    # Player 1 plays somewhere else
    env.step(4)

    with pytest.raises(ValueError):
        env.step(0)  # Player 0 tries to play at occupied (0,0)


def test_tictactoe_num_players() -> None:
    """Verify TicTacToe has 2 players."""
    env = TicTacToeEnv()
    assert env.num_players() == 2


def test_tictactoe_player_switches() -> None:
    """Verify that the player alternates after each move."""
    env = TicTacToeEnv()
    env.reset(seed=0)

    assert env.current_player() == 0
    env.step(0)
    assert env.current_player() == 1
    env.step(4)
    assert env.current_player() == 0


def test_tictactoe_draw() -> None:
    """Verify a draw game produces reward [0, 0]."""
    env = TicTacToeEnv()
    env.reset(seed=0)

    # Play a forced draw sequence
    # X: 0, 8, 6, 2, 4
    # O: 1, 3, 5, 7
    moves = [0, 1, 2, 3, 5, 4, 6, 8, 7]
    final_ts = None
    for move in moves:
        final_ts = env.step(move)

    assert final_ts is not None
    assert final_ts.terminated
    assert np.array_equal(final_ts.reward, np.array([0.0, 0.0], dtype=np.float32))


def test_tictactoe_observation_shape() -> None:
    """Verify observation shape is [3, 3]."""
    env = TicTacToeEnv()
    spec = env.observation_space_spec()
    assert spec.shape == (3, 3)


# ---- Connect4 Tests ----


def test_connect4_legal_actions_initial() -> None:
    """Verify that all 7 columns are legal at the start of Connect4."""
    env = Connect4Env()
    env.reset(seed=0)

    legal = env.legal_actions()
    assert legal is not None
    assert np.all(legal)
    assert len(legal) == 7


def test_connect4_column_fills_up() -> None:
    """Verify that a column becomes illegal after being filled."""
    env = Connect4Env()
    env.reset(seed=0)

    # Fill column 0 (6 rows)
    for _ in range(6):
        env.step(0)

    legal = env.legal_actions()
    assert legal is not None
    assert not legal[0]  # Column 0 should be full
    assert np.sum(legal) == 6  # Other 6 columns still available


def test_connect4_vertical_win_reward_vector() -> None:
    """Verify that a vertical win produces correct reward vector."""
    env = Connect4Env()
    env.reset(seed=0)

    # Player 0 fills column 0: rows 5, 4, 3, 2 → wins
    # Player 1 fills column 1 alternately
    moves = [0, 1, 0, 1, 0, 1, 0]  # Player 0 gets 4 in col 0
    winner_found = False
    reward_at_win = None

    for move in moves:
        ts = env.step(move)
        if ts.terminated:
            winner_found = True
            reward_at_win = ts.reward
            break

    assert winner_found, "Expected player 0 to win vertically in column 0"
    assert reward_at_win is not None
    assert isinstance(reward_at_win, np.ndarray)
    assert reward_at_win[0] == 1.0
    assert reward_at_win[1] == -1.0


def test_connect4_num_players() -> None:
    """Verify Connect4 has 2 players."""
    env = Connect4Env()
    assert env.num_players() == 2


def test_connect4_observation_shape() -> None:
    """Verify observation shape is [6, 7]."""
    env = Connect4Env()
    spec = env.observation_space_spec()
    assert spec.shape == (6, 7)


def test_connect4_horizontal_win() -> None:
    """Verify horizontal win detection."""
    env = Connect4Env()
    env.reset(seed=0)

    # P0: col 0, 1, 2, 3 → horizontal win
    # P1: col 0, 0, 0 (filling up column 0)
    moves = [0, 0, 1, 0, 2, 0, 3]
    winner_found = False
    for move in moves:
        ts = env.step(move)
        if ts.terminated:
            winner_found = True
            break

    assert winner_found, "Expected player 0 to win horizontally"


def test_connect4_diagonal_win() -> None:
    """Verify diagonal win detection."""
    env = Connect4Env()
    env.reset(seed=0)

    # Set up a diagonal
    # P0 builds diagonal, P1 fills other spots
    moves = [
        0,
        1,  # P0 col0, P1 col1
        1,
        2,  # P0 col1, P1 col2
        2,
        3,  # P0 col2, P1 col3
        2,
        3,  # P0 col2, P1 col3
        3,
        0,  # P0 col3, P1 col0
        3,  # P0 col3 (wins: diagonal col0-col3 going up)
    ]
    winner_found = False
    for move in moves:
        ts = env.step(move)
        if ts.terminated:
            winner_found = True
            break

    assert winner_found, "Expected player 0 to win diagonally"
