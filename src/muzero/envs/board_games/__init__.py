"""Board game environments package."""

from muzero.envs.board_games.connect4 import Connect4Env
from muzero.envs.board_games.tictactoe import TicTacToeEnv

__all__ = ["Connect4Env", "TicTacToeEnv"]
