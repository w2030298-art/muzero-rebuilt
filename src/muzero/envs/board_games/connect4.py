"""Connect4 environment — two-player zero-sum board game.

Board: 6 rows x 7 columns. Player 0 marks 1, Player 1 marks -1.
Actions: 0-6 (column to drop piece). Win: 4 in a row.
"""

from __future__ import annotations

import numpy as np

from muzero.core.specs import ActionSpaceSpec, ObservationSpaceSpec
from muzero.core.types import Action, LegalActions, TimeStep

ROWS = 6
COLS = 7
N_ACTIONS = COLS  # 7


class Connect4Env:
    """Connect4 for two players.

    Player 0 places ``1``, Player 1 places ``-1``.
    Pieces drop to the lowest empty row in the selected column.
    Observation is a ``[6, 7]`` board with values ``{0, 1, -1}``.
    Reward is a ``[2]`` numpy vector: ``[r_p0, r_p1]``.

    Attributes:
        board: Current board state, shape [6, 7].
        to_play: Current player (0 or 1).
    """

    def __init__(self) -> None:
        self.board: np.ndarray = np.zeros((ROWS, COLS), dtype=np.float32)
        self.to_play: int = 0

    def reset(self, seed: int | None = None) -> TimeStep:
        """Reset the board to empty.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Initial TimeStep with empty board.
        """
        if seed is not None:
            np.random.seed(seed)
        self.board = np.zeros((ROWS, COLS), dtype=np.float32)
        self.to_play = 0
        return TimeStep(
            observation=self.board.copy(),
            reward=np.array([0.0, 0.0], dtype=np.float32),
            terminated=False,
            truncated=False,
            to_play=0,
        )

    def step(self, action: Action) -> TimeStep:
        """Drop a piece in the given column.

        Args:
            action: Integer 0-6 representing the column.

        Returns:
            TimeStep with updated board and reward.

        Raises:
            ValueError: If the column is full.
        """
        col = int(action)  # type: ignore[arg-type]
        if col < 0 or col >= COLS:
            raise ValueError(f"Invalid column {col}, must be in 0..{COLS - 1}")

        # Find lowest empty row in this column
        col_data = self.board[:, col]
        empty_rows = np.where(col_data == 0)[0]
        if len(empty_rows) == 0:
            raise ValueError(f"Column {col} is full")

        row = empty_rows[-1]  # Lowest empty row
        piece = 1 if self.to_play == 0 else -1
        self.board[row, col] = piece

        # Check winner
        winner = self._check_winner(row, col)

        if winner is not None:
            reward = (
                np.array([1.0, -1.0], dtype=np.float32)
                if winner == 0
                else np.array([-1.0, 1.0], dtype=np.float32)
            )
            terminated = True
        elif self._is_draw():
            reward = np.array([0.0, 0.0], dtype=np.float32)
            terminated = True
        else:
            reward = np.array([0.0, 0.0], dtype=np.float32)
            terminated = False

        # Switch player
        self.to_play = 1 - self.to_play

        return TimeStep(
            observation=self.board.copy(),
            reward=reward,
            terminated=terminated,
            truncated=False,
            to_play=self.to_play,
        )

    def _check_winner(self, row: int, col: int) -> int | None:
        """Check if the last move at (row, col) created a win.

        Checks horizontal, vertical, and both diagonals for 4 in a row.

        Args:
            row: Row of the last placed piece.
            col: Column of the last placed piece.

        Returns:
            Winner player index (0 or 1), or None if no winner.
        """
        piece = self.board[row, col]
        if piece == 0:
            return None

        player = 0 if piece == 1 else 1

        # Directions: (dr, dc)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            # Positive direction
            r, c = row + dr, col + dc
            while 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == piece:
                count += 1
                r += dr
                c += dc
            # Negative direction
            r, c = row - dr, col - dc
            while 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == piece:
                count += 1
                r -= dr
                c -= dc
            if count >= 4:
                return player

        return None

    def _is_draw(self) -> bool:
        """Check if the board is full (draw)."""
        return not np.any(self.board == 0)

    def legal_actions(self) -> LegalActions:
        """Return a boolean mask of legal actions (non-full columns).

        Returns:
            Boolean array of shape [7], True where column is not full.
        """
        return self.board[0, :] == 0  # Top row == 0 means column not full

    def current_player(self) -> int:
        """Return the current player index.

        Returns:
            0 or 1.
        """
        return self.to_play

    def num_players(self) -> int:
        """Return the number of players.

        Returns:
            2
        """
        return 2

    def action_space_spec(self) -> ActionSpaceSpec:
        """Return the action space specification.

        Returns:
            ActionSpaceSpec with 7 discrete actions.
        """
        return ActionSpaceSpec(type="discrete", n=N_ACTIONS)

    def observation_space_spec(self) -> ObservationSpaceSpec:
        """Return the observation space specification.

        Returns:
            ObservationSpaceSpec for [6, 7] float32 board.
        """
        return ObservationSpaceSpec(shape=(ROWS, COLS), dtype="float32")

    def render(self, mode: str = "human") -> str:
        """Render the current board as a string.

        Args:
            mode: Rendering mode. Currently only "human" string output.

        Returns:
            String representation of the board.
        """
        chars = {0: ".", 1: "X", -1: "O"}
        rows = []
        for i in range(ROWS):
            row = " ".join(chars[int(self.board[i, j])] for j in range(COLS))
            rows.append(row)
        rows.append("-" * (COLS * 2 - 1))
        rows.append(" ".join(str(j) for j in range(COLS)))
        return "\n".join(rows)

    def close(self) -> None:
        """No-op for Connect4."""
        pass
