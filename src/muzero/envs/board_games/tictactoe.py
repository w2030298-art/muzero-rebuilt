"""TicTacToe environment — two-player zero-sum board game.

Board: 3x3 grid. Player 0 marks 1, Player 1 marks -1.
Actions: 0-8 (row-major). Win rewards: +1 (winner), -1 (loser), 0 (draw).
"""

from __future__ import annotations

import numpy as np

from muzero.core.specs import ActionSpaceSpec, ObservationSpaceSpec
from muzero.core.types import Action, LegalActions, TimeStep

BOARD_SIZE = 3
N_ACTIONS = BOARD_SIZE * BOARD_SIZE  # 9


class TicTacToeEnv:
    """TicTacToe (Noughts and Crosses) for two players.

    Player 0 places ``1``, Player 1 places ``-1``.
    Observation is a ``[3, 3]`` board with values ``{0, 1, -1}``.
    Reward is a ``[2]`` numpy vector: ``[r_p0, r_p1]``.

    Attributes:
        board: Current board state, shape [3, 3].
        to_play: Current player (0 or 1).
    """

    def __init__(self) -> None:
        self.board: np.ndarray = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self.to_play: int = 0

    def reset(self, seed: int | None = None) -> TimeStep:
        """Reset the board to empty.

        Args:
            seed: Optional random seed (used for reproducibility by NumPy).

        Returns:
            Initial TimeStep with empty board.
        """
        if seed is not None:
            np.random.seed(seed)
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self.to_play = 0
        return TimeStep(
            observation=self.board.copy(),
            reward=np.array([0.0, 0.0], dtype=np.float32),
            terminated=False,
            truncated=False,
            to_play=0,
        )

    def step(self, action: Action) -> TimeStep:
        """Place a piece at the given position.

        Args:
            action: Integer 0-8 representing board position (row-major).

        Returns:
            TimeStep with updated board and reward.

        Raises:
            ValueError: If the action is illegal (cell already occupied).
        """
        action = int(action)  # type: ignore[arg-type]
        row, col = divmod(action, BOARD_SIZE)

        if self.board[row, col] != 0:
            raise ValueError(f"Illegal action {action}: cell ({row},{col}) already occupied")

        # Place piece
        piece = 1 if self.to_play == 0 else -1
        self.board[row, col] = piece

        # Check winner
        winner = self._check_winner()

        if winner is not None:
            # Terminal: winner gets +1, loser gets -1
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

    def _check_winner(self) -> int | None:
        """Check if there is a winner.

        Returns:
            Winner player index (0 or 1), or None if no winner.
        """
        b = self.board
        # Rows
        for i in range(BOARD_SIZE):
            if abs(b[i, :].sum()) == BOARD_SIZE and np.all(b[i, :] == b[i, 0]):
                return 0 if b[i, 0] == 1 else 1
        # Columns
        for j in range(BOARD_SIZE):
            if abs(b[:, j].sum()) == BOARD_SIZE and np.all(b[:, j] == b[0, j]):
                return 0 if b[0, j] == 1 else 1
        # Diagonals
        if abs(np.trace(b)) == BOARD_SIZE and np.all(np.diag(b) == b[0, 0]):
            return 0 if b[0, 0] == 1 else 1
        if abs(np.trace(np.fliplr(b))) == BOARD_SIZE and np.all(np.diag(np.fliplr(b)) == b[0, -1]):
            return 0 if b[0, -1] == 1 else 1

        return None

    def _is_draw(self) -> bool:
        """Check if the board is full (draw)."""
        return not np.any(self.board == 0)

    def legal_actions(self) -> LegalActions:
        """Return a boolean mask of legal actions.

        Returns:
            Boolean array of shape [9], True where cell is empty.
        """
        flat = self.board.flatten()
        return flat == 0  # type: ignore[return-value]

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
            ActionSpaceSpec with 9 discrete actions.
        """
        return ActionSpaceSpec(type="discrete", n=N_ACTIONS)

    def observation_space_spec(self) -> ObservationSpaceSpec:
        """Return the observation space specification.

        Returns:
            ObservationSpaceSpec for [3, 3] float32 board.
        """
        return ObservationSpaceSpec(shape=(BOARD_SIZE, BOARD_SIZE), dtype="float32")

    def render(self, mode: str = "human") -> str:
        """Render the current board as a string.

        Args:
            mode: Rendering mode. Currently only "human" string output.

        Returns:
            String representation of the board.
        """
        chars = {0: ".", 1: "X", -1: "O"}
        rows = []
        for i in range(BOARD_SIZE):
            row = " ".join(chars[int(self.board[i, j])] for j in range(BOARD_SIZE))
            rows.append(row)
        return "\n".join(rows)

    def close(self) -> None:
        """No-op for TicTacToe."""
        pass
