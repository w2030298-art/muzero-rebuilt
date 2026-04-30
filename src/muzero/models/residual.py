"""Residual network for MuZero — stub implementation.

Reserved for board game environments (TicTacToe, Connect4, Gomoku)
that benefit from residual connections.
"""

from __future__ import annotations

from muzero.models.base import BaseMuZeroNetwork


class ResidualNetwork(BaseMuZeroNetwork):
    """Residual network placeholder — not yet implemented.

    Raises NotImplementedError on instantiation.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "ResidualNetwork is reserved for board game environments. "
            "Use MLPNetwork for initial development."
        )
