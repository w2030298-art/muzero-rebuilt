"""Convolutional network for MuZero — stub implementation.

Reserved for image-based environments (Atari, visual tasks).
"""

from __future__ import annotations

from muzero.models.base import BaseMuZeroNetwork


class ConvNetwork(BaseMuZeroNetwork):
    """Convolutional network placeholder — not yet implemented.

    Raises NotImplementedError on instantiation.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "ConvNetwork is reserved for image-based environments (Atari). "
            "Use MLPNetwork for initial development."
        )
