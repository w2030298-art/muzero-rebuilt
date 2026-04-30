"""Environment adapter layer.

Converts Gymnasium, board games, and continuous control environments
to a unified ``GameAdapter`` interface.
"""

from muzero.envs.base import GameAdapter
from muzero.envs.factory import EnvFactory
from muzero.envs.gymnasium_adapter import GymnasiumAdapter

__all__ = ["EnvFactory", "GameAdapter", "GymnasiumAdapter"]
