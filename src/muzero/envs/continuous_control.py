"""Continuous control environment adapter placeholder.

Uses GymnasiumAdapter for continuous control environments like Pendulum.
This module exists for future specialized continuous control adapters.
"""

from muzero.envs.gymnasium_adapter import GymnasiumAdapter

# Continuous control environments currently use GymnasiumAdapter directly.
# This module is reserved for future specialized adapters.
ContinuousControlAdapter = GymnasiumAdapter

__all__ = ["ContinuousControlAdapter"]
