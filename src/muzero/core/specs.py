"""Action space and observation space specifications."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ActionSpaceSpec(BaseModel):
    """Specification for an environment's action space.

    Supports discrete, continuous, and sampled action spaces.
    """

    type: Literal["discrete", "continuous", "sampled"]
    """Action space type: discrete (n categories), continuous (box), or sampled."""

    n: int | None = None
    """Number of discrete actions (for discrete spaces)."""

    shape: tuple[int, ...] | None = None
    """Shape of continuous action vector (for continuous/sampled spaces)."""

    low: list[float] | None = None
    """Lower bounds for continuous actions."""

    high: list[float] | None = None
    """Upper bounds for continuous actions."""


class ObservationSpaceSpec(BaseModel):
    """Specification for an environment's observation space."""

    shape: tuple[int, ...]
    """Shape of the observation tensor."""

    dtype: str = "float32"
    """Data type of observation values."""

    low: float | None = None
    """Minimum observation value."""

    high: float | None = None
    """Maximum observation value."""


class EnvironmentSpec(BaseModel):
    """Full specification describing an environment's spaces and player count."""

    observation: ObservationSpaceSpec
    """Observation space specification."""

    action: ActionSpaceSpec
    """Action space specification."""

    num_players: int = Field(default=1, ge=1)
    """Number of players in this environment."""
