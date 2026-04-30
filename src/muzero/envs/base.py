"""Protocol defining the unified game adapter interface.

All environments (Gymnasium, board games, continuous control) must implement
this protocol to be used by the MuZero search and training pipeline.
"""

from __future__ import annotations

from typing import Any, Protocol

from muzero.core.specs import ActionSpaceSpec, ObservationSpaceSpec
from muzero.core.types import Action, LegalActions, TimeStep


class GameAdapter(Protocol):
    """Unified interface for all game environments.

    Implementations must handle:
    - Resetting to initial state
    - Stepping with actions
    - Providing legal action masks (or None if N/A)
    - Tracking current player and total player count
    - Exposing action/observation space specifications
    """

    def reset(self, seed: int | None = None) -> TimeStep:
        """Reset the environment to its initial state.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Initial TimeStep with observation and player information.
        """
        ...

    def step(self, action: Action) -> TimeStep:
        """Execute an action and advance the environment.

        Args:
            action: The action to execute.

        Returns:
            TimeStep with new observation, reward, and termination info.
        """
        ...

    def legal_actions(self) -> LegalActions:
        """Return the legal action mask for the current state.

        Returns:
            Boolean array where True indicates a legal action,
            or None if all actions are legal.
        """
        ...

    def current_player(self) -> int:
        """Return the index (0-based) of the player whose turn it is.

        Returns:
            Current player index.
        """
        ...

    def num_players(self) -> int:
        """Return the total number of players in this game.

        Returns:
            Number of players.
        """
        ...

    def action_space_spec(self) -> ActionSpaceSpec:
        """Return the specification of the action space.

        Returns:
            ActionSpaceSpec describing type, size, and bounds.
        """
        ...

    def observation_space_spec(self) -> ObservationSpaceSpec:
        """Return the specification of the observation space.

        Returns:
            ObservationSpaceSpec describing shape, dtype, and bounds.
        """
        ...

    def render(self, mode: str = "human") -> Any:
        """Render the current state.

        Args:
            mode: Rendering mode (e.g., "human", "rgb_array").

        Returns:
            Render output, type depends on mode.
        """
        ...

    def close(self) -> None:
        """Close the environment and release resources."""
        ...
