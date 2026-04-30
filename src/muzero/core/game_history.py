"""Game history: records the trajectory of a self-play episode.

Stores observations, actions, rewards, player indices, root values, visit distributions,
legal action masks, and search metadata at each step.
"""

from __future__ import annotations

import numpy as np

from muzero.core.types import (
    Action,
    Mask,
    PolicyTarget,
    Reward,
    SearchMetadata,
    SearchResult,
    TimeStep,
    Value,
)


class GameHistory:
    """Records the full trajectory of a self-play episode.

    Each step records the observation before the action, the action taken,
    the resulting reward, the current player, the MCTS search result,
    and any legal action mask.
    """

    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[Action] = []
        self.rewards: list[Reward] = []
        self.players: list[int] = []
        self.root_values: list[Value] = []
        self.child_visit_distributions: list[PolicyTarget] = []
        self.legal_action_masks: list[Mask] = []
        self.search_metadata: list[SearchMetadata] = []
        self._to_play_history: list[int] = []

    def append_initial(self, timestep: TimeStep) -> None:
        """Record the initial observation and player before any action.

        Should be called once at the start of an episode.

        Args:
            timestep: The initial timestep from ``env.reset()``.
        """
        self.observations.append(timestep.observation)
        self.players.append(timestep.to_play)
        # No action, reward, or search result for the initial state yet

    def append(self, action: Action, timestep: TimeStep, search_result: SearchResult) -> None:
        """Record an action, its resulting timestep, and the MCTS search result.

        Args:
            action: The action taken after the search.
            timestep: The timestep returned by ``env.step(action)``.
            search_result: The MCTS result used to select this action.
        """
        self.actions.append(action)
        self.rewards.append(timestep.reward)
        self.players.append(timestep.to_play)
        self.root_values.append(search_result.root_value)
        self.child_visit_distributions.append(search_result.policy_target)
        self.legal_action_masks.append(timestep.legal_actions)
        self.search_metadata.append(
            SearchMetadata(
                root_value=search_result.root_value,
                search_depth=search_result.search_depth,
                num_expanded_nodes=search_result.num_expanded_nodes,
            )
        )
        self.observations.append(timestep.observation)

    def __len__(self) -> int:
        """Number of actions taken (number of steps)."""
        return len(self.actions)
