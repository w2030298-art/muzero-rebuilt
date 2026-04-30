"""Core type aliases and dataclasses shared across all modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---- Type aliases ----

Observation = np.ndarray
"""Observation from an environment, typically a numpy array."""

Action = int | np.ndarray
"""An action: integer for discrete spaces, ndarray for continuous."""

Reward = float | np.ndarray
"""Reward: float for single-player, ndarray for multi-player."""

Value = float | np.ndarray
"""Value: float for scalar, ndarray for multi-player vector."""

LegalActions = np.ndarray | None
"""Legal action mask or None if no mask is available."""

PolicyTarget = np.ndarray
"""Policy target distribution over actions."""

Mask = np.ndarray | None
"""Mask tensor for valid/invalid transitions."""

# ---- Dataclasses ----


@dataclass(slots=True)
class TimeStep:
    """A single step in an environment trajectory.

    Preserves Gymnasium's ``terminated`` / ``truncated`` distinction.
    ``done`` is a convenience property returning ``terminated or truncated``.
    """

    observation: Observation
    reward: Reward
    terminated: bool
    truncated: bool
    to_play: int
    legal_actions: LegalActions = None
    info: dict[str, Any] = field(default_factory=lambda: dict())  # type: ignore[arg-type]

    @property
    def done(self) -> bool:
        """True if the episode ended (either naturally or by time limit)."""
        return self.terminated or self.truncated


@dataclass(slots=True)
class SearchMetadata:
    """Metadata collected during MCTS search."""

    root_value: Value
    search_depth: int
    num_expanded_nodes: int
    extra: dict[str, Any] = field(default_factory=lambda: dict())  # type: ignore[arg-type]


@dataclass(slots=True)
class SearchResult:
    """Result of an MCTS search at a single root state."""

    action: Action
    root_value: Value
    visit_counts: np.ndarray
    policy_target: PolicyTarget
    search_depth: int = 0
    num_expanded_nodes: int = 0
    metadata: dict[str, Any] = field(default_factory=lambda: dict())  # type: ignore[arg-type]
    sampled_actions: np.ndarray | None = None


@dataclass(slots=True)
class TargetSequence:
    """Training targets for an unrolled sequence.

    Each field is a stacked tensor across unroll steps [K, ...].
    """

    observations: np.ndarray
    """Shape: [num_unroll_steps + 1, *obs_shape]."""

    actions: np.ndarray
    """Shape: [num_unroll_steps, ...]."""

    target_values: np.ndarray
    """Shape: [num_unroll_steps + 1, ...] - n-step bootstrapped values."""

    target_rewards: np.ndarray
    """Shape: [num_unroll_steps, ...] - immediate rewards."""

    target_policies: np.ndarray
    """Shape: [num_unroll_steps + 1, action_dim]."""

    target_value_prefixes: np.ndarray | None = None
    """Shape: [num_unroll_steps, ...] - value prefixes (EfficientZero)."""

    masks: np.ndarray | None = None
    """Shape: [num_unroll_steps + 1] - valid position mask."""


@dataclass(slots=True)
class TrainingBatch:
    """A batch of training data sampled from the replay buffer."""

    observations: np.ndarray
    actions: np.ndarray
    target_values: np.ndarray
    target_rewards: np.ndarray
    target_policies: np.ndarray
    importance_weights: np.ndarray | None = None
    indices: np.ndarray | None = None
    masks: np.ndarray | None = None
    target_value_prefixes: np.ndarray | None = None
    to_play: np.ndarray | None = None
