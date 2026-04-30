"""Weight store for synchronized model weights across workers."""

from __future__ import annotations

import copy

import torch


class WeightStore:
    """Thread-safe storage for the latest model weights.

    Workers (self-play, evaluator) pull the latest weights, while
    the trainer pushes updated weights.
    """

    def __init__(self) -> None:
        self._weights: dict[str, torch.Tensor] | None = None
        self._step: int = 0

    def set_weights(self, weights: dict[str, torch.Tensor], step: int) -> None:
        """Store a deep copy of the given weights.

        Args:
            weights: Model weights as CPU tensors.
            step: Current training step.
        """
        self._weights = {k: v.detach().cpu().clone() for k, v in weights.items()}
        self._step = step

    def get_weights(self) -> tuple[dict[str, torch.Tensor] | None, int]:
        """Return a deep copy of the latest weights.

        Returns:
            Tuple of (weights dict, step number). Weights may be None if never set.
        """
        if self._weights is None:
            return None, 0
        return copy.deepcopy(self._weights), self._step
