"""Optimizer factory for creating AdamW optimizers."""

from __future__ import annotations

import torch
import torch.nn as nn

from muzero.config.schema import TrainingConfig


class OptimizerFactory:
    """Creates optimizers and optional learning rate schedulers."""

    def create(self, model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
        """Create an AdamW optimizer with the given config.

        Args:
            model: Neural network model.
            config: Training configuration (lr, weight_decay).

        Returns:
            AdamW optimizer.
        """
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,  # noqa: ARG002
    ) -> object | None:
        """Create a learning rate scheduler (returns None in v1).

        Args:
            optimizer: The optimizer to wrap.
            config: Training configuration.

        Returns:
            None (no scheduler in first version).
        """
        return None
