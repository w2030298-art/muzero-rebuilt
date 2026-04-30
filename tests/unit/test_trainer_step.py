"""
Tests for Trainer.train_step with synthetic data.
"""

from __future__ import annotations

import numpy as np
import torch

from muzero.config.schema import EnvironmentConfig, MuZeroConfig
from muzero.core.types import TrainingBatch
from muzero.models.mlp import MLPNetwork
from muzero.training.losses import MuZeroLoss
from muzero.training.optimizer import OptimizerFactory
from muzero.training.trainer import Trainer


def test_train_step_runs() -> None:
    """Verify train_step runs without error on synthetic data."""
    network = MLPNetwork(observation_shape=(4,), action_space_size=2, hidden_size=32)

    env_cfg = EnvironmentConfig(type="gymnasium", id="CartPole-v1")
    cfg = MuZeroConfig(environment=env_cfg)

    optimizer = OptimizerFactory().create(network, cfg.training)
    loss_fn = MuZeroLoss()
    trainer = Trainer(network, optimizer, loss_fn, cfg, torch.device("cpu"))

    # Create synthetic batch
    B, K = 2, 3
    batch = TrainingBatch(
        observations=np.random.randn(B, K + 1, 4).astype(np.float32),
        actions=np.zeros((B, K), dtype=np.int64),
        target_values=np.random.randn(B, K + 1).astype(np.float32),
        target_rewards=np.random.randn(B, K).astype(np.float32),
        target_policies=np.ones((B, K + 1, 2), dtype=np.float32) / 2,
    )

    result = trainer.train_step(batch)
    assert result.step == 1
    assert result.loss_total > 0
