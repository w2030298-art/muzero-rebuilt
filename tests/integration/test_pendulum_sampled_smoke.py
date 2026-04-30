"""Smoke test: Pendulum with Sampled MuZero (continuous actions)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from muzero.cli.train import build_training_components
from muzero.config.loader import ConfigLoader
from muzero.search.action_sampler import ContinuousActionSampler


def test_pendulum_continuous_action_sampler() -> None:
    """Verify ContinuousActionSampler produces bounded actions."""
    from muzero.envs.gymnasium_adapter import GymnasiumAdapter

    env = GymnasiumAdapter("Pendulum-v1")
    spec = env.action_space_spec()
    assert spec.low is not None and spec.high is not None

    sampler = ContinuousActionSampler(
        action_low=np.array(spec.low, dtype=np.float32),
        action_high=np.array(spec.high, dtype=np.float32),
        sampling_std=0.3,
    )
    policy = torch.zeros(2, 1)  # 2 batch, 1 action dim
    result = sampler.sample(policy, torch.zeros(2, 8), 4, None)
    assert result.actions.shape == (2, 4, 1)
    assert result.priors.shape == (2, 4)
    assert float(result.actions.max()) <= float(spec.high[0])
    assert float(result.actions.min()) >= float(spec.low[0])
    env.close()


def test_pendulum_components_build() -> None:
    """Verify Pendulum sampled config builds components."""
    cfg = ConfigLoader().load(Path("configs/pendulum_sampled_muzero.yaml"), profile="cpu_debug")
    components = build_training_components(cfg)
    assert components.env.num_players() == 1
    components.env.close()
