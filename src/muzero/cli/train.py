"""Training component assembly for MuZero."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from muzero.config.schema import MuZeroConfig
from muzero.core.specs import EnvironmentSpec
from muzero.envs.base import GameAdapter
from muzero.envs.factory import EnvFactory
from muzero.models.base import BaseMuZeroNetwork
from muzero.models.factory import NetworkFactory
from muzero.replay.buffer import ReplayBuffer
from muzero.replay.prioritized import PrioritizedReplayBuffer
from muzero.replay.target_builder import TargetBuilder
from muzero.search.action_sampler import DiscreteActionSampler
from muzero.search.mcts import MCTS
from muzero.search.policies import PUCTPolicy
from muzero.training.losses import MuZeroLoss
from muzero.training.optimizer import OptimizerFactory
from muzero.training.trainer import Trainer


@dataclass(slots=True)
class TrainingComponents:
    """All assembled components for a training session."""

    env: GameAdapter
    network: BaseMuZeroNetwork
    search: MCTS
    replay: ReplayBuffer
    trainer: Trainer
    config: MuZeroConfig
    optimizer: torch.optim.Optimizer


def build_training_components(config: MuZeroConfig) -> TrainingComponents:
    """Assemble all training components from configuration."""
    # Environment
    env = EnvFactory().make(config.environment)
    env_spec = EnvironmentSpec(
        observation=env.observation_space_spec(),
        action=env.action_space_spec(),
        num_players=env.num_players(),
    )
    action_dim = env_spec.action.n if env_spec.action.n is not None else 1

    # Network
    network = NetworkFactory().create(
        config=config.network,
        env_spec=env_spec,
        algo_config=config.algorithm,
    )

    # Device
    device = torch.device(config.execution.device)
    network.to(device)

    # Search
    action_sampler = DiscreteActionSampler()
    search_policy = PUCTPolicy()
    search = MCTS(
        network=network,
        config=config.search,
        action_sampler=action_sampler,
        search_policy=search_policy,
        device=device,
        num_players=config.algorithm.num_players,
        action_dim=action_dim,
    )

    # Replay
    target_builder = TargetBuilder(discount=config.search.discount)
    if config.replay.prioritized:
        replay: ReplayBuffer = PrioritizedReplayBuffer(
            config=config.replay, target_builder=target_builder
        )
    else:
        replay = ReplayBuffer(config=config.replay, target_builder=target_builder)

    # Trainer
    optimizer = OptimizerFactory().create(network, config.training)
    loss_fn = MuZeroLoss()
    trainer = Trainer(
        network=network,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        device=device,
    )

    return TrainingComponents(
        env=env,
        network=network,
        search=search,
        replay=replay,
        trainer=trainer,
        config=config,
        optimizer=optimizer,
    )
