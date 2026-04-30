"""Configuration schema definitions using Pydantic v2."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProjectConfig(BaseModel):
    """Top-level project configuration."""

    name: str = "muzero"
    seed: int = 0
    output_dir: Path = Field(default=Path("runs"))


class AlgorithmConfig(BaseModel):
    """Algorithm selection and feature flags."""

    name: Literal["muzero", "efficientzero", "sampled_muzero", "gumbel_muzero"] = "muzero"
    use_value_prefix: bool = False
    use_consistency_loss: bool = False
    use_reanalyze: bool = False
    num_players: int = Field(default=1, ge=1)


class EnvironmentConfig(BaseModel):
    """Environment specification."""

    type: Literal["gymnasium", "board_game", "continuous_control"]
    id: str
    max_episode_steps: int | None = None


class NetworkConfig(BaseModel):
    """Network architecture configuration."""

    type: Literal["mlp", "residual", "conv"] = "mlp"
    hidden_size: int = 128
    num_blocks: int = 2
    support_size: int = 10


class SearchConfig(BaseModel):
    """MCTS search configuration."""

    num_simulations: int = Field(default=25, ge=1)
    pb_c_base: float = 19652
    pb_c_init: float = 1.25
    discount: float = Field(default=0.997, gt=0, le=1)
    dirichlet_alpha: float = 0.25
    root_exploration_fraction: float = 0.25
    temperature: float = 1.0
    action_sampler_type: Literal["discrete", "continuous"] = "discrete"
    num_sampled_actions: int = 16


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    batch_size: int = Field(default=64, ge=1)
    max_steps: int = Field(default=1000, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = 1e-4
    unroll_steps: int = Field(default=5, ge=1)
    td_steps: int = Field(default=10, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    precision: Literal["fp32", "amp_fp16", "amp_bf16"] = "fp32"
    compile_model: bool = False


class ReplayConfig(BaseModel):
    """Replay buffer configuration."""

    capacity: int = Field(default=50000, ge=1)
    prioritized: bool = True
    alpha: float = 0.6
    beta: float = 0.4


class ExecutionConfig(BaseModel):
    """Execution backend and device configuration."""

    backend: Literal["local", "ray"] = "local"
    device: Literal["cpu", "cuda"] = "cpu"
    num_self_play_workers: int = Field(default=1, ge=1)
    num_envs_per_worker: int = Field(default=1, ge=1)
    inference_batch_size: int = Field(default=16, ge=1)


class CheckpointConfig(BaseModel):
    """Checkpoint management configuration."""

    save_interval_steps: int = 1000
    keep_last: int = 5
    resume_path: Path | None = None


class LoggingConfig(BaseModel):
    """Logging configuration."""

    tensorboard: bool = True
    log_interval_steps: int = 100


class MuZeroConfig(BaseModel):
    """Root configuration aggregating all sub-configs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    environment: EnvironmentConfig
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
