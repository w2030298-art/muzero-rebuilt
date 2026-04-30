"""Checkpoint manager for saving and loading training state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel


class CheckpointMetadata(BaseModel):
    """Metadata describing a checkpoint for inspection and export."""

    format_version: int = 1
    project_name: str = ""
    algorithm: Literal["muzero", "efficientzero", "sampled_muzero", "gumbel_muzero"] = "muzero"
    env_id: str = ""
    network_type: Literal["mlp", "residual", "conv"] = "mlp"
    observation_shape: list[int] = []
    action_space: dict[str, Any] = {}
    num_players: int = 1
    training_steps: int = 0
    created_at: str = ""
    git_commit: str | None = None
    config_hash: str = ""


def build_checkpoint_metadata(
    env_id: str,
    network_type: str,
    observation_shape: tuple[int, ...],
    action_space: dict[str, Any],
    num_players: int,
    training_steps: int,
    algorithm: str = "muzero",
    project_name: str = "muzero",
    config_data: dict[str, Any] | None = None,
) -> CheckpointMetadata:
    """Build a CheckpointMetadata instance from training state.

    Args:
        env_id: Environment ID.
        network_type: Network architecture type.
        observation_shape: Observation space shape.
        action_space: Action space description.
        num_players: Number of players.
        training_steps: Current training step.
        algorithm: Algorithm name.
        project_name: Project name.
        config_data: Full config for computing config_hash.

    Returns:
        CheckpointMetadata instance.
    """
    config_hash = ""
    if config_data is not None:
        config_hash = compute_config_hash(config_data)

    return CheckpointMetadata(
        format_version=1,
        project_name=project_name,
        algorithm=algorithm,  # type: ignore[arg-type]
        env_id=env_id,
        network_type=network_type,  # type: ignore[arg-type]
        observation_shape=list(observation_shape),
        action_space=action_space,
        num_players=num_players,
        training_steps=training_steps,
        created_at=datetime.now(UTC).isoformat(),
        config_hash=config_hash,
    )


def compute_config_hash(config_data: dict[str, Any]) -> str:
    """Compute a SHA-256 hash of JSON-serialized config data.

    Args:
        config_data: Configuration dictionary.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    json_str = json.dumps(config_data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class CheckpointState:
    """Full training state to save/restore.

    Attributes:
        model_state_dict: Model parameters.
        optimizer_state_dict: Optimizer state.
        config: Configuration dictionary.
        step: Training step number.
        metadata: Additional metadata.
    """

    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Save and load training checkpoints using torch.save/load."""

    def save(self, state: CheckpointState, path: Path) -> Path:  # noqa: ARG002
        """Save checkpoint state to disk.

        Args:
            state: Checkpoint state to save.
            path: Output file path.

        Returns:
            The output path (may differ from input if auto-named).
        """
        actual_path = path
        actual_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": state.model_state_dict,
                "optimizer_state_dict": state.optimizer_state_dict,
                "config": state.config,
                "step": state.step,
                "metadata": state.metadata,
            },
            actual_path,
        )
        return actual_path

    def load(self, path: Path, map_location: str | torch.device = "cpu") -> CheckpointState:
        """Load checkpoint from disk.

        Args:
            path: Path to checkpoint file.
            map_location: Device to map tensors to.

        Returns:
            CheckpointState with loaded data.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        data: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)

        return CheckpointState(
            model_state_dict=data.get("model_state_dict", {}),
            optimizer_state_dict=data.get("optimizer_state_dict"),
            config=data.get("config", {}),
            step=int(data.get("step", 0)),
            metadata=data.get("metadata", {}),
        )

    def inspect(self, path: Path) -> CheckpointMetadata:
        """Read checkpoint metadata without loading model weights.

        Args:
            path: Path to checkpoint file.

        Returns:
            CheckpointMetadata with training info.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        data: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        meta = data.get("metadata", {})
        return CheckpointMetadata(**meta) if isinstance(meta, dict) else CheckpointMetadata()

    def export(self, path: Path, out_dir: Path) -> None:
        """Export checkpoint to a structured directory.

        Produces: model.pt, config.yaml, metadata.yaml, README.md.

        Args:
            path: Path to checkpoint file.
            out_dir: Output directory.
        """
        import yaml

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        out_dir.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)

        # Save model weights
        torch.save(data.get("model_state_dict", {}), out_dir / "model.pt")

        # Save config
        config = data.get("config", {})
        with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False)

        # Save metadata
        meta = data.get("metadata", {})
        with open(out_dir / "metadata.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, default_flow_style=False)

        # Save README
        step = data.get("step", 0)
        readme = (
            f"# MuZero Checkpoint\n\n"
            f"- Algorithm: {meta.get('algorithm', 'unknown')}\n"
            f"- Environment: {meta.get('env_id', 'unknown')}\n"
            f"- Training Steps: {step}\n"
            f"- Network: {meta.get('network_type', 'mlp')}\n"
            f"- Players: {meta.get('num_players', 1)}\n"
        )
        with open(out_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(readme)

    def import_checkpoint(self, path: Path) -> CheckpointState:
        """Load checkpoint (alias for load).

        Args:
            path: Path to checkpoint file.

        Returns:
            CheckpointState.
        """
        return self.load(path)
