"""Tests for CheckpointMetadata and checkpoint roundtrip."""

from __future__ import annotations

from pathlib import Path

import torch

from muzero.training.checkpoint import (
    CheckpointManager,
    CheckpointState,
    build_checkpoint_metadata,
    compute_config_hash,
)


def test_checkpoint_metadata_required_fields() -> None:
    """Verify CheckpointMetadata can be created with required fields."""
    meta = build_checkpoint_metadata(
        env_id="CartPole-v1",
        network_type="mlp",
        observation_shape=(4,),
        action_space={"type": "discrete", "n": 2},
        num_players=1,
        training_steps=100,
        algorithm="muzero",
        project_name="test",
    )
    assert meta.format_version == 1
    assert meta.env_id == "CartPole-v1"
    assert meta.num_players == 1
    assert meta.training_steps == 100


def test_config_hash_stable_for_same_config() -> None:
    """Verify config_hash is deterministic for same data."""
    data = {"a": 1, "b": 2}
    h1 = compute_config_hash(data)
    h2 = compute_config_hash(dict(data))
    assert h1 == h2


def test_checkpoint_save_load_roundtrip(tmp_path: Path) -> None:
    """Verify checkpoint save and load preserves data."""
    mgr = CheckpointManager()
    path = tmp_path / "test.pt"

    state = CheckpointState(
        model_state_dict={"layer.weight": torch.ones(2, 2)},
        step=42,
        metadata={"algo": "muzero"},
    )
    mgr.save(state, path)
    loaded = mgr.load(path)

    assert loaded.step == 42
    assert loaded.metadata.get("algo") == "muzero"


def test_checkpoint_inspect(tmp_path: Path) -> None:
    """Verify inspect reads metadata from checkpoint."""
    mgr = CheckpointManager()
    path = tmp_path / "test.pt"

    meta = build_checkpoint_metadata(
        env_id="CartPole-v1",
        network_type="mlp",
        observation_shape=(4,),
        action_space={"type": "discrete", "n": 2},
        num_players=1,
        training_steps=50,
    )
    state = CheckpointState(
        model_state_dict={},
        metadata=meta.model_dump(),
    )
    mgr.save(state, path)

    inspected = mgr.inspect(path)
    assert inspected.env_id == "CartPole-v1"
    assert inspected.training_steps == 50


def test_checkpoint_export(tmp_path: Path) -> None:
    """Verify export creates model.pt, config.yaml, metadata.yaml, README.md."""
    mgr = CheckpointManager()
    ckpt_path = tmp_path / "ckpt.pt"
    out_dir = tmp_path / "exported"

    state = CheckpointState(
        model_state_dict={"w": torch.zeros(1)},
        step=10,
        config={"project": {"name": "test"}},
        metadata={"algorithm": "muzero"},
    )
    mgr.save(state, ckpt_path)
    mgr.export(ckpt_path, out_dir)

    assert (out_dir / "model.pt").exists()
    assert (out_dir / "config.yaml").exists()
    assert (out_dir / "metadata.yaml").exists()
    assert (out_dir / "README.md").exists()
