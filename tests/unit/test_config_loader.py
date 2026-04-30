"""Tests for the configuration loader and schema."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from muzero.config.loader import ConfigLoader


def test_load_cartpole_cpu_debug() -> None:
    """Verify that loading CartPole config with cpu_debug profile overrides correctly."""
    loader = ConfigLoader()
    cfg = loader.load(Path("configs/cartpole_muzero.yaml"), profile="cpu_debug")

    assert cfg.execution.device == "cpu"
    assert cfg.training.batch_size == 8
    assert cfg.training.max_steps == 10
    assert cfg.training.precision == "fp32"
    assert cfg.search.num_simulations == 5
    assert cfg.replay.capacity == 100
    assert cfg.environment.id == "CartPole-v1"
    assert cfg.algorithm.num_players == 1


def test_recursive_merge_preserves_missing_keys() -> None:
    """Verify that merge preserves keys present in base but absent in override."""
    loader = ConfigLoader()
    base = {"a": 1, "b": {"x": 10, "y": 20}, "c": 3}
    override = {"b": {"y": 99}}

    result = loader.merge(base, override)

    assert result["a"] == 1  # preserved
    assert result["b"]["x"] == 10  # preserved
    assert result["b"]["y"] == 99  # overridden
    assert result["c"] == 3  # preserved


def test_recursive_merge_does_not_mutate_inputs() -> None:
    """Verify that merge returns a new dict and does not mutate inputs."""
    loader = ConfigLoader()
    base = {"a": 1}
    override = {"a": 2}

    base_copy = dict(base)
    override_copy = dict(override)

    result = loader.merge(base, override)

    assert base == base_copy
    assert override == override_copy
    assert result is not base
    assert result["a"] == 2


def test_dotlist_override_training_batch_size() -> None:
    """Verify dot-notation override sets nested config values correctly."""
    loader = ConfigLoader()
    data = {"training": {"batch_size": 64, "max_steps": 1000}}

    result = loader.apply_dotlist_overrides(data, {"training.batch_size": 128})

    assert result["training"]["batch_size"] == 128
    assert result["training"]["max_steps"] == 1000  # unchanged


def test_dotlist_override_deeply_nested() -> None:
    """Verify dot-notation override works with deeply nested paths."""
    loader = ConfigLoader()
    data = {"a": {"b": {"c": 1}}}

    result = loader.apply_dotlist_overrides(data, {"a.b.c": 42, "a.b.d": 99})

    assert result["a"]["b"]["c"] == 42
    assert result["a"]["b"]["d"] == 99


def test_laptop_profile_sets_cuda_and_amp() -> None:
    """Verify that laptop_rtx4060_8gb profile enables CUDA and AMP."""
    loader = ConfigLoader()
    cfg = loader.load(Path("configs/cartpole_muzero.yaml"), profile="laptop_rtx4060_8gb")

    assert cfg.execution.device == "cuda"
    assert cfg.training.precision == "amp_fp16"
    assert cfg.training.compile_model is False
    assert cfg.execution.inference_batch_size == 16
    assert cfg.search.num_simulations == 25


def test_invalid_algorithm_name_raises_validation_error() -> None:
    """Verify that an invalid algorithm name raises a ValidationError."""
    loader = ConfigLoader()
    data = {
        "project": {"name": "test"},
        "environment": {"type": "gymnasium", "id": "CartPole-v1"},
        "algorithm": {"name": "invalid_algorithm"},
    }

    with pytest.raises(ValidationError):
        loader.validate(data)


def test_load_yaml_file_not_found() -> None:
    """Verify that loading a non-existent file raises FileNotFoundError."""
    loader = ConfigLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_yaml(Path("configs/nonexistent.yaml"))


def test_dump_resolved_creates_file(tmp_path: Path) -> None:
    """Verify that dump_resolved writes a valid YAML file."""
    loader = ConfigLoader()
    cfg = loader.load(Path("configs/cartpole_muzero.yaml"), profile="cpu_debug")

    out_path = tmp_path / "resolved.yaml"
    loader.dump_resolved(cfg, out_path)

    assert out_path.exists()
    # Verify the dumped file can be re-loaded
    reloaded = loader.load(out_path)
    assert reloaded.project.name == cfg.project.name
    assert reloaded.environment.id == cfg.environment.id


def test_no_profile_uses_config_defaults() -> None:
    """Verify that loading without a profile uses the config file values."""
    loader = ConfigLoader()
    cfg = loader.load(Path("configs/cartpole_muzero.yaml"))

    # CartPole config doesn't set execution, so defaults apply
    assert cfg.execution.device == "cpu"  # schema default
    assert cfg.training.batch_size == 64  # base.yaml default


def test_algorithm_config_extra() -> None:
    """Verify that algorithm config files can be loaded and merged."""
    loader = ConfigLoader()
    cfg = loader.load(Path("configs/cartpole_muzero.yaml"), profile="cpu_debug")

    assert cfg.algorithm.name == "muzero"
    assert cfg.algorithm.use_value_prefix is False
    assert cfg.algorithm.use_consistency_loss is False
