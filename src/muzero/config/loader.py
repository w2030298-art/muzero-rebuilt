"""YAML configuration loader with profile merging and dotlist overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from muzero.config.schema import MuZeroConfig


class ConfigLoader:
    """Loads, merges, and validates YAML configuration files."""

    def load(
        self,
        path: Path,
        profile: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> MuZeroConfig:
        """Load main config, optionally merge a profile, apply overrides, and validate.

        Args:
            path: Path to the main YAML config file.
            profile: Optional profile name (loads configs/profiles/{profile}.yaml).
            overrides: Optional dot-notation overrides (e.g., {"training.batch_size": 32}).

        Returns:
            Validated MuZeroConfig instance.

        Raises:
            FileNotFoundError: If the config file or profile file does not exist.
            ValueError: If validation fails.
        """
        base_data = self.load_yaml(path)
        if profile is not None:
            profile_path = Path(f"configs/profiles/{profile}.yaml")
            if profile_path.exists():
                profile_data = self.load_yaml(profile_path)
                base_data = self.merge(base_data, profile_data)
        if overrides is not None:
            base_data = self.apply_dotlist_overrides(base_data, overrides)
        return self.validate(base_data)

    def load_yaml(self, path: Path) -> dict[str, Any]:
        """Load and parse a single YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed YAML data as a dictionary.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file cannot be parsed.
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] | None = yaml.safe_load(f)
        return data if data is not None else {}

    def merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge override into base dict.

        Values in override take precedence. Nested dicts are merged recursively.
        Non-dict values in override replace base values entirely.

        Args:
            base: The base configuration dictionary.
            override: The override configuration dictionary.

        Returns:
            A new merged dictionary (does not mutate inputs).
        """
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge(result[key], dict(value))  # type: ignore[arg-type]
            else:
                result[key] = deepcopy(value)
        return result

    def apply_dotlist_overrides(
        self, data: dict[str, Any], overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply dot-notation overrides to a config dictionary.

        Example: ``{"training.batch_size": 32}`` sets ``data["training"]["batch_size"] = 32``.
        Supports nested paths with arbitrary depth.

        Args:
            data: The configuration dictionary to modify.
            overrides: Flat dict with dot-separated keys.

        Returns:
            A new dictionary with overrides applied (does not mutate input).
        """
        result = deepcopy(data)
        for key, value in overrides.items():
            parts = key.split(".")
            target = result
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = deepcopy(value)
        return result

    def validate(self, data: dict[str, Any]) -> MuZeroConfig:
        """Validate config data against the MuZeroConfig schema.

        Args:
            data: Raw configuration dictionary.

        Returns:
            Validated MuZeroConfig instance.

        Raises:
            pydantic.ValidationError: If validation fails.
        """
        return MuZeroConfig(**data)

    def dump_resolved(self, config: MuZeroConfig, out_path: Path) -> None:
        """Write resolved configuration to a YAML file.

        Args:
            config: Resolved MuZeroConfig instance.
            out_path: Output file path.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = config.model_dump(mode="json")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
