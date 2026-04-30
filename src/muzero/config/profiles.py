"""Profile definitions for common hardware and execution scenarios.

Profiles are YAML files in configs/profiles/. This module provides
programmatic access when needed.
"""

from __future__ import annotations


def get_default_profile_names() -> list[str]:
    """Return the list of recognized profile names.

    Returns:
        List of profile name strings.
    """
    return ["cpu_debug", "laptop_rtx4060_8gb", "ray_local"]


def get_profile_description(name: str) -> str:
    """Return a human-readable description of a profile.

    Args:
        name: The profile name.

    Returns:
        Description string.

    Raises:
        ValueError: If the profile name is not recognized.
    """
    descriptions: dict[str, str] = {
        "cpu_debug": "CPU-only debug profile: small batch, few simulations, fp32",
        "laptop_rtx4060_8gb": "RTX 4060 Laptop GPU: CUDA + AMP, moderate batch size",
        "ray_local": "Local Ray cluster: distributed self-play on CPU",
    }
    if name not in descriptions:
        raise ValueError(f"Unknown profile: {name}. Available: {list(descriptions)}")
    return descriptions[name]
