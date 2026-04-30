"""Support transform: convert between scalar values and support bins.

The support is a set of evenly-spaced bins used to represent scalar values
as categorical distributions. This is a core component of MuZero's value/reward
prediction.
"""

from __future__ import annotations

import torch


class SupportTransform:
    """Transforms scalar values to/from support (categorical) representations.

    The support range is ``[-support_size, support_size]`` with ``2 * support_size + 1`` bins.
    Values are clamped to this range and projected onto adjacent bins via linear interpolation.

    Args:
        support_size: Half-width of the support range. Total bins = 2 * support_size + 1.
    """

    def __init__(self, support_size: int) -> None:
        if support_size < 1:
            raise ValueError("support_size must be >= 1")
        self._support_size = support_size
        n_bins = 2 * support_size + 1
        self._support: torch.Tensor = torch.linspace(-support_size, support_size, n_bins)

    @property
    def support(self) -> torch.Tensor:
        """The support bin centers, shape ``[n_bins]``."""
        return self._support

    @property
    def num_bins(self) -> int:
        """Total number of support bins."""
        return 2 * self._support_size + 1

    def scalar_to_support(self, x: torch.Tensor) -> torch.Tensor:
        """Convert scalar values to support (categorical) representation.

        Uses linear interpolation between the two nearest bins.
        Values outside ``[-support_size, support_size]`` are clamped.

        Args:
            x: Scalar tensor of any shape.

        Returns:
            Support logits tensor of shape ``[*x.shape, num_bins]``.
        """
        x = torch.clamp(x, -self._support_size, self._support_size)
        # Expand to [*batch, num_bins]
        support = self._support.to(x.device)
        x_expanded = x.unsqueeze(-1)
        # Distance to each bin center
        diff = x_expanded - support  # [*batch, num_bins]
        # Contribution: the closer to a bin, the higher the weight
        # Use linear interpolation: weight = max(0, 1 - |x - bin|)
        weight = torch.clamp(1.0 - torch.abs(diff), 0.0, 1.0)
        # Normalize so it sums to 1
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return weight

    def support_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert support logits to scalar values via softmax expectation.

        Args:
            logits: Support logits tensor of shape ``[*batch, num_bins]``.

        Returns:
            Scalar tensor of shape ``[*batch]``.
        """
        probs = torch.softmax(logits, dim=-1)
        support = self._support.to(logits.device)
        return (probs * support).sum(dim=-1)
