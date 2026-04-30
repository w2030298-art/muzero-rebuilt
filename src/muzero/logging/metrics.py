"""Metrics logger with TensorBoard support."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MetricsLogger:
    """Logs scalar and histogram metrics to TensorBoard and JSONL.

    Args:
        log_dir: Directory for log output.
        enable_tensorboard: Whether to create a SummaryWriter.
    """

    def __init__(self, log_dir: Path, enable_tensorboard: bool = True) -> None:
        self._log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        self._writer = None
        if enable_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(str(log_dir))

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar value.

        Args:
            name: Metric name (e.g., "train/loss_total").
            value: Scalar value.
            step: Training step.
        """
        if self._writer is not None:
            self._writer.add_scalar(name, value, step)

    def log_histogram(self, name: str, values: np.ndarray, step: int) -> None:
        """Log a histogram of values.

        Args:
            name: Metric name.
            values: Array of values.
            step: Training step.
        """
        if self._writer is not None:
            self._writer.add_histogram(name, values, step)

    def log_text(self, name: str, text: str, step: int) -> None:
        """Log a text message.

        Args:
            name: Metric name.
            text: Text content.
            step: Training step.
        """
        if self._writer is not None:
            self._writer.add_text(name, text, step)

    def flush(self) -> None:
        """Flush buffered data."""
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """Close the logger."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
