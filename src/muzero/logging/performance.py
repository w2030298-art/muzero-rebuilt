"""Performance tracker for benchmarking components."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class PerformanceSummary:
    """Aggregated performance metrics."""

    inference_latency_ms_p50: float = 0.0
    inference_latency_ms_p95: float = 0.0
    self_play_steps_per_sec: float = 0.0
    training_steps_per_sec: float = 0.0
    gpu_memory_allocated_mb: float | None = None


class PerformanceTracker:
    """Tracks performance metrics during training."""

    def __init__(self, window_size: int = 100) -> None:
        self._inference_latencies: deque[float] = deque(maxlen=window_size)
        self._self_play_start: float | None = None
        self._self_play_steps: int = 0
        self._train_start: float | None = None
        self._train_steps: int = 0

    def record_inference_latency(self, ms: float, batch_size: int = 1) -> None:  # noqa: ARG002
        """Record an inference latency measurement.

        Args:
            ms: Latency in milliseconds.
            batch_size: Batch size of the inference call.
        """
        self._inference_latencies.append(ms)

    def record_self_play_steps(self, count: int) -> None:
        """Record self-play step count.

        Args:
            count: Number of steps completed.
        """
        if self._self_play_start is None:
            self._self_play_start = time.perf_counter()
        self._self_play_steps += count

    def record_training_steps(self, count: int) -> None:
        """Record training step count.

        Args:
            count: Number of steps completed.
        """
        if self._train_start is None:
            self._train_start = time.perf_counter()
        self._train_steps += count

    def record_gpu_memory(self, step: int) -> None:  # noqa: ARG002
        """Record GPU memory usage (no-op on CPU, placeholder for CUDA)."""
        pass  # GPU memory tracking requires CUDA at runtime

    def summary(self) -> PerformanceSummary:
        """Compute aggregated performance summary.

        Returns:
            PerformanceSummary with percentile latencies and throughput.
        """
        import numpy as np

        p50, p95 = 0.0, 0.0
        if self._inference_latencies:
            arr = np.array(list(self._inference_latencies))
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))

        sps, tps = 0.0, 0.0
        if self._self_play_start and self._self_play_steps > 0:
            elapsed = time.perf_counter() - self._self_play_start
            if elapsed > 0:
                sps = self._self_play_steps / elapsed
        if self._train_start and self._train_steps > 0:
            elapsed = time.perf_counter() - self._train_start
            if elapsed > 0:
                tps = self._train_steps / elapsed

        return PerformanceSummary(
            inference_latency_ms_p50=p50,
            inference_latency_ms_p95=p95,
            self_play_steps_per_sec=sps,
            training_steps_per_sec=tps,
        )
