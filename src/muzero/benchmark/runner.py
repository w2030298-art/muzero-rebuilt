"""Benchmark runner for measuring component performance."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from muzero.cli.train import build_training_components
from muzero.config.schema import MuZeroConfig


@dataclass(slots=True)
class BenchmarkResult:
    """Result of a component benchmark."""

    component: str
    steps: int
    latency_ms_p50: float
    latency_ms_p95: float
    throughput: float
    gpu_memory_mb: float | None = None


class BenchmarkRunner:
    """Runs performance benchmarks for inference and search components."""

    def benchmark_inference(self, config: MuZeroConfig, steps: int) -> BenchmarkResult:
        """Benchmark network inference throughput.

        Args:
            config: MuZero configuration.
            steps: Number of inference calls to measure.

        Returns:
            BenchmarkResult with latency and throughput.
        """
        components = build_training_components(config)
        network = components.network

        import torch

        # Create random input
        obs_dim = int(np.prod(components.env.observation_space_spec().shape))
        obs = torch.randn(1, obs_dim, device=torch.device(config.execution.device))

        latencies: list[float] = []
        for _ in range(steps):
            start = time.perf_counter()
            with torch.no_grad():
                network.initial_inference(obs)
            latencies.append((time.perf_counter() - start) * 1000)

        lat_arr = np.array(latencies)
        components.env.close()

        return BenchmarkResult(
            component="inference",
            steps=steps,
            latency_ms_p50=float(np.percentile(lat_arr, 50)),
            latency_ms_p95=float(np.percentile(lat_arr, 95)),
            throughput=steps / sum(latencies) * 1000,
        )

    def benchmark_search(self, config: MuZeroConfig, steps: int) -> BenchmarkResult:
        """Benchmark MCTS search throughput.

        Args:
            config: MuZero configuration.
            steps: Number of search calls to measure.

        Returns:
            BenchmarkResult with latency and throughput.
        """
        components = build_training_components(config)
        search = components.search
        env = components.env

        ts = env.reset(seed=0)
        legal = env.legal_actions()

        latencies: list[float] = []
        for _ in range(steps):
            start = time.perf_counter()
            search.run(ts.observation, legal, ts.to_play)
            latencies.append((time.perf_counter() - start) * 1000)

        lat_arr = np.array(latencies)
        env.close()

        return BenchmarkResult(
            component="search",
            steps=steps,
            latency_ms_p50=float(np.percentile(lat_arr, 50)),
            latency_ms_p95=float(np.percentile(lat_arr, 95)),
            throughput=steps / sum(latencies) * 1000,
        )
