"""Benchmark CLI command."""

# ruff: noqa: B008

from __future__ import annotations

from pathlib import Path

import typer

from muzero.config.loader import ConfigLoader

app = typer.Typer(name="benchmark")


@app.command()
def main(
    config: Path = typer.Option(..., "--config", help="Path to YAML config file"),
    profile: str = typer.Option("cpu_debug", "--profile", help="Hardware profile name"),
    component: str = typer.Option(
        "inference", "--component", help="Component to benchmark: inference|search"
    ),
    steps: int = typer.Option(10, "--steps", help="Number of steps to run"),
) -> None:
    """Run a benchmark on a specific component."""
    from muzero.benchmark.runner import BenchmarkRunner

    cfg = ConfigLoader().load(config, profile=profile)
    runner = BenchmarkRunner()

    if component == "inference":
        result = runner.benchmark_inference(cfg, steps)
    elif component == "search":
        result = runner.benchmark_search(cfg, steps)
    else:
        print(f"Unknown component: {component}. Supported: inference, search")
        raise typer.Exit(1)

    print(f"Benchmark: {result.component} x{result.steps} steps")
    print(f"  P50 latency: {result.latency_ms_p50:.3f} ms")
    print(f"  P95 latency: {result.latency_ms_p95:.3f} ms")
    print(f"  Throughput:  {result.throughput:.2f} steps/s")
