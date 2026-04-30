"""CLI entry point for the MuZero framework."""

# ruff: noqa: B008

from __future__ import annotations

from pathlib import Path

import typer

from muzero.cli.train import build_training_components
from muzero.config.loader import ConfigLoader

app = typer.Typer(name="muzero")
# Import and register checkpoint sub-typer
from muzero.cli.checkpoint import app as checkpoint_app  # noqa: E402

app.add_typer(checkpoint_app, name="checkpoint")


@app.command()
def version() -> None:
    """Print the MuZero version."""
    import muzero

    print(f"muzero-rebuilt v{muzero.__version__}")


@app.command()
def train(
    config: Path = typer.Option(..., "--config", help="Path to YAML config file"),
    profile: str = typer.Option("cpu_debug", "--profile", help="Hardware profile name"),
    seed: int = typer.Option(0, "--seed", help="Random seed"),
    resume: Path | None = typer.Option(None, "--resume", help="Resume from checkpoint"),
) -> None:
    """Run a training session."""
    cfg = ConfigLoader().load(config, profile=profile)
    cfg.project.seed = seed

    components = build_training_components(cfg)
    env = components.env
    search = components.search
    replay = components.replay
    trainer = components.trainer
    network = components.network
    optimizer = components.optimizer

    from muzero.execution.self_play_worker import SelfPlayWorker

    # Self-play to fill replay buffer
    worker = SelfPlayWorker(env, search, cfg)
    for ep in range(2):
        history = worker.run_episode(seed=seed + ep)
        replay.add_game(history)

    # Training loop
    max_steps = cfg.training.max_steps
    for step in range(max_steps):
        batch = replay.sample_batch(
            batch_size=cfg.training.batch_size,
            num_unroll_steps=cfg.training.unroll_steps,
            td_steps=cfg.training.td_steps,
        )
        result = trainer.train_step(batch)

        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"Step {result.step}/{max_steps} | "
                f"loss={result.loss_total:.4f} "
                f"(p={result.loss_policy:.4f} v={result.loss_value:.4f} r={result.loss_reward:.4f})"
            )

    # Save checkpoint
    output_dir = Path(cfg.project.output_dir) / cfg.project.name / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    from muzero.training.checkpoint import (
        CheckpointManager,
        CheckpointState,
        build_checkpoint_metadata,
    )

    env_spec = components.env.action_space_spec()
    action_dict: dict[str, object] = {"type": env_spec.type}
    if env_spec.n is not None:
        action_dict["n"] = env_spec.n

    meta = build_checkpoint_metadata(
        env_id=cfg.environment.id,
        network_type=cfg.network.type,
        observation_shape=tuple(components.env.observation_space_spec().shape),
        action_space=action_dict,
        num_players=cfg.algorithm.num_players,
        training_steps=trainer.step,
        algorithm=cfg.algorithm.name,
        project_name=cfg.project.name,
        config_data=cfg.model_dump(mode="json"),
    )

    ckpt_mgr = CheckpointManager()
    ckpt_mgr.save(
        CheckpointState(
            model_state_dict=network.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            config=cfg.model_dump(mode="json"),
            step=trainer.step,
            metadata=meta.model_dump(),
        ),
        output_dir / "final.pt",
    )
    print(f"Checkpoint saved to {output_dir / 'final.pt'}")

    env.close()


@app.command()
def benchmark(
    config: Path = typer.Option(..., "--config", help="Path to YAML config file"),
    profile: str = typer.Option("cpu_debug", "--profile", help="Hardware profile name"),
    component: str = typer.Option("inference", "--component", help="Component: inference|search"),
    steps: int = typer.Option(10, "--steps", help="Number of benchmark steps"),
) -> None:
    """Run a performance benchmark on a specific component."""
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


if __name__ == "__main__":
    app()
