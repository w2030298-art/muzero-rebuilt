"""Checkpoint management CLI."""

# ruff: noqa: B008

from __future__ import annotations

from pathlib import Path

import typer

from muzero.training.checkpoint import CheckpointManager

app = typer.Typer(name="checkpoint")


@app.command()
def inspect(path: Path = typer.Option(..., "--path", help="Path to checkpoint file")):
    """Inspect checkpoint metadata."""
    mgr = CheckpointManager()
    try:
        meta = mgr.inspect(path)
    except FileNotFoundError:
        print(f"Error: checkpoint not found: {path}")
        raise typer.Exit(1) from None

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Checkpoint: {path.name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Algorithm", meta.algorithm)
    table.add_row("Environment", meta.env_id)
    table.add_row("Network", meta.network_type)
    table.add_row("Players", str(meta.num_players))
    table.add_row("Training Steps", str(meta.training_steps))
    table.add_row("Config Hash", meta.config_hash[:16] + "..." if meta.config_hash else "N/A")
    table.add_row("Created", meta.created_at or "N/A")
    console.print(table)


@app.command()
def export(
    path: Path = typer.Option(..., "--path", help="Path to checkpoint file"),
    out: Path = typer.Option(..., "--out", help="Output directory"),
):
    """Export checkpoint to a structured directory."""
    mgr = CheckpointManager()
    try:
        mgr.export(path, out)
    except FileNotFoundError:
        print(f"Error: checkpoint not found: {path}")
        raise typer.Exit(1) from None
    print(f"Exported to {out}/")
    print(f"  {out}/model.pt")
    print(f"  {out}/config.yaml")
    print(f"  {out}/metadata.yaml")
    print(f"  {out}/README.md")
