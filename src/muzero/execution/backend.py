"""Execution backend protocol and local implementation."""

from __future__ import annotations

from typing import Any, Protocol


class TaskHandle(Protocol):
    """Handle to an asynchronous task."""

    def result(self) -> Any: ...


class ExecutorBackend(Protocol):
    """Protocol for execution backends (local or Ray)."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def run_training(self) -> None: ...
    def run_self_play(self) -> None: ...
    def submit_task(self, name: str, payload: dict[str, Any]) -> TaskHandle: ...


class LocalTaskHandle:
    """Synchronous task handle for local execution."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def result(self) -> Any:
        return self._result


class LocalExecutorBackend:
    """Local (single-process) execution backend.

    Args:
        config: MuZero configuration (unused in v1 stub).
    """

    def __init__(self, config: object) -> None:  # noqa: ARG002
        pass

    def start(self) -> None:
        """Start the backend (no-op for local)."""
        pass

    def stop(self) -> None:
        """Stop the backend (no-op for local)."""
        pass

    def run_training(self) -> None:
        """Run training (implemented in CLI)."""
        raise NotImplementedError("Training loop is in CLI")

    def run_self_play(self) -> None:
        """Run self-play (implemented in CLI)."""
        raise NotImplementedError("Self-play loop is in CLI")

    def submit_task(self, name: str, payload: dict[str, Any]) -> LocalTaskHandle:
        """Submit a task synchronously.

        Args:
            name: Task name.
            payload: Task payload.

        Returns:
            LocalTaskHandle with result.
        """
        return LocalTaskHandle({"name": name, "done": True})
