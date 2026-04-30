"""Ray execution backend — optional distributed execution.  # noqa: A005

Ray is imported lazily; this module is the ONLY file allowed to import ray.
"""

from __future__ import annotations

from typing import Any


class RayExecutorBackend:
    """Ray-based distributed execution backend.

    Uses Ray actors for self-play, training, and evaluation.
    This is a stub — full implementation reserved for future.

    Args:
        config: MuZero configuration.

    Raises:
        RuntimeError: If Ray is not installed.
    """

    def __init__(self, config: object) -> None:  # noqa: ARG002
        self._check_ray_installed()

    @staticmethod
    def _check_ray_installed() -> None:
        """Verify Ray is installed."""
        try:
            import ray  # type: ignore[import-not-found,unused-ignore]  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Ray backend requires installing the 'ray' extra: uv sync --extra ray"
            ) from None

    def start(self) -> None:
        """Initialize Ray."""
        import ray  # type: ignore[import-not-found]

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def stop(self) -> None:
        """Shutdown Ray."""
        import ray  # type: ignore[import-not-found]

        ray.shutdown()

    def submit_task(self, name: str, payload: dict[str, Any]) -> object:
        """Submit a task (stub).

        Args:
            name: Task name.
            payload: Task payload.

        Returns:
            Placeholder result.
        """
        return {"name": name, "status": "submitted"}
