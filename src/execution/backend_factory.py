"""Factory for selecting an :class:`~src.execution.backend.ExecutionBackend`."""
from __future__ import annotations

from src.execution.backend import ExecutionBackend


def get_backend(mode: str) -> ExecutionBackend:
    """Return an :class:`ExecutionBackend` for the given *mode* string.

    Parameters
    ----------
    mode:
        ``"local"`` (default), ``"e2b"``, or ``"docker"``.

    Raises
    ------
    ValueError
        If *mode* is not recognised.
    """
    if mode == "local":
        from src.execution.python_executor import LocalPythonExecutor
        return LocalPythonExecutor()
    if mode == "e2b":
        from src.execution.e2b_executor import E2BExecutor
        return E2BExecutor()
    if mode == "docker":
        from src.execution.docker_executor import DockerExecutor
        return DockerExecutor()
    raise ValueError(
        f"Unknown execution mode: '{mode}'. Choose from 'local', 'e2b', or 'docker'."
    )
