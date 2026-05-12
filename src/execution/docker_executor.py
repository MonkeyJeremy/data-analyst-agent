"""Docker container execution backend (stub).

To activate: install Docker and the ``docker`` Python SDK.
"""
from __future__ import annotations

import pandas as pd

from src.execution.backend import ExecutionBackend
from src.execution.result import ExecutionResult


class DockerExecutor(ExecutionBackend):
    """Runs code inside a Docker container for full process isolation."""

    @property
    def name(self) -> str:
        return "docker"

    def execute(self, code: str, dataframes: dict[str, pd.DataFrame]) -> ExecutionResult:
        raise NotImplementedError(
            "Docker backend is not yet configured. "
            "Install Docker and the docker Python SDK (pip install docker) to enable it."
        )
