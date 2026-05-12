"""E2B cloud sandbox execution backend (stub).

To activate: set E2B_API_KEY and install ``e2b-code-interpreter``.
"""
from __future__ import annotations

import pandas as pd

from src.execution.backend import ExecutionBackend
from src.execution.result import ExecutionResult


class E2BExecutor(ExecutionBackend):
    """Runs code in an E2B cloud sandbox for full process isolation."""

    @property
    def name(self) -> str:
        return "e2b"

    def execute(self, code: str, dataframes: dict[str, pd.DataFrame]) -> ExecutionResult:
        raise NotImplementedError(
            "E2B backend is not yet configured. "
            "Set the E2B_API_KEY environment variable and install "
            "e2b-code-interpreter (pip install e2b-code-interpreter) to enable it."
        )
