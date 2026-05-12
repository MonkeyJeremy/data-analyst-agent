"""Abstract execution backend interface.

Concrete backends:
- :class:`LocalPythonExecutor`  — in-process exec() with AST sandboxing (default)
- :class:`E2BExecutor`          — e2b-code-interpreter sandbox (requires API key)
- :class:`DockerExecutor`       — Docker container isolation (requires Docker)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.execution.result import ExecutionResult


class ExecutionBackend(ABC):
    """Execute Python code against one or more DataFrames and return structured results."""

    @abstractmethod
    def execute(self, code: str, dataframes: dict[str, pd.DataFrame]) -> ExecutionResult:
        """Run *code* with all *dataframes* available as named variables.

        The dict maps variable name → DataFrame.  A ``"df"`` key is always
        included as an alias for the primary (first-loaded) table so that
        single-df agent code continues to work without modification.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier shown in the UI."""
