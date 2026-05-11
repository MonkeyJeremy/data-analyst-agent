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
    """Execute Python code against a DataFrame and return structured results."""

    @abstractmethod
    def execute(self, code: str, df: pd.DataFrame) -> ExecutionResult:
        """Run *code* with *df* available as the ``df`` variable."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier shown in the UI."""
