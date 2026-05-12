"""Tests for src/execution/backend_factory.py and the ExecutionBackend ABC."""
from __future__ import annotations

import pytest

from src.execution.backend import ExecutionBackend
from src.execution.backend_factory import get_backend
from src.execution.python_executor import LocalPythonExecutor


def test_local_backend_returns_local_executor():
    backend = get_backend("local")
    assert isinstance(backend, LocalPythonExecutor)
    assert backend.name == "local"


def test_e2b_backend_raises_on_execute():
    backend = get_backend("e2b")
    assert backend.name == "e2b"
    with pytest.raises(NotImplementedError, match="E2B"):
        import pandas as pd
        backend.execute("print(1)", {"df": pd.DataFrame()})


def test_docker_backend_raises_on_execute():
    backend = get_backend("docker")
    assert backend.name == "docker"
    with pytest.raises(NotImplementedError, match="Docker"):
        import pandas as pd
        backend.execute("print(1)", {"df": pd.DataFrame()})


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError, match="Unknown execution mode"):
        get_backend("unknown_mode")


def test_local_executor_is_execution_backend():
    assert isinstance(get_backend("local"), ExecutionBackend)


def test_local_executor_runs_code():
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = get_backend("local").execute("print(df.shape)", {"df": df})
    assert result.error is None
    assert "(3, 1)" in result.stdout
