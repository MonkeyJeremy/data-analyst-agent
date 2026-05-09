import pandas as pd
import pytest

from src.execution.python_executor import _check_imports, execute_python
from src.execution.result import ExecutionResult


def test_stdout_captured(sample_df):
    result = execute_python("print('hello world')", sample_df)
    assert result.error is None
    assert "hello world" in result.stdout


def test_no_df_mutation(sample_df):
    original_values = sample_df["age"].tolist()
    execute_python("df['age'] = 0", sample_df)
    assert sample_df["age"].tolist() == original_values


def test_figure_captured(sample_df):
    code = "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot([1,2,3])"
    result = execute_python(code, sample_df)
    assert result.error is None
    assert len(result.figures) == 1
    assert result.figures[0][:4] == b"\x89PNG"  # PNG magic bytes


def test_multiple_figures(sample_df):
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig1, ax1 = plt.subplots()\nax1.plot([1])\n"
        "fig2, ax2 = plt.subplots()\nax2.plot([2])"
    )
    result = execute_python(code, sample_df)
    assert result.error is None
    assert len(result.figures) == 2


def test_error_captured(sample_df):
    result = execute_python("raise ValueError('boom')", sample_df)
    assert result.error is not None
    assert "ValueError" in result.error
    assert "boom" in result.error


def test_summary_reflects_stdout(sample_df):
    result = execute_python("print('answer: 42')", sample_df)
    assert "answer: 42" in result.summary


def test_summary_error_prefix(sample_df):
    result = execute_python("1/0", sample_df)
    assert result.summary.startswith("ERROR:")


def test_summary_no_output(sample_df):
    result = execute_python("x = 1 + 1", sample_df)
    assert "successfully" in result.summary.lower()


def test_df_accessible(sample_df):
    result = execute_python("print(len(df))", sample_df)
    assert str(len(sample_df)) in result.stdout


def test_pandas_pre_imported(sample_df):
    result = execute_python("print(pd.__version__)", sample_df)
    assert result.error is None
    assert len(result.stdout.strip()) > 0


def test_result_is_frozen():
    r = ExecutionResult(stdout="", error=None, figures=(), summary="ok")
    try:
        r.stdout = "changed"  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except Exception:
        pass


# ── AST import blocklist tests ────────────────────────────────────────────────

def test_check_imports_safe_code():
    """Pure pandas / numpy code passes the check."""
    assert _check_imports("import pandas as pd\ndf.head()") is None


def test_check_imports_blocks_os():
    """Importing 'os' is blocked."""
    err = _check_imports("import os\nos.listdir('.')")
    assert err is not None
    assert "SecurityError" in err
    assert "os" in err


def test_check_imports_blocks_subprocess():
    """Importing 'subprocess' is blocked."""
    err = _check_imports("import subprocess")
    assert err is not None
    assert "SecurityError" in err


def test_check_imports_blocks_from_import():
    """'from os import ...' is also blocked."""
    err = _check_imports("from os.path import join")
    assert err is not None
    assert "SecurityError" in err


def test_check_imports_blocks_socket():
    """Network access via 'socket' is blocked."""
    err = _check_imports("import socket\nsocket.connect(('evil.com', 80))")
    assert err is not None


def test_check_imports_syntax_error():
    """Syntactically invalid code returns a SyntaxError message."""
    err = _check_imports("def foo(: pass")
    assert err is not None
    assert "SyntaxError" in err


def test_execute_python_blocks_os_import(sample_df):
    """execute_python() returns a SecurityError result for blocked imports."""
    result = execute_python("import os\nprint(os.getcwd())", sample_df)
    assert result.error is not None
    assert "SecurityError" in result.error
    assert result.stdout == ""


def test_execute_python_blocks_subprocess(sample_df):
    """execute_python() blocks subprocess even in nested code."""
    result = execute_python("import subprocess\nsubprocess.run(['ls'])", sample_df)
    assert result.error is not None
    assert "SecurityError" in result.error
