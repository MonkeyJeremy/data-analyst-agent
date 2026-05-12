import pandas as pd

from src.agent.tools import TOOL_SCHEMAS, dispatch_tool


def test_tool_schemas_non_empty():
    assert len(TOOL_SCHEMAS) >= 1


def test_execute_python_schema_has_required_keys():
    schema = next(s for s in TOOL_SCHEMAS if s["name"] == "execute_python")
    assert "description" in schema
    assert "input_schema" in schema
    required = schema["input_schema"]["required"]
    assert "code" in required
    assert "purpose" in required


def test_dispatch_execute_python(sample_df):
    result = dispatch_tool(
        "execute_python",
        {"code": "print(len(df))", "purpose": "Count rows"},
        df=sample_df,
    )
    assert result.error is None
    assert str(len(sample_df)) in result.stdout


def test_dispatch_execute_python_with_dataframes(sample_df):
    """dispatch_tool accepts the new dataframes dict kwarg."""
    ns = {"df": sample_df}
    result = dispatch_tool(
        "execute_python",
        {"code": "print(len(df))", "purpose": "Count rows"},
        dataframes=ns,
    )
    assert result.error is None
    assert str(len(sample_df)) in result.stdout


def test_dispatch_unknown_tool_returns_error(sample_df):
    result = dispatch_tool("nonexistent_tool", {}, df=sample_df)
    assert result.error is not None
    assert "nonexistent_tool" in result.error
    assert "ERROR" in result.summary


def test_dispatch_code_error_propagates(sample_df):
    result = dispatch_tool(
        "execute_python",
        {"code": "raise RuntimeError('test')", "purpose": "test"},
        df=sample_df,
    )
    assert result.error is not None
    assert "RuntimeError" in result.error
