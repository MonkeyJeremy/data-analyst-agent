import pandas as pd

from src.data.schema import SchemaContext, describe_schema


def test_schema_fields_populated(sample_df):
    ctx = describe_schema(sample_df)
    assert isinstance(ctx, SchemaContext)
    assert ctx.n_rows == 4
    assert ctx.n_cols == 4


def test_schema_formatted_dtypes_contains_columns(sample_df):
    ctx = describe_schema(sample_df)
    for col in sample_df.columns:
        assert col in ctx.formatted_dtypes


def test_schema_head_markdown_non_empty(sample_df):
    ctx = describe_schema(sample_df)
    assert len(ctx.head_markdown) > 0


def test_schema_describe_markdown_non_empty(sample_df):
    ctx = describe_schema(sample_df)
    assert len(ctx.describe_markdown) > 0


def test_schema_single_column_df():
    df = pd.DataFrame({"x": [1, 2, 3]})
    ctx = describe_schema(df)
    assert ctx.n_rows == 3
    assert ctx.n_cols == 1
    assert "x" in ctx.formatted_dtypes


def test_schema_empty_df():
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    ctx = describe_schema(df)
    assert ctx.n_rows == 0
    assert ctx.n_cols == 1
