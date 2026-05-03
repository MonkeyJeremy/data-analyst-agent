import io
import pathlib

import pandas as pd
import pytest

from src.data.loader import load_tabular


TITANIC_PATH = pathlib.Path(__file__).parent / "fixtures" / "titanic.csv"


def test_load_csv_from_path():
    with open(TITANIC_PATH, "rb") as f:
        df = load_tabular(f, "titanic.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert "Survived" in df.columns


def test_load_csv_from_bytes():
    csv_bytes = b"a,b,c\n1,2,3\n4,5,6"
    df = load_tabular(io.BytesIO(csv_bytes), "data.csv")
    assert df.shape == (2, 3)
    assert list(df.columns) == ["a", "b", "c"]


def test_load_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_tabular(io.BytesIO(b"data"), "file.json")


def test_load_malformed_csv():
    # Mismatched quotes cause a ParserError which loader wraps as ValueError
    bad_csv = b'"col_a","col_b"\n"unclosed_quote,1\n2,3'
    with pytest.raises(ValueError):
        load_tabular(io.BytesIO(bad_csv), "bad.csv")


def test_load_csv_preserves_column_names():
    csv_bytes = b"first_name,last_name,age\nAlice,Smith,30"
    df = load_tabular(io.BytesIO(csv_bytes), "people.csv")
    assert "first_name" in df.columns
    assert "last_name" in df.columns
    assert "age" in df.columns
