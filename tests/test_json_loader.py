"""Tests for JSON support in src/data/loader.py."""
from __future__ import annotations

import io
import json
import pathlib

import pandas as pd
import pytest

from src.data.loader import load_tabular


FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


# ── Flat JSON array ───────────────────────────────────────────────────────────

def test_flat_json_array():
    """Top-level array of dicts → flat DataFrame."""
    data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    buf = io.BytesIO(json.dumps(data).encode())
    df = load_tabular(buf, "data.json")
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]
    assert df["a"].tolist() == [1, 2]


# ── Nested JSON with "data" wrapper key ──────────────────────────────────────

def test_nested_json_data_key():
    """Dict with a 'data' key containing an array → normalized."""
    data = {"data": [{"id": 1, "val": 10}, {"id": 2, "val": 20}], "meta": "ignored"}
    buf = io.BytesIO(json.dumps(data).encode())
    df = load_tabular(buf, "response.json")
    assert len(df) == 2
    assert "id" in df.columns
    assert "val" in df.columns


def test_nested_json_records_key():
    """Dict with a 'records' key containing an array → normalized."""
    data = {"records": [{"x": 1}, {"x": 2}, {"x": 3}]}
    buf = io.BytesIO(json.dumps(data).encode())
    df = load_tabular(buf, "export.json")
    assert len(df) == 3
    assert "x" in df.columns


def test_nested_json_results_key():
    """Dict with a 'results' key → normalized."""
    data = {"results": [{"score": 88}, {"score": 95}]}
    buf = io.BytesIO(json.dumps(data).encode())
    df = load_tabular(buf, "api_response.json")
    assert len(df) == 2
    assert "score" in df.columns


# ── Deeply nested — json_normalize flattens dotted columns ──────────────────

def test_deeply_nested_json_normalize():
    """Nested object fields produce dotted column names via json_normalize."""
    data = [
        {"id": 1, "metrics": {"revenue": 1000, "cost": 400}},
        {"id": 2, "metrics": {"revenue": 1500, "cost": 600}},
    ]
    buf = io.BytesIO(json.dumps(data).encode())
    df = load_tabular(buf, "nested.json")
    assert "id" in df.columns
    # json_normalize flattens nested dicts to "metrics.revenue" etc.
    assert any("revenue" in c for c in df.columns)


# ── Single object → one-row DataFrame ────────────────────────────────────────

def test_json_single_object():
    """Top-level dict with no known wrapper key → one-row DataFrame."""
    data = {"name": "Alice", "age": 30, "city": "NYC"}
    buf = io.BytesIO(json.dumps(data).encode())
    df = load_tabular(buf, "record.json")
    assert len(df) == 1
    assert "name" in df.columns
    assert df["name"].iloc[0] == "Alice"


# ── Empty array ───────────────────────────────────────────────────────────────

def test_json_empty_array():
    """Empty array → empty DataFrame, no crash."""
    buf = io.BytesIO(b"[]")
    df = load_tabular(buf, "empty.json")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ── Unsupported top-level type ────────────────────────────────────────────────

def test_json_unsupported_scalar_raises():
    """Top-level scalar (string, number) → ValueError."""
    buf = io.BytesIO(b'"just a string"')
    with pytest.raises(ValueError, match="Cannot parse JSON"):
        load_tabular(buf, "bad.json")


def test_json_invalid_syntax_raises():
    """Malformed JSON → ValueError."""
    buf = io.BytesIO(b"{not valid json}")
    with pytest.raises(ValueError):
        load_tabular(buf, "malformed.json")


# ── Via load_tabular dispatch ─────────────────────────────────────────────────

def test_json_via_load_tabular_fixture():
    """Load the sample.json fixture via load_tabular."""
    path = FIXTURES_DIR / "sample.json"
    with open(path, "rb") as f:
        df = load_tabular(f, "sample.json")
    assert len(df) == 5
    assert "name" in df.columns
    assert "score" in df.columns


def test_nested_json_fixture():
    """Load the nested.json fixture; should expand the 'data' wrapper."""
    path = FIXTURES_DIR / "nested.json"
    with open(path, "rb") as f:
        df = load_tabular(f, "nested.json")
    assert len(df) == 3
    assert "id" in df.columns
    # metrics sub-fields should be present (dotted or flat)
    assert any("revenue" in c for c in df.columns)
