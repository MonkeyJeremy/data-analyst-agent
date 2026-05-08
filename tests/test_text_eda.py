"""Tests for src/text/eda.py — text column detection and word frequency."""
from __future__ import annotations

import pandas as pd
import pytest

from src.text.eda import compute_top_words, detect_text_cols
from src.eda.auto_eda import run_auto_eda


# ── detect_text_cols ──────────────────────────────────────────────────────────

def test_detect_text_cols_long_strings():
    """Column with long avg character length → detected as text."""
    df = pd.DataFrame({
        "review": [
            "This product is absolutely fantastic and I would recommend it to everyone.",
            "Terrible quality, broke after two days, very disappointed with the purchase.",
            "Average item, nothing special but it does what it says on the tin.",
        ]
    })
    result = detect_text_cols(df)
    assert "review" in result


def test_detect_text_cols_short_strings():
    """Column with short avg character length → NOT detected as text."""
    df = pd.DataFrame({
        "status": ["ok", "fail", "ok", "ok", "fail", "pending"]
    })
    result = detect_text_cols(df)
    assert "status" not in result


def test_detect_text_cols_categorical():
    """Low-cardinality string column (like 'department') → NOT detected as text."""
    df = pd.DataFrame({
        "department": ["Engineering", "Marketing", "Engineering", "Sales",
                        "Marketing", "Engineering", "Sales", "Marketing",
                        "Engineering", "Engineering"],
    })
    result = detect_text_cols(df)
    assert "department" not in result


def test_detect_text_cols_mixed_df():
    """Only long, high-cardinality text column detected; short/categorical ignored."""
    reviews = [
        f"This is review number {i}, with some additional filler text to make it long enough."
        for i in range(20)
    ]
    df = pd.DataFrame({
        "review": reviews,
        "status": ["ok", "fail"] * 10,
        "score": range(20),
    })
    result = detect_text_cols(df)
    assert "review" in result
    assert "status" not in result
    assert "score" not in result  # numeric, not selected


def test_detect_text_cols_empty_df():
    """Empty DataFrame → empty tuple, no crash."""
    df = pd.DataFrame({"review": pd.Series([], dtype="object")})
    result = detect_text_cols(df)
    assert result == ()


# ── compute_top_words ─────────────────────────────────────────────────────────

def test_compute_top_words_basic():
    """Most frequent non-stop word appears first."""
    series = pd.Series([
        "great product great service great value",
        "great quality product",
        "product is great",
    ])
    words = dict(compute_top_words(series))
    assert "great" in words
    assert "product" in words
    # "great" appears 4 times, should be first
    top_word = compute_top_words(series)[0][0]
    assert top_word == "great"


def test_compute_top_words_excludes_stop_words():
    """Common stop words ('the', 'and', 'is', etc.) are excluded."""
    series = pd.Series([
        "the product is great and the service is also great",
        "this is the best product and it is amazing",
    ])
    words = dict(compute_top_words(series))
    for stop in ("the", "and", "is", "this", "also"):
        assert stop not in words, f"Stop word '{stop}' should be excluded"


def test_compute_top_words_empty_series():
    """Empty / all-null series → empty tuple, no crash."""
    result = compute_top_words(pd.Series([], dtype="object"))
    assert result == ()

    result_nulls = compute_top_words(pd.Series([None, None, None]))
    assert result_nulls == ()


def test_compute_top_words_respects_n():
    """Returns at most n words."""
    series = pd.Series(["alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"])
    result = compute_top_words(series, n=5)
    assert len(result) <= 5


# ── EDAReport integration ─────────────────────────────────────────────────────

def test_eda_report_text_fields_populated():
    """run_auto_eda() populates text_cols and top_words when text columns exist."""
    reviews = [
        f"This is detailed review number {i} with enough words to qualify as free text content"
        for i in range(30)
    ]
    df = pd.DataFrame({
        "review": reviews,
        "score": range(30),
    })
    eda = run_auto_eda(df)
    assert "review" in eda.text_cols
    # top_words should have an entry for the review column
    assert len(eda.top_words) >= 1
    col_names = [col for col, _ in eda.top_words]
    assert "review" in col_names


def test_eda_report_no_text_cols_defaults():
    """run_auto_eda() on a purely numeric/short-string df → text_cols=() top_words=()."""
    df = pd.DataFrame({
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        "category": ["A", "B", "A", "B", "A"],
    })
    eda = run_auto_eda(df)
    assert eda.text_cols == ()
    assert eda.top_words == ()
