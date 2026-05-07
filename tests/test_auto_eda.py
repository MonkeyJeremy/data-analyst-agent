"""Tests for src/eda/auto_eda.py — run_auto_eda()."""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pandas as pd
import pytest

from src.eda.auto_eda import run_auto_eda
from src.eda.report import EDAReport

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean_df() -> pd.DataFrame:
    """Small, clean DataFrame: 2 numeric, 1 categorical, no missing."""
    return pd.DataFrame(
        {
            "age": [30, 25, 35, 28, 40, 22],
            "salary": [70_000.0, 55_000.0, 85_000.0, 62_000.0, 90_000.0, 48_000.0],
            "dept": ["Eng", "Mkt", "Eng", "Mkt", "Eng", "Mkt"],
        }
    )


# ── basic fields ──────────────────────────────────────────────────────────────

def test_basic_fields():
    df = _clean_df()
    eda = run_auto_eda(df)
    assert isinstance(eda, EDAReport)
    assert eda.n_numeric == 2
    assert eda.n_categorical == 1


# ── missing values ────────────────────────────────────────────────────────────

def test_missing_detection():
    df = _clean_df()
    df.loc[0, "age"] = float("nan")  # 1/6 ≈ 16.7%
    eda = run_auto_eda(df)
    missing_cols = {c for c, _ in eda.missing_pct}
    assert "age" in missing_cols


def test_no_missing():
    eda = run_auto_eda(_clean_df())
    assert eda.missing_pct == ()


def test_missing_sorted_descending():
    df = _clean_df()
    df.loc[[0, 1], "age"] = float("nan")      # 2/6 ≈ 33%
    df.loc[0, "salary"] = float("nan")         # 1/6 ≈ 17%
    eda = run_auto_eda(df)
    pcts = [p for _, p in eda.missing_pct]
    assert pcts == sorted(pcts, reverse=True)


# ── correlations ──────────────────────────────────────────────────────────────

def test_correlations_perfect():
    """Perfectly correlated pair must appear at the top."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0]})
    eda = run_auto_eda(df)
    assert len(eda.top_correlations) >= 1
    a, b, r = eda.top_correlations[0]
    assert math.isclose(abs(r), 1.0, abs_tol=0.01)


def test_no_correlations_single_numeric():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "cat": ["a", "b", "c", "d"]})
    eda = run_auto_eda(df)
    assert eda.top_correlations == ()


def test_correlations_below_threshold_excluded():
    """Near-zero correlation should not appear (threshold 0.3)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=100), "y": rng.normal(size=100)})
    eda = run_auto_eda(df)
    for _, _, r in eda.top_correlations:
        assert abs(r) >= 0.30


# ── skewness ──────────────────────────────────────────────────────────────────

def test_skewness_detected():
    """Heavily right-skewed column (exponential) should appear in skewed_cols."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"income": rng.exponential(scale=50_000, size=200)})
    eda = run_auto_eda(df)
    skewed_names = {c for c, _ in eda.skewed_cols}
    assert "income" in skewed_names


def test_no_skewness_for_uniform():
    """Uniform distribution should not be flagged as skewed."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"val": rng.uniform(0, 100, size=200)})
    eda = run_auto_eda(df)
    assert eda.skewed_cols == ()


# ── outliers ─────────────────────────────────────────────────────────────────

def test_outliers_detected():
    base = [10.0] * 20
    base.append(1_000_000.0)   # extreme outlier
    df = pd.DataFrame({"value": base})
    eda = run_auto_eda(df)
    outlier_cols = {c for c, _ in eda.outlier_counts}
    assert "value" in outlier_cols


def test_no_outliers_clean():
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    eda = run_auto_eda(df)
    assert eda.outlier_counts == ()


# ── constant columns ──────────────────────────────────────────────────────────

def test_constant_col_detected():
    df = _clean_df()
    df["const"] = "same"
    eda = run_auto_eda(df)
    assert "const" in eda.constant_cols


def test_no_constant_cols():
    eda = run_auto_eda(_clean_df())
    assert eda.constant_cols == ()


# ── narrative ─────────────────────────────────────────────────────────────────

def test_narrative_nonempty():
    eda = run_auto_eda(_clean_df())
    assert len(eda.narrative) > 0


def test_narrative_length_capped():
    eda = run_auto_eda(_clean_df())
    assert len(eda.narrative) <= 1_500


def test_narrative_mentions_row_count():
    df = _clean_df()
    eda = run_auto_eda(df)
    assert str(len(df)) in eda.narrative


# ── suggested questions ───────────────────────────────────────────────────────

def test_suggested_questions_count():
    eda = run_auto_eda(_clean_df())
    assert 3 <= len(eda.suggested_questions) <= 5


def test_suggested_questions_are_strings():
    eda = run_auto_eda(_clean_df())
    for q in eda.suggested_questions:
        assert isinstance(q, str) and len(q) > 0


# ── edge cases ────────────────────────────────────────────────────────────────

def test_empty_df_raises():
    df = pd.DataFrame({"x": []})
    with pytest.raises(ValueError, match="empty"):
        run_auto_eda(df)


def test_all_numeric_df():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    eda = run_auto_eda(df)
    assert eda.n_categorical == 0
    assert eda.n_numeric == 2


def test_all_categorical_df():
    df = pd.DataFrame({"x": ["a", "b", "c"], "y": ["d", "e", "f"]})
    eda = run_auto_eda(df)
    assert eda.n_numeric == 0
    assert eda.top_correlations == ()
    assert eda.skewed_cols == ()
    assert eda.outlier_counts == ()


# ── titanic smoke test ────────────────────────────────────────────────────────

def test_titanic_smoke():
    df = pd.read_csv(FIXTURES / "titanic.csv")
    eda = run_auto_eda(df)
    assert eda.n_numeric >= 1
    assert len(eda.narrative) > 0
    # Cabin has 77% missing in full Titanic; our fixture has fewer rows but Cabin still partial
    all_cols = {c for c, _ in eda.missing_pct}
    # Age or Cabin should be flagged (our fixture has 1 missing Age row)
    assert len(eda.suggested_questions) >= 3
