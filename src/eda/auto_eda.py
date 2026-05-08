"""Automated EDA computation — pure function, no side effects."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eda.report import EDAReport
from src.text.eda import compute_top_words, detect_text_cols

_NARRATIVE_CAP = 1_500
_CORR_THRESHOLD = 0.30
_SKEW_THRESHOLD = 1.0
_HIGH_CARD_RATIO = 0.20
_TOP_CORR_N = 10


def run_auto_eda(df: pd.DataFrame) -> EDAReport:
    """Compute an EDAReport from *df*.

    Parameters
    ----------
    df:
        DataFrame to analyse.  Must have at least 1 row.

    Returns
    -------
    EDAReport
        Immutable snapshot of key EDA findings.

    Raises
    ------
    ValueError
        If *df* has 0 rows.
    """
    if len(df) == 0:
        raise ValueError("Cannot run EDA on an empty DataFrame (0 rows).")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    missing_pct = _compute_missing(df)
    top_correlations = _compute_correlations(df, num_cols)
    skewed_cols = _compute_skewness(df, num_cols)
    outlier_counts = _compute_outliers(df, num_cols)
    high_cardinality_cols = _compute_high_cardinality(df, cat_cols)
    constant_cols = _compute_constant(df)

    text_cols = detect_text_cols(df)
    top_words: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] = tuple(
        (col, compute_top_words(df[col])) for col in text_cols
    )

    suggested_questions = _build_questions(
        num_cols, cat_cols, missing_pct, top_correlations, skewed_cols, outlier_counts,
        text_cols=text_cols,
    )
    narrative = _build_narrative(
        df, num_cols, cat_cols, missing_pct, top_correlations, skewed_cols, outlier_counts
    )

    return EDAReport(
        n_numeric=len(num_cols),
        n_categorical=len(cat_cols),
        missing_pct=missing_pct,
        top_correlations=top_correlations,
        skewed_cols=skewed_cols,
        outlier_counts=outlier_counts,
        high_cardinality_cols=tuple(high_cardinality_cols),
        constant_cols=tuple(constant_cols),
        suggested_questions=tuple(suggested_questions),
        narrative=narrative,
        text_cols=text_cols,
        top_words=top_words,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_missing(df: pd.DataFrame) -> tuple[tuple[str, float], ...]:
    pct = (df.isnull().mean() * 100).round(1)
    result = [(col, float(pct[col])) for col in pct.index if pct[col] > 0]
    result.sort(key=lambda x: x[1], reverse=True)
    return tuple((c, p) for c, p in result)


def _compute_correlations(
    df: pd.DataFrame, num_cols: list[str]
) -> tuple[tuple[str, str, float], ...]:
    if len(num_cols) < 2:
        return ()

    # Drop columns that are entirely NaN before computing correlation
    valid = df[num_cols].dropna(axis=1, how="all")
    if valid.shape[1] < 2:
        return ()

    corr = valid.corr()
    pairs: list[tuple[str, str, float]] = []

    cols = corr.columns.tolist()
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            r = corr.loc[col_a, col_b]
            if np.isnan(r):
                continue
            if abs(r) >= _CORR_THRESHOLD:
                pairs.append((col_a, col_b, round(float(r), 3)))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return tuple(pairs[:_TOP_CORR_N])


def _compute_skewness(
    df: pd.DataFrame, num_cols: list[str]
) -> tuple[tuple[str, float], ...]:
    result: list[tuple[str, float]] = []
    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 3:  # need at least 3 points for meaningful skewness
            continue
        skew = float(series.skew())
        if abs(skew) >= _SKEW_THRESHOLD:
            result.append((col, round(skew, 3)))
    result.sort(key=lambda x: abs(x[1]), reverse=True)
    return tuple(result)


def _compute_outliers(
    df: pd.DataFrame, num_cols: list[str]
) -> tuple[tuple[str, int], ...]:
    result: list[tuple[str, int]] = []
    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            # IQR is zero when most values are identical — fall back to z-score
            std = float(series.std())
            if std == 0:
                continue  # truly constant numeric column
            mean = float(series.mean())
            n_out = int((np.abs(series - mean) > 3 * std).sum())
        else:
            n_out = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        if n_out > 0:
            result.append((col, n_out))
    result.sort(key=lambda x: x[1], reverse=True)
    return tuple(result)


def _compute_high_cardinality(
    df: pd.DataFrame, cat_cols: list[str]
) -> list[str]:
    n = len(df)
    return [c for c in cat_cols if df[c].nunique() / n > _HIGH_CARD_RATIO]


def _compute_constant(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].nunique(dropna=True) <= 1]


# ── Narrative ─────────────────────────────────────────────────────────────────

def _build_narrative(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    missing_pct: tuple[tuple[str, float], ...],
    top_correlations: tuple[tuple[str, str, float], ...],
    skewed_cols: tuple[tuple[str, float], ...],
    outlier_counts: tuple[tuple[str, int], ...],
) -> str:
    parts: list[str] = []

    # Sentence 1 — shape
    parts.append(
        f"The dataset has {len(df):,} rows and {len(df.columns)} columns "
        f"({len(num_cols)} numeric, {len(cat_cols)} categorical)."
    )

    # Sentence 2 — missing values (top 3)
    if missing_pct:
        top3 = missing_pct[:3]
        miss_desc = ", ".join(f"'{c}' ({p:.1f}%)" for c, p in top3)
        parts.append(f"Missing values detected in: {miss_desc}.")

    # Sentence 3 — top correlation
    if top_correlations:
        a, b, r = top_correlations[0]
        direction = "positive" if r > 0 else "negative"
        strength = "strong" if abs(r) >= 0.7 else "moderate"
        parts.append(
            f"{strength.capitalize()} {direction} correlation between "
            f"'{a}' and '{b}' (r={r:.2f})."
        )

    # Sentence 4 — skewness or outliers
    if skewed_cols:
        cols_str = ", ".join(f"'{c}'" for c, _ in skewed_cols[:3])
        parts.append(f"Skewed numeric columns: {cols_str}.")
    elif outlier_counts:
        col, n = outlier_counts[0]
        parts.append(f"'{col}' has {n} IQR outliers.")

    narrative = " ".join(parts)
    return narrative[:_NARRATIVE_CAP]


# ── Suggested questions ───────────────────────────────────────────────────────

_GENERIC_FALLBACKS = (
    "Give me a summary of this dataset.",
    "Show the distribution of each numeric column.",
    "Are there any missing values? Which columns are affected?",
    "What are the top correlations between numeric columns?",
    "Plot the most interesting relationship you can find.",
)


def _build_questions(
    num_cols: list[str],
    cat_cols: list[str],
    missing_pct: tuple[tuple[str, float], ...],
    top_correlations: tuple[tuple[str, str, float], ...],
    skewed_cols: tuple[tuple[str, float], ...],
    outlier_counts: tuple[tuple[str, int], ...],
    text_cols: tuple[str, ...] = (),
) -> list[str]:
    questions: list[str] = []

    # 0. Text-specific questions (prepended when text columns exist)
    if text_cols:
        questions.append(
            f"What is the overall sentiment in the '{text_cols[0]}' column?"
        )
        questions.append(
            f"What are the most common topics or themes in '{text_cols[0]}'?"
        )

    # 1. Top correlation pair
    if top_correlations:
        a, b, _ = top_correlations[0]
        questions.append(f"What is the relationship between '{a}' and '{b}'?")

    # 2. Missing values
    if missing_pct:
        top_miss = missing_pct[0][0]
        questions.append(f"How should we handle missing values in '{top_miss}'?")

    # 3. Most skewed column
    if skewed_cols:
        col = skewed_cols[0][0]
        questions.append(f"Show the distribution of '{col}'.")
    elif num_cols:
        questions.append(f"Show the distribution of '{num_cols[0]}'.")

    # 4. Outliers
    if outlier_counts:
        col = outlier_counts[0][0]
        questions.append(f"Are there outliers in '{col}'? Show me a box plot.")

    # 5. Always include a summary question
    questions.append("Give me a summary of this dataset.")

    # Pad with generics if somehow we still have < 3
    idx = 0
    while len(questions) < 3 and idx < len(_GENERIC_FALLBACKS):
        if _GENERIC_FALLBACKS[idx] not in questions:
            questions.append(_GENERIC_FALLBACKS[idx])
        idx += 1

    return questions[:5]
