"""Text-column detection and word-frequency computation.

No external NLP libraries required — pure Python + pandas.
"""
from __future__ import annotations

import re

import pandas as pd

# ── Thresholds ────────────────────────────────────────────────────────────────

_MIN_AVG_CHARS = 30     # average character length — below this it's a short label
_MIN_CARDINALITY = 0.3  # unique/total ratio — below this it's a categorical column

# Built-in stop-word set (common English function words that carry no analytic value)
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "in", "it", "of", "to", "and", "or",
    "for", "on", "at", "with", "this", "that", "was", "are",
    "be", "i", "my", "we", "they", "he", "she", "not", "but",
    "have", "has", "had", "do", "did", "so", "if", "as", "by",
    "from", "up", "out", "its", "about", "into", "than", "then",
    "there", "their", "our", "your", "all", "been", "more", "when",
    "will", "would", "could", "should", "also", "very", "just",
    "can", "get", "got", "one", "two", "three", "new", "no",
})


def detect_text_cols(df: pd.DataFrame) -> tuple[str, ...]:
    """Return column names that contain long free-form text.

    A column is considered a text column when:
    - its dtype is ``object`` (string),
    - its average character length is ≥ :data:`_MIN_AVG_CHARS`, AND
    - its cardinality ratio (``nunique / len``) is ≥ :data:`_MIN_CARDINALITY`
      (low-cardinality columns like "department" are categoricals, not text).

    Parameters
    ----------
    df:
        DataFrame to inspect.

    Returns
    -------
    tuple[str, ...]
        Column names that qualify as free-form text, in DataFrame column order.
    """
    text_cols: list[str] = []
    n = len(df)
    if n == 0:
        return ()

    for col in df.select_dtypes(include="object").columns:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue
        avg_chars = float(series.str.len().mean())
        cardinality = series.nunique() / n
        if avg_chars >= _MIN_AVG_CHARS and cardinality >= _MIN_CARDINALITY:
            text_cols.append(col)

    return tuple(text_cols)


def compute_top_words(
    series: pd.Series,
    n: int = 15,
) -> tuple[tuple[str, int], ...]:
    """Return the *n* most frequent non-stop words in *series*.

    Tokenises by extracting ``[a-z]+`` sequences (lowercased), filters out
    :data:`_STOP_WORDS` and words shorter than 3 characters, then sorts by
    descending count.

    Parameters
    ----------
    series:
        A pandas Series of strings (text column).
    n:
        Maximum number of words to return (default 15).

    Returns
    -------
    tuple[tuple[str, int], ...]
        Pairs of ``(word, count)`` sorted by count descending.
    """
    counts: dict[str, int] = {}
    for text in series.dropna().astype(str):
        for word in re.findall(r"[a-z]+", text.lower()):
            if word not in _STOP_WORDS and len(word) > 2:
                counts[word] = counts.get(word, 0) + 1

    sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return tuple(sorted_words[:n])
