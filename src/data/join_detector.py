"""Detect likely join keys between DataFrames in a registry."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.registry import DataFrameEntry, DataFrameRegistry

_OVERLAP_THRESHOLD = 0.80
_PK_UNIQUENESS_THRESHOLD = 0.90
_MAX_SAMPLE = 10_000
_MAX_SUGGESTIONS = 10


@dataclass(frozen=True)
class JoinSuggestion:
    left_table: str
    left_col: str
    right_table: str
    right_col: str
    match_rate: float
    join_type: str  # "inner" | "left"
    source: str = "auto"  # "auto" | "manual"

    def example_code(self) -> str:
        on_clause = (
            f"on='{self.left_col}'"
            if self.left_col == self.right_col
            else f"left_on='{self.left_col}', right_on='{self.right_col}'"
        )
        return (
            f"{self.left_table}.merge({self.right_table}, "
            f"{on_clause}, how='{self.join_type}')"
        )


def _value_overlap(left: pd.Series, right: pd.Series) -> float:
    left_vals = left.dropna().unique()
    if len(left_vals) == 0:
        return 0.0
    if len(left_vals) > _MAX_SAMPLE:
        rng = np.random.default_rng(42)
        left_vals = rng.choice(left_vals, _MAX_SAMPLE, replace=False)
    right_set = set(right.dropna().unique())
    hits = sum(1 for v in left_vals if v in right_set)
    return hits / len(left_vals)


def _is_pk_candidate(series: pd.Series) -> bool:
    n = len(series.dropna())
    if n == 0:
        return False
    return series.nunique() / n > _PK_UNIQUENESS_THRESHOLD


def _compatible_dtypes(a: pd.Series, b: pd.Series) -> bool:
    def kind(s: pd.Series) -> str:
        if pd.api.types.is_integer_dtype(s):
            return "int"
        if pd.api.types.is_float_dtype(s):
            return "float"
        if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
            return "str"
        return str(s.dtype)
    return kind(a) == kind(b)


def detect_join_keys(registry: DataFrameRegistry) -> list[JoinSuggestion]:
    entries = registry.entries()
    if len(entries) < 2:
        return []
    suggestions: list[JoinSuggestion] = []
    seen: set[tuple] = set()
    for i, left in enumerate(entries):
        for right in entries[i + 1:]:
            _check_pair(left, right, suggestions, seen)
    suggestions.sort(key=lambda s: -s.match_rate)
    return suggestions[:_MAX_SUGGESTIONS]


def _check_pair(
    left: DataFrameEntry,
    right: DataFrameEntry,
    suggestions: list[JoinSuggestion],
    seen: set[tuple],
) -> None:
    left_df, right_df = left.df, right.df
    shared = set(left_df.columns) & set(right_df.columns)

    for col in shared:
        if not _compatible_dtypes(left_df[col], right_df[col]):
            continue
        rate = _value_overlap(left_df[col], right_df[col])
        if rate >= _OVERLAP_THRESHOLD:
            key = (left.name, col, right.name, col)
            if key not in seen:
                seen.add(key)
                suggestions.append(JoinSuggestion(
                    left_table=left.name, left_col=col,
                    right_table=right.name, right_col=col,
                    match_rate=rate,
                    join_type="inner" if rate > 0.95 else "left",
                ))

    for r_col in right_df.columns:
        if not _is_pk_candidate(right_df[r_col]):
            continue
        for l_col in left_df.columns:
            if l_col in shared:
                continue
            if not _compatible_dtypes(left_df[l_col], right_df[r_col]):
                continue
            rate = _value_overlap(left_df[l_col], right_df[r_col])
            if rate >= _OVERLAP_THRESHOLD:
                key = (left.name, l_col, right.name, r_col)
                if key not in seen:
                    seen.add(key)
                    suggestions.append(JoinSuggestion(
                        left_table=left.name, left_col=l_col,
                        right_table=right.name, right_col=r_col,
                        match_rate=rate,
                        join_type="inner" if rate > 0.95 else "left",
                    ))
