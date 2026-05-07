"""EDAReport — immutable snapshot of automated EDA findings for one DataFrame."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EDAReport:
    """Frozen snapshot produced by run_auto_eda().

    All fields use plain Python types (int, float, str, tuples) so the
    dataclass is truly hashable and safe to store in Streamlit session state.
    No pandas objects are retained.
    """

    # Column type counts
    n_numeric: int
    n_categorical: int

    # Missing values — only columns where pct > 0, sorted desc by pct
    # Each entry: (column_name, pct_missing)  e.g. ("Age", 19.9)
    missing_pct: tuple[tuple[str, float], ...]

    # Top Pearson correlations between numeric column pairs (|r| > 0.3)
    # Each entry: (col_a, col_b, r)  sorted by abs(r) descending
    top_correlations: tuple[tuple[str, str, float], ...]

    # Numeric columns with |skewness| > 1
    # Each entry: (column_name, skewness)
    skewed_cols: tuple[tuple[str, float], ...]

    # IQR-based outlier counts per numeric column (only where n_outliers > 0)
    # Each entry: (column_name, n_outliers)
    outlier_counts: tuple[tuple[str, int], ...]

    # Categorical columns where unique_count / n_rows > 0.20
    high_cardinality_cols: tuple[str, ...]

    # Columns with only 1 unique non-null value
    constant_cols: tuple[str, ...]

    # 3–5 personalised suggested questions derived from findings
    suggested_questions: tuple[str, ...]

    # 3–4 sentence human-readable summary for injection into the system prompt.
    # Capped at 1 500 characters inside run_auto_eda().
    narrative: str
