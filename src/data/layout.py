"""Non-standard layout detection for uploaded tabular files.

Detects when a CSV/Excel file has multi-row headers, blank leading rows,
or other formatting that causes pandas to produce all-unnamed columns.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd

_UNNAMED_OK_THRESHOLD = 0.30    # below this → layout is fine
_AUTO_FIX_NAMED_THRESHOLD = 0.70  # above this → auto-fix with confidence
_PARTIAL_NAMED_THRESHOLD = 0.50   # above this → needs_confirmation (partial fix)
_HEADER_SCORE_THRESHOLD = 0.40   # minimum score to be a candidate header row
_AUTO_FIX_CONFIDENCE_THRESHOLD = 0.65  # minimum score to auto-fix silently
_MAX_SCAN_ROWS = 10              # how many leading rows to inspect


@dataclass(frozen=True)
class LayoutResult:
    """Result of layout detection for one uploaded file.

    status:
        "ok"                — default load is fine, no action needed
        "auto_fixed"        — non-standard layout detected and corrected automatically
        "needs_confirmation"— ambiguous; user must pick the header row
    header_row:
        Row index (0-based in the raw DataFrame) used as the column header.
    unnamed_ratio:
        Fraction of ``Unnamed:`` columns produced by the *default* load.
    confidence:
        Detection confidence 0.0–1.0. Meaningful only for auto_fixed.
    candidate_rows:
        Plausible header row indices offered to the user (needs_confirmation only).
    message:
        Human-readable explanation for UI display.
    """
    status: str
    header_row: int
    unnamed_ratio: float
    confidence: float
    candidate_rows: tuple[int, ...]
    message: str


def detect_layout(file_bytes: bytes, filename: str) -> LayoutResult:
    """Detect whether *file_bytes* has a non-standard header layout.

    Parameters
    ----------
    file_bytes:
        Raw file content (bytes).
    filename:
        Original filename — used to choose the correct pandas reader.

    Returns
    -------
    LayoutResult
        Detection result with status, detected header row, and message.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # ── Step 1: default load ──────────────────────────────────────────────────
    try:
        df_default = _read(file_bytes, ext, header=0)
    except Exception:
        return LayoutResult(
            status="needs_confirmation",
            header_row=0,
            unnamed_ratio=1.0,
            confidence=0.0,
            candidate_rows=tuple(range(min(5, _MAX_SCAN_ROWS))),
            message="Could not parse the file. Please select the header row manually.",
        )

    unnamed_ratio = _unnamed_ratio(df_default)
    # Also flag when column names are mostly raw integers/floats (pandas uses
    # row values as-is when the header row contains numbers rather than strings)
    effective_ratio = max(unnamed_ratio, _numeric_colname_ratio(df_default))

    if effective_ratio < _UNNAMED_OK_THRESHOLD:
        return LayoutResult(
            status="ok",
            header_row=0,
            unnamed_ratio=unnamed_ratio,
            confidence=1.0,
            candidate_rows=(),
            message="",
        )

    # ── Step 2: suspicious layout — scan leading rows ─────────────────────────
    unnamed_ratio = effective_ratio  # update for LayoutResult
    try:
        df_raw = _read(file_bytes, ext, header=None)
    except Exception:
        return LayoutResult(
            status="needs_confirmation",
            header_row=0,
            unnamed_ratio=unnamed_ratio,
            confidence=0.0,
            candidate_rows=tuple(range(min(5, _MAX_SCAN_ROWS))),
            message="Non-standard layout detected. Please select the header row.",
        )

    n_rows = len(df_raw)
    scan_up_to = min(_MAX_SCAN_ROWS, n_rows)

    scored: list[tuple[float, int]] = []   # (score, row_index)
    for i in range(scan_up_to):
        score = _header_row_score(df_raw, i)
        if score >= _HEADER_SCORE_THRESHOLD:
            scored.append((score, i))

    scored.sort(reverse=True)

    if not scored:
        # No good candidates found — expose rows 0..max(scan_up_to, 4) for manual selection
        fallback_n = max(scan_up_to, min(4, n_rows)) if n_rows > 0 else 4
        return LayoutResult(
            status="needs_confirmation",
            header_row=0,
            unnamed_ratio=unnamed_ratio,
            confidence=0.0,
            candidate_rows=tuple(range(fallback_n)),
            message=(
                "Non-standard layout detected but header row could not be identified. "
                "Please select the header row."
            ),
        )

    best_score, best_row = scored[0]
    candidate_rows = tuple(r for _, r in scored[:5])

    # ── Step 3: validate the best candidate by re-loading ────────────────────
    try:
        df_try = _read(file_bytes, ext, header=best_row)
        named_ratio = 1.0 - _unnamed_ratio(df_try)
    except Exception:
        named_ratio = 0.0

    if named_ratio >= _AUTO_FIX_NAMED_THRESHOLD and best_score >= _AUTO_FIX_CONFIDENCE_THRESHOLD:
        return LayoutResult(
            status="auto_fixed",
            header_row=best_row,
            unnamed_ratio=unnamed_ratio,
            confidence=round(best_score, 3),
            candidate_rows=candidate_rows,
            message=(
                f"Non-standard layout detected. "
                f"Headers found on row {best_row + 1} — loaded automatically."
            ),
        )

    if named_ratio >= _PARTIAL_NAMED_THRESHOLD:
        return LayoutResult(
            status="needs_confirmation",
            header_row=best_row,
            unnamed_ratio=unnamed_ratio,
            confidence=round(best_score, 3),
            candidate_rows=candidate_rows,
            message=(
                f"Header row is unclear (best guess: row {best_row + 1}). "
                "Please confirm which row contains the column names."
            ),
        )

    return LayoutResult(
        status="needs_confirmation",
        header_row=best_row,
        unnamed_ratio=unnamed_ratio,
        confidence=round(best_score, 3),
        candidate_rows=candidate_rows,
        message="Non-standard layout detected. Please select the header row.",
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read(file_bytes: bytes, ext: str, header: int | None) -> pd.DataFrame:
    """Read bytes into a DataFrame using the correct pandas reader."""
    buf = io.BytesIO(file_bytes)
    if ext in ("xlsx", "xls"):
        return pd.read_excel(buf, header=header)
    # Default: CSV
    return pd.read_csv(buf, header=header)


def _unnamed_ratio(df: pd.DataFrame) -> float:
    """Fraction of columns whose name starts with 'Unnamed:'."""
    if df.empty or len(df.columns) == 0:
        return 1.0
    n_unnamed = sum(1 for c in df.columns if str(c).startswith("Unnamed:"))
    return n_unnamed / len(df.columns)


def _numeric_colname_ratio(df: pd.DataFrame) -> float:
    """Fraction of column names that are raw integers or floats.

    When pandas reads a row of numbers as the header (e.g. [1, 2, 3, 4]),
    the column names become integer objects rather than descriptive strings.
    This is just as suspicious as 'Unnamed:' columns.
    """
    if df.empty or len(df.columns) == 0:
        return 0.0
    n_numeric = sum(1 for c in df.columns if isinstance(c, (int, float)))
    return n_numeric / len(df.columns)


def _header_row_score(df_raw: pd.DataFrame, row_idx: int) -> float:
    """Score row *row_idx* as a candidate header row.

    A good header row has:
    - High non-null ratio (most columns have a value)
    - High string ratio (most values are text, not numbers or dates)

    Returns a float in [0, 1].
    """
    row = df_raw.iloc[row_idx]
    non_null = row.dropna()
    if len(non_null) == 0:
        return 0.0

    non_null_ratio = len(non_null) / len(row)
    string_ratio = sum(1 for v in non_null if isinstance(v, str)) / len(non_null)
    return non_null_ratio * string_ratio


def preview_row(file_bytes: bytes, filename: str, row_idx: int, n_cells: int = 5) -> str:
    """Return a short preview of *row_idx* from the raw file for UI display."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    try:
        df_raw = _read(file_bytes, ext, header=None)
        if row_idx >= len(df_raw):
            return "(empty)"
        row = df_raw.iloc[row_idx].dropna()
        cells = [str(v)[:18] for v in row[:n_cells]]
        suffix = " …" if len(row) > n_cells else ""
        return " | ".join(cells) + suffix
    except Exception:
        return "(could not preview)"
