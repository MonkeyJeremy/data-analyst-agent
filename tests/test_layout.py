"""Tests for src/data/layout.py — detect_layout() heuristics."""
from __future__ import annotations

import io
import pathlib

import openpyxl
import pandas as pd
import pytest

from src.data.layout import LayoutResult, detect_layout, preview_row
from src.data.loader import load_tabular

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


# ── helpers ───────────────────────────────────────────────────────────────────

def _csv_bytes(*rows: list) -> bytes:
    """Build CSV bytes from a list of rows (each row is a list of values)."""
    lines = [",".join(str(v) for v in row) for row in rows]
    return "\n".join(lines).encode()


def _xlsx_bytes(rows: list[list]) -> bytes:
    """Build Excel bytes from a list of rows."""
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ── clean / standard files ────────────────────────────────────────────────────

def test_clean_csv_ok():
    """Standard CSV with named headers → status='ok'."""
    data = _csv_bytes(
        ["name", "age", "salary"],
        ["Alice", 30, 70000],
        ["Bob", 25, 55000],
    )
    result = detect_layout(data, "data.csv")
    assert result.status == "ok"
    assert result.header_row == 0


def test_clean_xlsx_ok():
    """Standard Excel with named headers → status='ok'."""
    data = _xlsx_bytes([
        ["name", "age", "salary"],
        ["Alice", 30, 70000],
        ["Bob", 25, 55000],
    ])
    result = detect_layout(data, "data.xlsx")
    assert result.status == "ok"


def test_unnamed_ratio_clean():
    """No Unnamed columns → unnamed_ratio near 0."""
    data = _csv_bytes(
        ["col_a", "col_b", "col_c"],
        [1, 2, 3],
    )
    result = detect_layout(data, "data.csv")
    assert result.unnamed_ratio < 0.30


# ── auto-fix cases ────────────────────────────────────────────────────────────

def test_blank_first_row_auto_fixed():
    """Blank row 0, real header on row 1 → auto_fixed with header_row=1."""
    data = _xlsx_bytes([
        [None, None, None],          # blank row 0
        ["Product", "Region", "Revenue"],
        ["Widget A", "North", 500],
        ["Widget B", "South", 700],
    ])
    result = detect_layout(data, "sales.xlsx")
    assert result.status in ("auto_fixed", "needs_confirmation")
    assert result.header_row == 1


def test_multi_blank_rows_then_header():
    """Two blank rows then a proper header → detected."""
    data = _xlsx_bytes([
        [None, None, None],
        [None, None, None],
        ["Date", "Channel", "Sales"],
        ["2024-01-01", "Amazon", 1000],
        ["2024-01-08", "Shopify", 500],
    ])
    result = detect_layout(data, "report.xlsx")
    assert result.status in ("auto_fixed", "needs_confirmation")
    assert result.header_row == 2


def test_auto_fixed_message_mentions_row():
    """auto_fixed message should mention the detected row number (1-based)."""
    data = _xlsx_bytes([
        [None, None, None],
        ["Name", "Age", "City"],
        ["Alice", 30, "London"],
    ])
    result = detect_layout(data, "people.xlsx")
    if result.status == "auto_fixed":
        assert str(result.header_row + 1) in result.message


# ── QC Sales-like structure ───────────────────────────────────────────────────

def test_qc_sales_structure():
    """Simulated QC Sales: blank, months, dates, data → auto_fixed or needs_confirmation."""
    months = [None, "Totals", None, "JUNE", "JUNE", "JUNE", "JULY", "JULY"]
    dates  = [None, None,     None, "5",    "12",   "19",   "3",    "10"  ]
    data_r = ["Amazon", 100000, None, 25750, 25770, 26495, 26865, 28015]
    data = _xlsx_bytes([
        [None] * 8,   # blank row 0
        months,
        dates,
        [None] * 8,   # blank separator
        data_r,
    ])
    result = detect_layout(data, "qc_sales.xlsx")
    # Should not return "ok" — the default load would have unnamed cols
    assert result.status in ("auto_fixed", "needs_confirmation")


# ── needs_confirmation cases ──────────────────────────────────────────────────

def test_all_blank_rows_needs_confirmation():
    """File with all blank rows → needs_confirmation."""
    data = _xlsx_bytes([
        [None, None, None],
        [None, None, None],
        [None, None, None],
    ])
    result = detect_layout(data, "empty.xlsx")
    assert result.status == "needs_confirmation"


def test_low_confidence_needs_confirmation():
    """First row is all numbers → low confidence, needs_confirmation."""
    data = _xlsx_bytes([
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        [11, 22, 33, 44, 55],
    ])
    result = detect_layout(data, "numbers.xlsx")
    # All numeric → string_ratio ≈ 0 → no good candidate → needs_confirmation
    assert result.status == "needs_confirmation"


def test_candidate_rows_returned_on_confirmation():
    """needs_confirmation result must include at least one candidate row."""
    data = _xlsx_bytes([
        [None, None, None],
        [None, None, None],
        [None, None, None],
    ])
    result = detect_layout(data, "blank.xlsx")
    assert result.status == "needs_confirmation"
    assert len(result.candidate_rows) >= 1


# ── LayoutResult fields ───────────────────────────────────────────────────────

def test_unnamed_ratio_field():
    """unnamed_ratio should be high for a file with blank first row."""
    data = _xlsx_bytes([
        [None, None, None, None],
        ["A", "B", "C", "D"],
        [1, 2, 3, 4],
    ])
    result = detect_layout(data, "test.xlsx")
    # Default load (header=0) picks up the blank row as header → mostly Unnamed
    assert result.unnamed_ratio >= 0.50


def test_confidence_zero_for_undetected():
    """Confidence should be 0 when nothing is detected."""
    data = _xlsx_bytes([[None] * 5] * 5)
    result = detect_layout(data, "blank.xlsx")
    assert result.confidence == 0.0


# ── load_tabular() with header param ─────────────────────────────────────────

def test_load_tabular_with_header_param():
    """`load_tabular(..., header=2)` should use row 2 as the column header."""
    data = _xlsx_bytes([
        [None, None, None],       # row 0 blank
        [None, None, None],       # row 1 blank
        ["Product", "Sales", "Region"],  # row 2 = real header
        ["Widget A", 500, "North"],
        ["Widget B", 700, "South"],
    ])
    buf = io.BytesIO(data)
    df = load_tabular(buf, "test.xlsx", header=2)
    assert list(df.columns) == ["Product", "Sales", "Region"]
    assert len(df) == 2


def test_load_tabular_default_unchanged():
    """Default call (no header arg) still works as before."""
    df = load_tabular(FIXTURES / "titanic.csv", "titanic.csv")
    assert "PassengerId" in df.columns
    assert len(df) == 10


# ── preview_row ───────────────────────────────────────────────────────────────

def test_preview_row_returns_string():
    data = _xlsx_bytes([
        ["Amazon EU", "Totals", "JUNE", "JUNE"],
        [None, 12345, 5000, 4800],
    ])
    s = preview_row(data, "test.xlsx", 0)
    assert isinstance(s, str)
    assert len(s) > 0


def test_preview_row_out_of_bounds():
    data = _xlsx_bytes([["A", "B"]])
    s = preview_row(data, "test.xlsx", 99)
    assert s == "(empty)"
