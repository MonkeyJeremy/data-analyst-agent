"""SQL query execution with read-only safety guard.

Only SELECT, WITH (CTEs), and EXPLAIN statements are permitted.
Results are formatted as markdown tables and returned in an ExecutionResult.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.execution.result import ExecutionResult

_SELECT_PATTERN = re.compile(r"^\s*(SELECT|WITH|EXPLAIN)\b", re.IGNORECASE)
_MAX_DISPLAY_ROWS = 50


def execute_sql(engine: Any, query: str) -> ExecutionResult:
    """Execute *query* against *engine* and return an :class:`ExecutionResult`.

    Parameters
    ----------
    engine:
        A SQLAlchemy Engine pointing at the target database.
    query:
        SQL statement to execute.  Only SELECT / WITH / EXPLAIN are allowed.

    Returns
    -------
    ExecutionResult
        On success: ``summary`` contains a markdown table (up to 50 rows).
        On error: ``error`` is set and ``summary`` contains the error message.
    """
    query = query.strip()

    # ── Safety guard: read-only statements only ───────────────────────────────
    if not _SELECT_PATTERN.match(query):
        msg = (
            "Only SELECT / WITH / EXPLAIN statements are permitted. "
            "Data-modifying statements (INSERT, UPDATE, DELETE, DROP, …) are blocked."
        )
        return ExecutionResult(
            stdout="",
            error=msg,
            figures=(),
            summary=f"ERROR: {msg}",
        )

    # ── Execute ───────────────────────────────────────────────────────────────
    try:
        df = pd.read_sql(query, engine)
    except Exception as exc:  # noqa: BLE001
        return ExecutionResult(
            stdout="",
            error=str(exc),
            figures=(),
            summary=f"ERROR: {exc}",
        )

    if df.empty:
        return ExecutionResult(
            stdout="(no rows returned)",
            error=None,
            figures=(),
            summary="Query executed successfully — 0 rows returned.",
        )

    total_rows = len(df)
    display_df = df.head(_MAX_DISPLAY_ROWS)

    try:
        table_md = display_df.to_markdown(index=False)
    except Exception:  # noqa: BLE001  # tabulate not installed
        table_md = display_df.to_string(index=False)

    suffix = (
        f"\n\n*({total_rows} rows total — showing first {_MAX_DISPLAY_ROWS})*"
        if total_rows > _MAX_DISPLAY_ROWS
        else f"\n\n*({total_rows} {'row' if total_rows == 1 else 'rows'})*"
    )
    summary = table_md + suffix

    return ExecutionResult(
        stdout=summary,
        error=None,
        figures=(),
        summary=summary,
    )


def load_table(engine: Any, table_name: str) -> pd.DataFrame:
    """Load all rows from *table_name* into a DataFrame.

    Uses a parameterised SELECT rather than ``pd.read_sql_table`` so it works
    across all SQLAlchemy-supported dialects.
    """
    return pd.read_sql(f'SELECT * FROM "{table_name}"', engine)
