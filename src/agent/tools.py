"""Agent tool schemas and dispatch routing.

Two execution modes:
- ``"dataframe"`` — ``execute_python`` runs arbitrary pandas/plotly code against ``df``.
- ``"sql"``       — ``execute_sql`` runs read-only SQL queries against a live engine.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.execution.python_executor import execute_python
from src.execution.result import ExecutionResult

# ── Tool definitions ──────────────────────────────────────────────────────────

_PYTHON_TOOL: dict = {
    "name": "execute_python",
    "description": (
        "Execute Python code against the user's DataFrame(s). "
        "All loaded DataFrames are available as named variables (see system prompt for names). "
        "`df` is always an alias for the first/primary table. "
        "Pre-imported: pandas (pd), numpy (np), "
        "plotly.graph_objects (go), plotly.express (px), "
        "matplotlib.pyplot (plt), seaborn (sns). "
        "PREFER plotly (go or px) for all charts — assign the figure to any variable "
        "and it will be rendered interactively (e.g. `fig = px.bar(...)`). "
        "Use print() for text output. "
        "Do NOT re-read the file. Do NOT use input() or network calls."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Valid Python code to execute.",
            },
            "purpose": {
                "type": "string",
                "description": "One-sentence explanation shown to the user.",
            },
        },
        "required": ["code", "purpose"],
    },
}

_SQL_TOOL: dict = {
    "name": "execute_sql",
    "description": (
        "Execute a SQL SELECT query against the connected database. "
        "Returns a markdown table of results (up to 50 rows). "
        "Use this to query, filter, aggregate, and explore the database tables. "
        "READ-ONLY — only SELECT, WITH (CTEs), and EXPLAIN are permitted."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The SQL SELECT statement to execute.",
            },
            "purpose": {
                "type": "string",
                "description": "One-sentence explanation of what this query finds.",
            },
        },
        "required": ["query", "purpose"],
    },
}

_TEXT_TOOL: dict = {
    "name": "analyze_text",
    "description": (
        "Analyse a list of text strings using Claude. Returns a structured table with "
        "a label, confidence, and brief note for each text. "
        "Use for: sentiment (positive/negative/neutral), topic classification, "
        "intent detection, custom categories — anything expressible as a labelling task. "
        "Pass df['col'].dropna().head(30).tolist() to sample a column. Max 50 texts per call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "texts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of text strings to analyse (max 50).",
            },
            "task": {
                "type": "string",
                "description": (
                    "What to analyse. Examples: 'sentiment', 'main topic in 1-3 words', "
                    "'classify as complaint / praise / neutral', 'extract product name mentioned'."
                ),
            },
            "purpose": {
                "type": "string",
                "description": "One-sentence explanation shown to the user.",
            },
        },
        "required": ["texts", "task", "purpose"],
    },
}

# Backward-compatible alias (used by existing tests that import TOOL_SCHEMAS directly)
TOOL_SCHEMAS: list[dict] = [_PYTHON_TOOL]


def get_tool_schemas(mode: str = "dataframe", has_text_cols: bool = False) -> list[dict]:
    """Return the tool list for the given execution *mode*.

    Parameters
    ----------
    mode:
        ``"dataframe"`` (default) — returns the ``execute_python`` tool.
        ``"sql"``                 — returns the ``execute_sql`` tool.
    has_text_cols:
        When ``True`` and *mode* is ``"dataframe"``, also includes the
        ``analyze_text`` tool.

    Returns
    -------
    list[dict]
        List of Anthropic tool schema dicts ready for the ``tools=`` parameter.
    """
    if mode == "sql":
        return [_SQL_TOOL]
    if has_text_cols:
        return [_PYTHON_TOOL, _TEXT_TOOL]
    return [_PYTHON_TOOL]


def dispatch_tool(
    name: str,
    tool_input: dict,
    df: pd.DataFrame | None = None,
    dataframes: dict[str, pd.DataFrame] | None = None,
    sql_engine: Any | None = None,
    client: Any | None = None,
) -> ExecutionResult:
    """Route a tool call to its implementation.

    Parameters
    ----------
    name:
        Tool name from the API response block.
    tool_input:
        Tool input dict from the API response block.
    df:
        Deprecated single-DataFrame shorthand.  When *dataframes* is also
        provided, *df* is ignored.  When only *df* is provided it is wrapped
        in ``{"df": df}`` for backward compatibility.
    dataframes:
        All loaded DataFrames keyed by their registered name.  Should include
        a ``"df"`` key as an alias for the primary table.
    sql_engine:
        SQLAlchemy engine for SQL mode; ignored in DataFrame mode.

    Returns
    -------
    ExecutionResult
        Error field is set for unknown tools or routing failures.
    """
    if name == "analyze_text":
        if client is None:
            return ExecutionResult(
                stdout="",
                error="No LLM client available for analyze_text.",
                figures=(),
                summary="ERROR: No LLM client provided.",
            )
        from src.text.analyzer import analyze_text_batch
        return analyze_text_batch(
            client,
            tool_input.get("texts", []),
            tool_input.get("task", ""),
        )

    if name == "execute_python":
        # Resolve the dataframe namespace — prefer the explicit dict; fall back
        # to wrapping the legacy single-df param for backward compatibility.
        resolved: dict[str, pd.DataFrame] | None = dataframes
        if resolved is None and df is not None:
            resolved = {"df": df}
        if resolved is None:
            return ExecutionResult(
                stdout="",
                error="No DataFrame available for execute_python.",
                figures=(),
                summary="ERROR: No DataFrame loaded.",
            )
        code = tool_input.get("code", "")
        return execute_python(code, resolved)

    if name == "execute_sql":
        if sql_engine is None:
            return ExecutionResult(
                stdout="",
                error="No SQL engine available for execute_sql.",
                figures=(),
                summary="ERROR: No SQL connection active.",
            )
        from src.db.executor import execute_sql
        query = tool_input.get("query", "")
        return execute_sql(sql_engine, query)

    return ExecutionResult(
        stdout="",
        error=f"Unknown tool: '{name}'",
        figures=(),
        summary=f"ERROR: Unknown tool '{name}'",
    )
