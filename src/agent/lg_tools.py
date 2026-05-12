"""LangChain tool factories for the LangGraph agent.

Each factory closes over the execution context (DataFrames, SQL engine, LLM
client) and returns a ``@tool``-decorated callable that LangGraph's tools node
can invoke.  Figures produced during execution are pushed into caller-supplied
sink lists so the stateful graph can accumulate them across iterations.

Usage::

    figures: list[bytes] = []
    plotly_figures: list[str] = []
    tools = make_tools(
        dataframes=registry.as_namespace(),
        figures_sink=figures,
        plotly_sink=plotly_figures,
        client=client,            # only needed for analyze_text
    )
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.tools import tool

from src.execution.chart_validator import validate_figures
from src.execution.python_executor import execute_python
from src.execution.result import ExecutionResult


# ── helpers ───────────────────────────────────────────────────────────────────

def _push_result_artifacts(
    result: ExecutionResult,
    figures_sink: list[bytes],
    plotly_sink: list[str],
) -> str:
    """Side-effect: append figures from *result* to the caller's sinks.

    Returns the summary string to surface back to the LLM.
    """
    figures_sink.extend(result.figures)

    # Validate Plotly figures before storing so the LLM gets correction hints.
    if result.plotly_figures:
        validation = validate_figures(result.plotly_figures)
        plotly_sink.extend(result.plotly_figures)
        if not validation.valid:
            return result.summary + "\n\n" + validation.correction_prompt

    return result.summary


# ── tool factories ────────────────────────────────────────────────────────────

def make_execute_python_tool(
    dataframes: dict[str, pd.DataFrame],
    figures_sink: list[bytes],
    plotly_sink: list[str],
):
    """Return a ``execute_python`` LangChain tool bound to *dataframes*."""

    @tool
    def execute_python_tool(code: str, purpose: str) -> str:
        """Execute Python code against the loaded DataFrame(s).

        All DataFrames are available as named variables (see system prompt for
        names).  ``df`` is always an alias for the first/primary table.
        Pre-imported: pandas (pd), numpy (np), plotly.graph_objects (go),
        plotly.express (px), matplotlib.pyplot (plt), seaborn (sns).
        Prefer plotly for charts — assign the figure to any variable.
        Do NOT re-read files or use input() / network calls.
        """
        result = execute_python(code, dataframes)
        return _push_result_artifacts(result, figures_sink, plotly_sink)

    return execute_python_tool


def make_execute_sql_tool(
    sql_engine: Any,
    figures_sink: list[bytes],
    plotly_sink: list[str],
):
    """Return an ``execute_sql`` LangChain tool bound to *sql_engine*."""

    @tool
    def execute_sql_tool(query: str, purpose: str) -> str:
        """Execute a SQL SELECT query against the connected database.

        Returns a markdown table of results (up to 50 rows).  READ-ONLY —
        only SELECT, WITH (CTEs), and EXPLAIN are permitted.
        """
        from src.db.executor import execute_sql  # lazy import

        result = execute_sql(sql_engine, query)
        return _push_result_artifacts(result, figures_sink, plotly_sink)

    return execute_sql_tool


def make_analyze_text_tool(
    client: Any,
    figures_sink: list[bytes],
    plotly_sink: list[str],
):
    """Return an ``analyze_text`` LangChain tool that delegates to *client*."""

    @tool
    def analyze_text_tool(texts: list[str], task: str, purpose: str) -> str:
        """Analyse up to 50 text strings using the LLM.

        Returns a structured table with a label, confidence, and brief note
        for each text.  Use for sentiment, topic classification, intent
        detection, or any custom labelling task.
        """
        from src.text.analyzer import analyze_text_batch  # lazy import

        result = analyze_text_batch(client, texts, task)
        return _push_result_artifacts(result, figures_sink, plotly_sink)

    return analyze_text_tool


def make_tools(
    *,
    dataframes: dict[str, pd.DataFrame] | None = None,
    sql_engine: Any | None = None,
    has_text_cols: bool = False,
    client: Any | None = None,
    figures_sink: list[bytes],
    plotly_sink: list[str],
) -> list:
    """Assemble the tool list for the current execution mode.

    Parameters
    ----------
    dataframes:
        DataFrame namespace for Python execution mode.
    sql_engine:
        SQLAlchemy engine for SQL execution mode.
    has_text_cols:
        When ``True``, includes the ``analyze_text`` tool.
    client:
        LLM client forwarded to ``analyze_text``; only used when
        *has_text_cols* is ``True``.
    figures_sink / plotly_sink:
        Mutable lists that tool executions append their artefacts to.
    """
    if sql_engine is not None:
        return [make_execute_sql_tool(sql_engine, figures_sink, plotly_sink)]

    if dataframes is None:
        raise ValueError("Either dataframes or sql_engine must be provided.")

    tools = [make_execute_python_tool(dataframes, figures_sink, plotly_sink)]
    if has_text_cols and client is not None:
        tools.append(make_analyze_text_tool(client, figures_sink, plotly_sink))
    return tools
