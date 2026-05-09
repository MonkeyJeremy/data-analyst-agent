from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.agent.client import LLMClient, TokenUsage
from src.agent.system_prompt import build_system_prompt, build_sql_system_prompt
from src.agent.tools import get_tool_schemas, dispatch_tool
from src.config import MAX_TOOL_ITERATIONS
from src.data.schema import SchemaContext
from src.execution.result import ExecutionResult


@dataclass(frozen=True)
class ToolCallRecord:
    tool_name: str
    tool_input: dict
    result: ExecutionResult


@dataclass
class TurnResult:
    final_text: str
    tool_calls: tuple[ToolCallRecord, ...]
    messages: list[dict]        # full updated history to persist in session state
    figures: tuple[bytes, ...]  # matplotlib PNG fallback figures
    plotly_figures: tuple[str, ...] = ()  # Plotly JSON figures (preferred)
    token_usage: TokenUsage | None = None  # cumulative session token counts


def run_agent_turn(
    client: LLMClient,
    messages: list[dict],
    df: pd.DataFrame | None = None,
    schema: SchemaContext | None = None,
    eda_summary: str | None = None,
    sql_engine: Any | None = None,
    sql_schema: tuple | None = None,
    text_cols: tuple[str, ...] = (),
) -> TurnResult:
    """Run one user turn through the bounded ReAct loop.

    Supports two execution modes:

    **DataFrame mode** (default):
        Pass *df* and *schema*.  The agent uses the ``execute_python`` tool.

    **SQL mode**:
        Pass *sql_engine* and *sql_schema*.  The agent uses the ``execute_sql``
        tool.  *df* and *schema* are ignored.

    Parameters
    ----------
    client:
        LLM client for making API calls.
    messages:
        Current conversation history (mutated locally; original is unchanged).
    df:
        DataFrame for DataFrame mode.
    schema:
        Schema context for DataFrame mode.
    eda_summary:
        Pre-computed EDA narrative injected into the system prompt.
    sql_engine:
        SQLAlchemy engine for SQL mode.
    sql_schema:
        Tuple of :class:`~src.db.schema.TableSchema` objects for SQL mode.

    Returns
    -------
    TurnResult
        Updated message history plus any figures produced.
    """
    # ── Determine mode ────────────────────────────────────────────────────────
    is_sql = sql_engine is not None
    mode = "sql" if is_sql else "dataframe"

    # ── Build system prompt ───────────────────────────────────────────────────
    if is_sql:
        if sql_schema is None:
            sql_schema = ()
        system = build_sql_system_prompt(sql_schema)
    else:
        if schema is None:
            raise ValueError("schema must be provided for DataFrame mode.")
        system = build_system_prompt(schema, eda_summary, text_cols=text_cols)

    has_text_cols = bool(text_cols) and not is_sql
    tools = get_tool_schemas(mode, has_text_cols=has_text_cols)

    history = copy.deepcopy(messages)
    tool_calls: list[ToolCallRecord] = []
    all_figures: list[bytes] = []
    all_plotly_figures: list[str] = []

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.call(system=system, messages=history, tools=tools)

        if response.stop_reason == "end_turn":
            final_text = _extract_text(response)
            history.append({"role": "assistant", "content": response.content})
            return TurnResult(
                final_text=final_text,
                tool_calls=tuple(tool_calls),
                messages=history,
                figures=tuple(all_figures),
                plotly_figures=tuple(all_plotly_figures),
                token_usage=client.usage,
            )

        if response.stop_reason == "tool_use":
            history.append({"role": "assistant", "content": response.content})

            tool_results: list[dict] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = dispatch_tool(
                    block.name,
                    block.input,
                    df=df,
                    sql_engine=sql_engine,
                    client=client,
                )
                tool_calls.append(
                    ToolCallRecord(
                        tool_name=block.name,
                        tool_input=dict(block.input),
                        result=result,
                    )
                )
                all_figures.extend(result.figures)
                all_plotly_figures.extend(result.plotly_figures)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result.summary,
                    }
                )

            history.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop_reason (e.g. "max_tokens")
        break

    return TurnResult(
        final_text=(
            "I reached my iteration limit before completing the analysis. "
            "Please try rephrasing your question or breaking it into smaller steps."
        ),
        tool_calls=tuple(tool_calls),
        messages=history,
        figures=tuple(all_figures),
        plotly_figures=tuple(all_plotly_figures),
        token_usage=client.usage,
    )


def _extract_text(response: Any) -> str:
    """Pull plain text out of a Message response."""
    parts: list[str] = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts).strip()
