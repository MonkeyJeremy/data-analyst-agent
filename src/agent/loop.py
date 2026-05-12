"""Bounded ReAct loop — provider-agnostic via :class:`BaseLLMClient`."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.agent.base import BaseLLMClient, TokenUsage
from src.agent.multi_prompt import build_multi_dataframe_prompt
from src.agent.system_prompt import build_system_prompt, build_sql_system_prompt
from src.agent.tools import get_tool_schemas, dispatch_tool
from src.config import MAX_TOOL_ITERATIONS
from src.data.registry import DataFrameRegistry
from src.data.schema import SchemaContext
from src.execution.chart_validator import validate_figures
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
    messages: list[dict]         # full updated history to persist in session state
    figures: tuple[bytes, ...]   # matplotlib PNG fallback figures
    plotly_figures: tuple[str, ...] = ()   # Plotly JSON figures (preferred)
    token_usage: TokenUsage | None = None  # cumulative session token counts


def run_agent_turn(
    client: BaseLLMClient,
    messages: list[dict],
    df: pd.DataFrame | None = None,
    schema: SchemaContext | None = None,
    eda_summary: str | None = None,
    sql_engine: Any | None = None,
    sql_schema: tuple | None = None,
    text_cols: tuple[str, ...] = (),
    viz_hint: str = "",
    registry: DataFrameRegistry | None = None,
    join_suggestions: list | None = None,
) -> TurnResult:
    """Run one user turn through the bounded ReAct loop.

    Supports two execution modes:

    **DataFrame mode** (default):
        Pass *registry* (preferred) or the deprecated *df* + *schema* pair.
        The agent uses the ``execute_python`` tool.

    **SQL mode**:
        Pass *sql_engine* and *sql_schema*.  The agent uses the ``execute_sql``
        tool.  DataFrame parameters are ignored.

    Parameters
    ----------
    client:
        Any :class:`~src.agent.base.BaseLLMClient` implementation
        (Anthropic, OpenAI, or a test double).
    messages:
        Current conversation history in the provider's expected format.
    df:
        Deprecated — single DataFrame for backward compatibility.  Ignored when
        *registry* is provided.
    schema:
        Deprecated — schema for the single *df*.  Ignored when *registry* is
        provided.
    eda_summary:
        Deprecated — EDA narrative for *df*.  Ignored when *registry* is
        provided.
    sql_engine:
        SQLAlchemy engine for SQL mode.
    sql_schema:
        Tuple of :class:`~src.db.schema.TableSchema` objects for SQL mode.
    text_cols:
        Names of free-form text columns for the primary table.  Used for the
        single-df path and as a fallback when *registry* is provided with one
        entry.
    viz_hint:
        Optional visualization hint injected into the system prompt.
    registry:
        Multi-dataframe registry.  When provided, *df*/*schema*/*eda_summary*
        are ignored.  A registry with a single entry behaves identically to
        the legacy single-df path from the agent's perspective.
    join_suggestions:
        Pre-computed join suggestions from :func:`~src.data.join_detector.detect_join_keys`.
        Only used when *registry* has more than one entry.

    Returns
    -------
    TurnResult
        Updated message history plus any figures produced.
    """
    # ── Determine mode ────────────────────────────────────────────────────────
    is_sql = sql_engine is not None
    mode = "sql" if is_sql else "dataframe"

    # ── Build dataframe namespace (DataFrame mode only) ───────────────────────
    # Resolve registry: if none provided, wrap legacy df param for compat.
    effective_registry = registry
    if effective_registry is None and df is not None:
        effective_registry = DataFrameRegistry()
        effective_registry.add("df", df)

    dataframes: dict[str, pd.DataFrame] | None = (
        effective_registry.as_namespace() if effective_registry is not None else None
    )

    # ── Build system prompt ───────────────────────────────────────────────────
    if is_sql:
        if sql_schema is None:
            sql_schema = ()
        system = build_sql_system_prompt(sql_schema)
    elif effective_registry is not None and effective_registry.count() > 1:
        # Multi-dataframe path: use richer prompt with join suggestions
        text_cols_by_table: dict[str, tuple[str, ...]] = {}
        for entry in effective_registry.entries():
            if entry.eda.text_cols:
                text_cols_by_table[entry.name] = entry.eda.text_cols
        system = build_multi_dataframe_prompt(
            effective_registry,
            join_suggestions or [],
            text_cols_by_table=text_cols_by_table,
            viz_hint=viz_hint,
        )
    else:
        # Single-dataframe path (legacy + new single-entry registry)
        if effective_registry is not None:
            primary = effective_registry.primary()
            if primary is not None:
                _schema = primary.schema
                _eda_summary = primary.eda.narrative
                _text_cols = primary.eda.text_cols
            else:
                raise ValueError("Registry is empty; cannot build system prompt.")
        else:
            if schema is None:
                raise ValueError("schema must be provided for DataFrame mode.")
            _schema = schema
            _eda_summary = eda_summary
            _text_cols = text_cols
        system = build_system_prompt(
            _schema, _eda_summary, text_cols=_text_cols, viz_hint=viz_hint
        )

    has_text_cols = bool(text_cols) and not is_sql
    if not has_text_cols and effective_registry is not None:
        has_text_cols = any(e.eda.text_cols for e in effective_registry.entries())
    tools = get_tool_schemas(mode, has_text_cols=has_text_cols)

    history = copy.deepcopy(messages)
    tool_calls: list[ToolCallRecord] = []
    all_figures: list[bytes] = []
    all_plotly_figures: list[str] = []

    for _ in range(MAX_TOOL_ITERATIONS):
        # client.call() accepts internal tool schema format and converts
        # to its own wire format internally
        response = client.call(system=system, messages=history, tools=tools)

        if response.stop_reason == "end_turn":
            history.append(client.build_assistant_entry(response))
            return TurnResult(
                final_text=response.text,
                tool_calls=tuple(tool_calls),
                messages=history,
                figures=tuple(all_figures),
                plotly_figures=tuple(all_plotly_figures),
                token_usage=client.usage,
            )

        if response.stop_reason == "tool_use":
            history.append(client.build_assistant_entry(response))

            raw_results: list[dict] = []
            for tc in response.tool_calls:
                result = dispatch_tool(
                    tc.name,
                    tc.input,
                    df=df if dataframes is None else None,
                    dataframes=dataframes,
                    sql_engine=sql_engine,
                    client=client,
                )

                # Validate Plotly figures; append issues to the summary so the
                # LLM can self-correct on the next iteration.
                summary = result.summary
                if tc.name == "execute_python" and result.plotly_figures:
                    validation = validate_figures(result.plotly_figures)
                    if not validation.valid:
                        summary = summary + "\n\n" + validation.correction_prompt

                tool_calls.append(
                    ToolCallRecord(
                        tool_name=tc.name,
                        tool_input=dict(tc.input),
                        result=result,
                    )
                )
                all_figures.extend(result.figures)
                all_plotly_figures.extend(result.plotly_figures)
                raw_results.append({"id": tc.id, "content": summary})

            # extend (not append) — OpenAI returns a list of separate messages
            history.extend(client.build_tool_result_entries(raw_results))
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
