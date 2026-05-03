from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.agent.client import LLMClient
from src.agent.system_prompt import build_system_prompt
from src.agent.tools import TOOL_SCHEMAS, dispatch_tool
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
    messages: list[dict]  # full updated history to persist in session state
    figures: tuple[bytes, ...]  # all figures from this turn, in order


def run_agent_turn(
    client: LLMClient,
    messages: list[dict],
    df: pd.DataFrame,
    schema: SchemaContext,
    eda_summary: str | None = None,
) -> TurnResult:
    """Run one user turn through the bounded ReAct loop.

    Mutates a local copy of messages; the caller receives the updated history
    in TurnResult.messages and should persist it to session state.
    """
    history = copy.deepcopy(messages)
    system = build_system_prompt(schema, eda_summary)
    tool_calls: list[ToolCallRecord] = []
    all_figures: list[bytes] = []

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.call(system=system, messages=history, tools=TOOL_SCHEMAS)

        if response.stop_reason == "end_turn":
            final_text = _extract_text(response)
            history.append({"role": "assistant", "content": response.content})
            return TurnResult(
                final_text=final_text,
                tool_calls=tuple(tool_calls),
                messages=history,
                figures=tuple(all_figures),
            )

        if response.stop_reason == "tool_use":
            # Append the full assistant message (may contain text + tool_use blocks)
            history.append({"role": "assistant", "content": response.content})

            tool_results: list[dict] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = dispatch_tool(block.name, block.input, df)
                tool_calls.append(
                    ToolCallRecord(
                        tool_name=block.name,
                        tool_input=dict(block.input),
                        result=result,
                    )
                )
                all_figures.extend(result.figures)
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
    )


def _extract_text(response: Any) -> str:
    """Pull plain text out of a Message response."""
    parts: list[str] = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts).strip()
