"""Agent loop tests using FakeLLMClient — no real API calls."""
from __future__ import annotations

import pytest

from src.agent.loop import TurnResult, run_agent_turn
from tests.conftest import FakeLLMClient, FakeMessage, FakeTextBlock, FakeToolUseBlock


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_text_response(text: str) -> FakeMessage:
    return FakeMessage(stop_reason="end_turn", content=[FakeTextBlock(text=text)])


def _make_tool_response(code: str, purpose: str = "test") -> FakeMessage:
    return FakeMessage(
        stop_reason="tool_use",
        content=[
            FakeToolUseBlock(
                id="tu_001",
                name="execute_python",
                input={"code": code, "purpose": purpose},
            )
        ],
    )


# ── Scenario 1: direct answer, no tool use ────────────────────────────────────

def test_direct_answer(sample_df, sample_schema):
    client = FakeLLMClient([_make_text_response("The answer is 42.")])
    messages = [{"role": "user", "content": "What is 6 times 7?"}]

    result = run_agent_turn(client, messages, sample_df, sample_schema)

    assert isinstance(result, TurnResult)
    assert result.final_text == "The answer is 42."
    assert len(result.tool_calls) == 0
    assert len(client.calls) == 1


# ── Scenario 2: one tool call then final answer ───────────────────────────────

def test_one_tool_call(sample_df, sample_schema):
    client = FakeLLMClient(
        [
            _make_tool_response("print(len(df))"),
            _make_text_response("The dataframe has 4 rows."),
        ]
    )
    messages = [{"role": "user", "content": "How many rows?"}]

    result = run_agent_turn(client, messages, sample_df, sample_schema)

    assert result.final_text == "The dataframe has 4 rows."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "execute_python"
    assert len(client.calls) == 2


# ── Scenario 3: tool error then retry and recovery ────────────────────────────

def test_tool_error_retry(sample_df, sample_schema):
    client = FakeLLMClient(
        [
            _make_tool_response("raise ValueError('oops')"),   # first call → error
            _make_tool_response("print('fixed')"),              # retry with fix
            _make_text_response("Done after retry."),
        ]
    )
    messages = [{"role": "user", "content": "Try something"}]

    result = run_agent_turn(client, messages, sample_df, sample_schema)

    assert result.final_text == "Done after retry."
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].result.error is not None
    assert result.tool_calls[1].result.error is None
    # The error summary was passed back to the model
    tool_result_msg = client.calls[1]["messages"][-1]
    assert tool_result_msg["role"] == "user"
    assert "ERROR" in str(tool_result_msg["content"])


# ── Scenario 4: max iterations hit ───────────────────────────────────────────

def test_max_iterations(sample_df, sample_schema, monkeypatch):
    # Patch MAX_TOOL_ITERATIONS to 2 for speed
    import src.agent.loop as loop_mod
    monkeypatch.setattr(loop_mod, "MAX_TOOL_ITERATIONS", 2)

    # Always returns tool_use → never end_turn
    client = FakeLLMClient(
        [_make_tool_response("x = 1")] * 10
    )
    messages = [{"role": "user", "content": "Loop forever"}]

    result = run_agent_turn(client, messages, sample_df, sample_schema)

    assert "iteration limit" in result.final_text.lower()
    assert len(client.calls) == 2  # stopped after MAX_TOOL_ITERATIONS


# ── Misc ──────────────────────────────────────────────────────────────────────

def test_figures_collected(sample_df, sample_schema):
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\nax.bar(['A','B'],[1,2])"
    )
    client = FakeLLMClient(
        [
            _make_tool_response(code),
            _make_text_response("Here is the chart."),
        ]
    )
    messages = [{"role": "user", "content": "Plot a bar chart"}]

    result = run_agent_turn(client, messages, sample_df, sample_schema)

    assert len(result.figures) == 1
    assert result.figures[0][:4] == b"\x89PNG"


def test_messages_include_full_history(sample_df, sample_schema):
    client = FakeLLMClient(
        [
            _make_tool_response("print(42)"),
            _make_text_response("42"),
        ]
    )
    messages = [{"role": "user", "content": "What is 42?"}]

    result = run_agent_turn(client, messages, sample_df, sample_schema)

    roles = [m["role"] for m in result.messages]
    # user → assistant (tool_use) → user (tool_result) → assistant (end_turn)
    assert roles.count("user") >= 2
    assert roles.count("assistant") >= 2
