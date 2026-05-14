# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (includes dev dependencies)
pip install -e ".[dev]"

# Run the app
python -m streamlit run app.py

# Run all tests with coverage
pytest

# Run a single test file
pytest tests/test_agent_loop.py

# Run a single test
pytest tests/test_agent_loop.py::test_end_turn_returns_final_text

# Run tests without coverage (faster)
pytest --no-cov

# Install OpenAI provider support
pip install -e ".[openai]"
```

`pytest` automatically runs with `--cov=src --cov-report=term-missing` (configured in `pyproject.toml`).

## Architecture

### Two branches

- **`main`** — hand-rolled ReAct loop in `src/agent/loop.py`
- **`feat/langgraph`** — LangGraph `StateGraph` in `src/agent/langgraph_loop.py` (drop-in replacement, same `TurnResult` return type)

### Agent loop (`main` branch)

`run_agent_turn()` in `src/agent/loop.py` is the core of the system. It operates in one of two modes determined by which parameters are passed:

- **DataFrame mode** — pass `registry: DataFrameRegistry`; agent uses `execute_python` tool
- **SQL mode** — pass `sql_engine` + `sql_schema`; agent uses `execute_sql` tool

The loop is bounded at `MAX_TOOL_ITERATIONS = 5` (configurable in `src/config.py`). It is provider-agnostic — it only calls methods on `BaseLLMClient` and never touches SDK types directly.

### Provider abstraction

`BaseLLMClient` ABC (`src/agent/base.py`) defines three methods all providers must implement:
- `call(*, system, messages, tools) -> AgentResponse`
- `build_assistant_entry(response) -> dict`
- `build_tool_result_entries(results) -> list[dict]`

The history format differs per provider (Anthropic bundles tool results into one user message; OpenAI uses separate `role: "tool"` messages). This difference is fully encapsulated in `build_tool_result_entries()` — the loop calls `history.extend(...)` in both cases.

Add a new provider by subclassing `BaseLLMClient` and registering it in `src/config.py`'s `PROVIDERS` dict.

### DataFrameRegistry

`src/data/registry.py` — replaces the old single-`df` pattern. Multiple uploaded files are stored here, keyed by safe Python identifiers derived from filenames (`"Sales Data 2024.csv"` → `"sales_data_2024"`). The executor namespace is built from `registry.as_namespace()`, which exposes all DataFrames by name plus `df` aliased to the primary for backward compat.

### Frozen dataclasses as pipeline contracts

Every stage outputs an immutable frozen dataclass. Do not add mutable state to these:

| Dataclass | Module |
|-----------|--------|
| `SchemaContext` | `src/data/schema.py` |
| `EDAReport` | `src/eda/report.py` |
| `LayoutResult` | `src/data/layout.py` |
| `ExecutionResult` | `src/execution/result.py` |
| `TurnResult` | `src/agent/loop.py` |
| `TokenUsage` | `src/agent/base.py` |

### Python executor safety

`src/execution/python_executor.py` runs two checks before any `exec()`:
1. `_check_imports()` — AST walk that blocks 25+ modules (`os`, `subprocess`, `socket`, etc.). Returns a `SecurityError` `ExecutionResult` immediately if triggered; code never reaches `exec()`.
2. `df = df.copy()` in the exec namespace — agent code can never mutate the session DataFrame.

`matplotlib.use("Agg")` is set at module import time. Plotly figures are captured by scanning the post-exec namespace for `go.Figure` instances and serialised to JSON via `pio.to_json()` before leaving the executor.

### Chart validation

`src/execution/chart_validator.py` — runs after `execute_python` when Plotly figures are produced. Checks for empty traces, bar/pie charts with too many categories, missing axis labels, and generic titles. On failure, appends a correction prompt to the tool result summary so the LLM self-repairs on the next iteration — no extra API call or loop iteration consumed.

### Prompt caching (Anthropic only)

`AnthropicClient` (`src/agent/providers/anthropic_provider.py`) wraps the system prompt in `{"cache_control": {"type": "ephemeral"}}`. The system prompt is static within a session (schema + EDA narrative never changes), so it is served from cache at ~10% of input token cost after the first call.

## Testing

All tests run without real API calls. The key test double is `FakeLLMClient` in `tests/conftest.py` — a queue-based fake that returns pre-scripted `AgentResponse` objects and deep-copies the messages list on each `call()` to prevent snapshot corruption.

When adding tests for the agent loop, use `FakeLLMClient` rather than mocking the Anthropic SDK directly. Each test constructs a scenario as an ordered list of `FakeMessage` responses.

`tests/fixtures/sales.xlsx` is generated programmatically by a session-scoped autouse fixture in `conftest.py` — do not commit it as a binary.

## Key config values (`src/config.py`)

| Constant | Default | Purpose |
|----------|---------|---------|
| `MAX_TOOL_ITERATIONS` | `5` | Loop bound per agent turn |
| `MAX_TOKENS` | `4096` | LLM output token limit |
| `EXECUTION_MODE` | `"local"` | `"local"` / `"e2b"` / `"docker"` |
| `PROVIDERS` | Anthropic + OpenAI | Drives sidebar provider/model selectors |

## Execution backends (`src/execution/`)

`ExecutionBackend` ABC + `get_backend(mode)` factory in `backend_factory.py`. Current implementations:
- `LocalPythonExecutor` — delegates to the module-level `execute_python()` function
- `E2BExecutor`, `DockerExecutor` — stubs that raise `NotImplementedError` with setup instructions

## Streamlit session state keys

`app.py` reads and writes these keys — be aware when modifying UI or loop integration:

| Key | Type | Purpose |
|-----|------|---------|
| `registry` | `DataFrameRegistry` | All loaded DataFrames |
| `messages` | `list[dict]` | Full Anthropic-format chat history |
| `sql_conn` | `SQLConnection \| None` | Active SQL connection |
| `last_tool_calls` | `tuple[ToolCallRecord]` | Tool calls from the latest turn (for download buttons) |
| `last_question` / `last_answer` | `str` | Latest Q&A pair (for Markdown export) |
| `pending_query` | `str \| None` | Suggestion chip → chat input bridge (store → rerun → pick up) |
