# Data Analyst Agent

A conversational AI agent that lets you upload a CSV, Excel, JSON, or SQLite file — or connect to a live SQL database — and ask questions about your data in plain English. Every answer is backed by real Python or SQL execution. The model never guesses numbers.

## Demo

```
User: What is the average sleep hours by gender?
Agent: [runs pandas code] Males average 6.8h, females average 7.1h.

User: Plot that as a bar chart.
Agent: [runs plotly code] [interactive chart appears inline]

User: What is the overall sentiment in the review column?
Agent: [calls analyze_text] [markdown table: label / confidence / note per row]
```

## Features

- **Natural language Q&A** — powered by a bounded ReAct loop; works with Anthropic Claude or OpenAI GPT models
- **Real code execution** — agent writes and runs Python/pandas or SQL against your data
- **Interactive charts** — Plotly figures rendered inline; matplotlib as fallback
- **Auto-retry on errors** — agent reads the traceback, fixes the code, and retries (max 2)
- **Auto EDA on upload** — distributions, correlations, missing values, outliers, skewness
- **Text analysis** — detects free-form text columns; nested Claude call for sentiment / topic classification
- **Multi-format upload** — `.csv`, `.xlsx`, `.xls`, `.json`, `.db`, `.sqlite`
- **SQL mode** — connect to SQLite or any SQLAlchemy-compatible database
- **Layout detection** — handles messy spreadsheets with blank rows, multi-row headers, merged cells
- **Multi-provider** — switch between Anthropic Claude and OpenAI GPT models from the sidebar; extensible to Gemini, Mistral, Groq, Ollama, and others via `BaseLLMClient`
- **Prompt caching** — system prompt cached across turns; ~80% reduction in input token billing (Anthropic)
- **Token metering** — live session token count + cache hit rate displayed in sidebar
- **Sandboxed execution** — AST import blocklist blocks `os`, `subprocess`, `socket`, and 20+ other dangerous modules before any code runs; DataFrame is never mutated

## Architecture

**`main` branch — hand-rolled ReAct loop (`src/agent/loop.py`)**

```
User question
  └─▶ build_system_prompt(schema, eda_summary, text_cols)
  └─▶ BaseLLMClient.call()          # Anthropic | OpenAI | … (swap via sidebar)
        ├─ stop_reason == end_turn  ──▶ return final text
        └─ stop_reason == tool_use
              ├─ execute_python(code, df)   # AST check → sandboxed exec()
              ├─ execute_sql(engine, query) # SELECT-only whitelist
              └─ analyze_text(texts, task)  # nested LLM call → JSON → table
              └─▶ append tool result → loop (max 5 iterations)
```

**`feat/langgraph` branch — LangGraph `StateGraph` (`src/agent/langgraph_loop.py`)**

```
User question
  └─▶ build_system_prompt(...)  ──▶  stored in AnalystState.system
  └─▶ StateGraph.invoke(state, config={thread_id})
        agent_node: ChatAnthropic.invoke([SystemMessage] + state.messages)
          ├─ AIMessage has tool_calls? ──▶ tools_node
          │     tools_node: dispatch each @tool, push figures to sink
          │     └─▶ ToolMessage results ──▶ back to agent_node
          └─ no tool_calls / max_iter ──▶ END
  └─▶ MemorySaver checkpointer persists state keyed by thread_id
```

**Key design choices:**

| Choice | Rationale |
|--------|-----------|
| Bounded ReAct loop (max 5) | Prevents runaway API costs; failure mode is explicit |
| Scripts over reasoning | EDA, layout detection, schema introspection are deterministic scripts — model only orchestrates |
| Prompt caching on system prompt | Schema + EDA context is static per session; caching cuts input token cost 80–90% |
| AST import blocklist | Blocks dangerous modules before `exec()` — no code ever runs that imports `os`, `subprocess`, etc. |
| `df.copy()` in executor | Agent code can never mutate the user's session DataFrame |
| Nested LLM call for text analysis | Zero NLP library dependencies; handles any classification task the user describes |
| `BaseLLMClient` provider abstraction | Loop never touches SDK types; swap Anthropic ↔ OpenAI ↔ others by implementing 3 methods |
| Frozen dataclasses as pipeline contracts | `SchemaContext`, `EDAReport`, `ExecutionResult`, `TurnResult` — immutable, hashable, low hallucination surface |
| `FakeLLMClient` test double | All 129 tests run without real API calls; agent loop tested as a state machine |

## Project Structure

```
data-analyst-agent/
├── app.py                          # Streamlit entrypoint
├── src/
│   ├── config.py                   # MODEL_NAME, MAX_TOKENS, MAX_TOOL_ITERATIONS
│   ├── agent/
│   │   ├── base.py                 # BaseLLMClient ABC + AgentResponse / ToolCall / TokenUsage
│   │   ├── client.py               # create_client() factory; LLMClient alias for backward compat
│   │   ├── providers/
│   │   │   ├── anthropic_provider.py  # AnthropicClient — prompt caching + token metering
│   │   │   └── openai_provider.py     # OpenAIClient — tool-schema conversion + history format
│   │   ├── loop.py                 # Bounded ReAct loop → TurnResult (provider-agnostic)
│   │   ├── system_prompt.py        # Schema + EDA + text-cols system prompt builder
│   │   └── tools.py                # Tool schemas + dispatch (python / sql / analyze_text)
│   ├── data/
│   │   ├── loader.py               # CSV / Excel / JSON → pd.DataFrame
│   │   ├── schema.py               # DataFrame → SchemaContext (frozen dataclass)
│   │   └── layout.py               # Messy spreadsheet layout detection + auto-fix
│   ├── db/
│   │   ├── connection.py           # SQLite file upload → SQLAlchemy engine
│   │   ├── executor.py             # SELECT-only SQL executor
│   │   └── schema.py               # DB introspection → TableSchema
│   ├── eda/
│   │   ├── auto_eda.py             # run_auto_eda() — pure function, no side effects
│   │   └── report.py               # EDAReport frozen dataclass
│   ├── execution/
│   │   ├── python_executor.py      # AST safety check + sandboxed exec() + chart capture
│   │   └── result.py               # ExecutionResult frozen dataclass
│   ├── text/
│   │   ├── eda.py                  # detect_text_cols() + compute_top_words()
│   │   └── analyzer.py             # analyze_text_batch() — nested Claude call
│   └── ui/
│       ├── eda_panel.py            # EDA expander (Overview / Distributions / Correlations / Text)
│       ├── chat_panel.py           # Chat history + inline chart renderer
│       ├── layout_panel.py         # Layout-fix confirmation banner
│       ├── sql_panel.py            # SQL connect panel + stats bar
│       └── upload_panel.py         # Sidebar file uploader
└── tests/                          # 129 tests, 80–100% coverage on all non-UI modules
```

## Getting Started

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/) **or** an [OpenAI API key](https://platform.openai.com/api-keys)
  — other providers (Gemini, Mistral, Groq, Ollama) can be added by implementing `BaseLLMClient`
- Windows: enable [Win32 Long Paths](https://pip.pypa.io/warnings/enable-long-paths) before installing

### Install

```bash
git clone https://github.com/MonkeyJeremy/data-analyst-agent.git
cd data-analyst-agent
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# For Anthropic:  set ANTHROPIC_API_KEY=sk-ant-...
# For OpenAI:     set OPENAI_API_KEY=sk-...
```

### Run

```bash
python -m streamlit run app.py
```

Or enter your API key directly in the sidebar — no `.env` needed.

### Test

```bash
pytest --cov=src --cov-report=term-missing
```

## v8 — LangGraph Orchestration (`feat/langgraph` branch)

> **Branch:** `feat/langgraph`  
> The `main` branch keeps the hand-rolled ReAct loop (`src/agent/loop.py`).
> This branch refactors the agent orchestration layer to use **LangGraph**,
> replacing the manual `while` loop with a declarative `StateGraph`.

### What changed

The hand-rolled bounded loop in `loop.py`:

```python
for _ in range(MAX_TOOL_ITERATIONS):
    response = client.call(...)
    if response.stop_reason == "end_turn": return ...
    if response.stop_reason == "tool_use":
        for tc in response.tool_calls:
            result = dispatch_tool(tc.name, ...)
        history.extend(...)
        continue
    break
```

…is replaced by a declarative `StateGraph` in `src/agent/langgraph_loop.py`:

```
                   ┌──────────┐   tool_calls   ┌──────────┐
                   │          │───────────────►│          │
       START ────► │  agent   │                │  tools   │
                   │  (LLM)   │◄───────────────│  (exec)  │
                   │          │    results     │          │
                   └────┬─────┘                └──────────┘
                        │ end_turn / max_iter
                        ▼
                       END
```

### New files

| File | Purpose |
|------|---------|
| `src/agent/langgraph_loop.py` | `StateGraph` definition, `AnalystState` TypedDict, message converters, `run_langgraph_turn()` |
| `src/agent/lg_tools.py` | LangChain `@tool` factories that wrap the existing executors |

### LangGraph concepts demonstrated

| Concept | Where |
|---------|-------|
| **Typed state** (`TypedDict` + `add_messages` reducer) | `AnalystState` in `langgraph_loop.py` |
| **`StateGraph` with named nodes** | `build_analyst_graph()` — `"agent"` and `"tools"` nodes |
| **Conditional edges** | `should_continue()` routes to `"tools"` or `END` based on the last message |
| **`MemorySaver` checkpointing** | Compiled with `checkpointer=MemorySaver()`; keyed by `thread_id` per session |
| **`ChatAnthropic` + `.bind_tools()`** | LangChain model interface replaces `BaseLLMClient` |
| **Closure-based tool factories** | `make_tools()` in `lg_tools.py` — tools close over dataframes + figure sinks |

### Why LangGraph improves on the hand-rolled loop

| Dimension | Hand-rolled (`loop.py`) | LangGraph (`langgraph_loop.py`) |
|-----------|------------------------|--------------------------------|
| Control flow | `while` + `if/elif` + `break/continue` | Declarative graph edges — topology is visual and auditable |
| State management | Mutable local lists; manual `append`/`extend` | Immutable `TypedDict` with `add_messages` reducer; each node returns a diff |
| Checkpointing | None — full history re-sent every turn | `MemorySaver` persists state per thread; resume mid-conversation |
| Extensibility | Add another `elif` branch | Add a new node + edge — no changes to existing logic |
| Testability | Mock the `client` and trace `history` mutations | Invoke the graph with a fake model; inspect state snapshots |

### Backward compatibility

`run_langgraph_turn()` is a **drop-in replacement** for `run_agent_turn()`:
- Same parameters (registry, sql_engine, text_cols, viz_hint, …)
- Returns the same `TurnResult` dataclass
- `st.session_state["messages"]` stays in Anthropic dict format (converted at the boundary)

## Build Phases

| Phase | Status | What it adds |
|-------|--------|-------------|
| v1 — Core | ✅ Done | CSV/Excel upload, natural language Q&A, ReAct loop, inline charts |
| v2 — SQL + Layout | ✅ Done | SQLite/SQL mode, messy spreadsheet layout detection, JSON support |
| v3 — Auto EDA | ✅ Done | Auto-generated EDA panel (distributions, correlations, outliers, skewness) |
| v4 — Text Analysis | ✅ Done | Text column detection, word frequency, nested Claude call for sentiment/topics |
| v5 — Hardening | ✅ Done | Prompt caching, token metering, AST import sandbox, README update |
| v6 — Multi-provider | ✅ Done | `BaseLLMClient` ABC; Anthropic + OpenAI providers; extensible to Gemini, Groq, Ollama |
| v7 — Multi-dataframe | ✅ Done | Upload multiple files, auto join detection, manual relationship builder, proactive join suggestions |
| v8 — LangGraph | ✅ Done (`feat/langgraph`) | LangGraph `StateGraph` replaces hand-rolled ReAct loop; `MemorySaver` checkpointing; typed agent state |

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [Anthropic Claude API](https://docs.anthropic.com/) — LLM with tool use + prompt caching
- [OpenAI API](https://platform.openai.com/docs) — alternative LLM provider
- [LangGraph](https://langchain-ai.github.io/langgraph/) — agent orchestration (`feat/langgraph`): `StateGraph`, conditional edges, `MemorySaver` checkpointing
- [LangChain Anthropic](https://python.langchain.com/docs/integrations/chat/anthropic/) — `ChatAnthropic` model with `.bind_tools()` (`feat/langgraph`)
- [pandas](https://pandas.pydata.org/) + [plotly](https://plotly.com/python/) + [matplotlib](https://matplotlib.org/) — data analysis and charts
- [SQLAlchemy](https://www.sqlalchemy.org/) — SQL database connectivity
- [pytest](https://pytest.org/) — testing (129 tests)

## Security Note

The Python executor uses `exec()` on agent-generated code. Dangerous imports (`os`, `subprocess`, `socket`, etc.) are blocked at the AST level before any code runs. The current version is intended for **local development**. For production deployment, replace the executor with a fully isolated sandbox such as [E2B](https://e2b.dev/) or a Docker container with no network access, filesystem isolation, and CPU/memory limits.

## Dev Log

See [`dev-log.md`](dev-log.md) for a structured build journal tracking decisions, mistakes, fixes, and insights from each development session.

See [`PROJECT_REPORT.md`](PROJECT_REPORT.md) for a formal architectural write-up covering all engineering decisions, tradeoffs, and lessons learned — written for interview preparation.
