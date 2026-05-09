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

- **Natural language Q&A** — powered by Claude tool-use API (ReAct loop)
- **Real code execution** — agent writes and runs Python/pandas or SQL against your data
- **Interactive charts** — Plotly figures rendered inline; matplotlib as fallback
- **Auto-retry on errors** — agent reads the traceback, fixes the code, and retries (max 2)
- **Auto EDA on upload** — distributions, correlations, missing values, outliers, skewness
- **Text analysis** — detects free-form text columns; nested Claude call for sentiment / topic classification
- **Multi-format upload** — `.csv`, `.xlsx`, `.xls`, `.json`, `.db`, `.sqlite`
- **SQL mode** — connect to SQLite or any SQLAlchemy-compatible database
- **Layout detection** — handles messy spreadsheets with blank rows, multi-row headers, merged cells
- **Prompt caching** — system prompt cached across turns; ~80% reduction in input token billing
- **Token metering** — live session token count + cache hit rate displayed in sidebar
- **Sandboxed execution** — AST import blocklist blocks `os`, `subprocess`, `socket`, and 20+ other dangerous modules before any code runs; DataFrame is never mutated

## Architecture

```
User question
  └─▶ build_system_prompt(schema, eda_summary, text_cols)
  └─▶ Claude API — tool-use / cached system prompt
        ├─ stop_reason == end_turn  ──▶ return final text
        └─ stop_reason == tool_use
              ├─ execute_python(code, df)   # AST check → sandboxed exec()
              ├─ execute_sql(engine, query) # SELECT-only whitelist
              └─ analyze_text(texts, task)  # nested Claude call → JSON → table
              └─▶ append tool result → loop (max 5 iterations)
```

**Key design choices:**

| Choice | Rationale |
|--------|-----------|
| Bounded ReAct loop (max 5) | Prevents runaway API costs; failure mode is explicit |
| Scripts over reasoning | EDA, layout detection, schema introspection are deterministic scripts — model only orchestrates |
| Prompt caching on system prompt | Schema + EDA context is static per session; caching cuts input token cost 80–90% |
| AST import blocklist | Blocks dangerous modules before `exec()` — no code ever runs that imports `os`, `subprocess`, etc. |
| `df.copy()` in executor | Agent code can never mutate the user's session DataFrame |
| Nested Claude call for text analysis | Zero NLP library dependencies; handles any classification task the user describes |
| Frozen dataclasses as pipeline contracts | `SchemaContext`, `EDAReport`, `ExecutionResult`, `TurnResult` — immutable, hashable, low hallucination surface |
| `FakeLLMClient` test double | All 121 tests run without real API calls; agent loop tested as a state machine |

## Project Structure

```
data-analyst-agent/
├── app.py                          # Streamlit entrypoint
├── src/
│   ├── config.py                   # MODEL_NAME, MAX_TOKENS, MAX_TOOL_ITERATIONS
│   ├── agent/
│   │   ├── client.py               # Anthropic API wrapper — prompt caching + token metering
│   │   ├── loop.py                 # Bounded ReAct loop → TurnResult
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
└── tests/                          # 121 tests, 80–100% coverage on all non-UI modules
```

## Getting Started

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)
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
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
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

## Build Phases

| Phase | Status | What it adds |
|-------|--------|-------------|
| v1 — Core | ✅ Done | CSV/Excel upload, natural language Q&A, ReAct loop, inline charts |
| v2 — SQL + Layout | ✅ Done | SQLite/SQL mode, messy spreadsheet layout detection, JSON support |
| v3 — Auto EDA | ✅ Done | Auto-generated EDA panel (distributions, correlations, outliers, skewness) |
| v4 — Text Analysis | ✅ Done | Text column detection, word frequency, nested Claude call for sentiment/topics |
| v5 — Hardening | ✅ Done | Prompt caching, token metering, AST import sandbox, README update |

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [Anthropic Claude API](https://docs.anthropic.com/) — LLM with tool use + prompt caching
- [pandas](https://pandas.pydata.org/) + [plotly](https://plotly.com/python/) + [matplotlib](https://matplotlib.org/) — data analysis and charts
- [SQLAlchemy](https://www.sqlalchemy.org/) — SQL database connectivity
- [pytest](https://pytest.org/) — testing (121 tests)

## Security Note

The Python executor uses `exec()` on agent-generated code. Dangerous imports (`os`, `subprocess`, `socket`, etc.) are blocked at the AST level before any code runs. The current version is intended for **local development**. For production deployment, replace the executor with a fully isolated sandbox such as [E2B](https://e2b.dev/) or a Docker container with no network access, filesystem isolation, and CPU/memory limits.

## Dev Log

See [`dev-log.md`](dev-log.md) for a structured build journal tracking decisions, mistakes, fixes, and insights from each development session.

See [`PROJECT_REPORT.md`](PROJECT_REPORT.md) for a formal architectural write-up covering all engineering decisions, tradeoffs, and lessons learned — written for interview preparation.
