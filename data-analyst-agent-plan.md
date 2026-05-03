# Data Analyst AI Agent — Full Implementation Plan

## Context

A data science student is building their first AI agent as a portfolio project. The goal is a **Conversational Data Analyst** — upload a CSV/Excel file, ask questions in natural language, and get answers backed by real Python/pandas execution and charts. Built in 3 phases so the student ships something demoable fast, then upgrades incrementally.

**Why this approach:** Claude's tool-use API lets the agent generate and execute Python code rather than guessing answers, making outputs trustworthy and demonstrable. Streamlit keeps the UI simple so the DS student can focus on the agent logic.

---

## Tech Stack

- **Language**: Python 3.11+
- **UI**: Streamlit
- **LLM**: Anthropic Claude API (tool use / function calling)
- **Analysis**: pandas, matplotlib, seaborn, numpy
- **DB (v3)**: SQLAlchemy + SQLite (then Postgres)
- **Testing**: pytest
- **Package manager**: uv or poetry

---

## Project Structure

```
data-analyst-agent/
├── README.md
├── pyproject.toml                  # deps: streamlit, anthropic, pandas, matplotlib, seaborn, openpyxl, python-dotenv, pytest, sqlalchemy
├── .env.example                    # ANTHROPIC_API_KEY=...
├── .gitignore
│
├── app.py                          # Streamlit entrypoint (thin UI layer only)
│
├── src/
│   ├── config.py                   # MODEL_NAME, MAX_TOKENS, MAX_TOOL_ITERATIONS=5
│   │
│   ├── agent/
│   │   ├── loop.py                 # run_agent_turn() — core ReAct loop
│   │   ├── client.py               # LLMClient wrapper (injectable for tests)
│   │   ├── system_prompt.py        # build_system_prompt(schema, eda_summary=None) -> str
│   │   └── tools.py                # TOOL_SCHEMAS list + dispatch_tool()
│   │
│   ├── execution/
│   │   ├── result.py               # ExecutionResult frozen dataclass
│   │   ├── python_executor.py      # execute_python(code, df) -> ExecutionResult
│   │   └── sql_executor.py         # v3: execute_sql(query, engine) -> ExecutionResult
│   │
│   ├── data/
│   │   ├── loader.py               # load_tabular(file, filename) -> pd.DataFrame
│   │   ├── schema.py               # describe_schema(df) -> SchemaContext
│   │   └── db.py                   # v3: connect_sqlite(), connect_postgres(), list_tables()
│   │
│   ├── eda/                        # v2 only
│   │   ├── auto_eda.py             # run_auto_eda(df) -> EDAReport
│   │   └── report.py               # EDAReport frozen dataclass
│   │
│   └── ui/
│       ├── upload_panel.py
│       ├── chat_panel.py
│       ├── eda_panel.py            # v2
│       └── db_panel.py             # v3
│
└── tests/
    ├── conftest.py                 # sample_df fixture, FakeLLMClient
    ├── test_python_executor.py
    ├── test_schema.py
    ├── test_tools.py
    ├── test_agent_loop.py
    ├── test_auto_eda.py            # v2
    ├── test_sql_executor.py        # v3
    └── fixtures/
        ├── titanic.csv
        └── sales.xlsx
```

---

## Core Architecture

The agent uses Anthropic's **tool-use API** in a bounded ReAct loop:

```
User uploads CSV
  → loader.load_tabular() → pd.DataFrame (session state)
  → schema.describe_schema() → SchemaContext (session state)

User asks question
  → append user message to history
  → run_agent_turn(client, messages, ctx)
      Loop (max 5 iterations):
        1. POST to Claude: system_prompt + messages + tool_schemas
        2. stop_reason == "end_turn"  → return final text
        3. stop_reason == "tool_use"  → for each tool_use block:
             result = dispatch_tool(name, input, ctx)
             append assistant msg (with tool_use blocks) to history
             append user msg (with tool_result blocks) to history
           continue loop
        4. max_iters hit → return sentinel error
  → update session_state.messages
  → st.rerun() to render new chat turn
```

**Critical detail:** Anthropic's API is stateless — you must replay the full message history (including all prior `tool_use` + `tool_result` blocks) on every call. The loop handles this correctly by appending messages before each iteration.

---

## Key Data Structures

```python
# src/execution/result.py
@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    error: str | None
    figures: tuple[bytes, ...]   # PNG bytes, one per matplotlib figure
    summary: str                 # short string sent back to Claude as tool_result

# src/data/schema.py
@dataclass(frozen=True)
class SchemaContext:
    n_rows: int
    n_cols: int
    formatted_dtypes: str        # column name + dtype as markdown table
    head_markdown: str           # df.head().to_markdown()
    describe_markdown: str       # df.describe().to_markdown()

# src/agent/loop.py
@dataclass(frozen=True)
class TurnResult:
    final_text: str
    tool_calls: tuple[ToolCallRecord, ...]
    messages: list               # updated full history
```

---

## Tool Definitions

### v1 — `execute_python`

```python
{
  "name": "execute_python",
  "description": "Execute Python code against the user's DataFrame (variable: `df`). pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns) are pre-imported. Use print() for text output; use plt.show() or leave a figure as last expression for charts. Do NOT re-read the file. Do NOT use input() or network calls.",
  "input_schema": {
    "type": "object",
    "properties": {
      "code":    {"type": "string", "description": "Valid Python code to execute."},
      "purpose": {"type": "string", "description": "One-sentence explanation shown to the user."}
    },
    "required": ["code", "purpose"]
  }
}
```

### v3 — `execute_sql`

```python
{
  "name": "execute_sql",
  "description": "Execute a read-only SELECT query against the connected database. Returns up to 1000 rows. For plotting or stats on results, chain with execute_python.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query":   {"type": "string", "description": "A SELECT statement only."},
      "purpose": {"type": "string", "description": "One-sentence explanation shown to the user."}
    },
    "required": ["query", "purpose"]
  }
}
```

Tool dispatch only registers `execute_sql` when `ctx.db_conn is not None`.

---

## System Prompt Design

```
You are a senior data analyst pair-programming with a user.

YOUR DATAFRAME:
- Variable: `df`
- Shape: {n_rows} rows × {n_cols} columns
- Columns and dtypes:
{formatted_dtypes}

FIRST 5 ROWS:
{head_markdown}

NUMERIC SUMMARY:
{describe_markdown}

INSTRUCTIONS:
1. For any computation, call execute_python. Never guess numbers.
2. After tool results return, interpret them in plain English.
3. For charts, use matplotlib or seaborn with titles and axis labels.
4. If a question is ambiguous, ask for clarification instead of guessing.
5. One analytical step per tool call.
6. If a tool returns an error, read the traceback, fix it, and retry (max 2 retries).
7. Never reference column names not listed above.
```

In v2, append: `PRE-COMPUTED EDA INSIGHTS:\n{eda_summary}` (truncate to 4k tokens).
In v3, also inject table schemas and tool-selection guidance.

---

## Python Executor Implementation

```python
# src/execution/python_executor.py
def execute_python(code: str, df: pd.DataFrame) -> ExecutionResult:
    namespace = {"df": df.copy(), "pd": pd, "np": np, "plt": plt, "sns": sns}
    stdout_buf = io.StringIO()
    figures: list[bytes] = []
    error: str | None = None

    plt.close("all")  # clear leftover figures from prior calls

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
            exec(compile(code, "<agent>", "exec"), namespace)
        for fig_num in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(fig_num).savefig(buf, format="png", bbox_inches="tight", dpi=100)
            figures.append(buf.getvalue())
    except Exception:
        error = traceback.format_exc(limit=5)
    finally:
        plt.close("all")

    return ExecutionResult(
        stdout=stdout_buf.getvalue(),
        error=error,
        figures=tuple(figures),
        summary=_build_summary(stdout_buf.getvalue(), figures, error),
    )
```

Key: `df.copy()` prevents agent code from mutating the user's DataFrame. `plt.close("all")` prevents memory leaks.

---

## Streamlit App Layout

```python
# app.py
st.set_page_config("Data Analyst Agent", layout="wide")
init_session_state()  # keys: df, schema, messages, eda_report, db_conn

with st.sidebar:
    api_key = st.text_input("Anthropic API Key", type="password")
    uploaded = render_file_upload()       # returns file or None
    if uploaded:
        handle_upload(uploaded)           # sets session_state.df + schema (+ runs EDA in v2)

# v2: EDA panel above chat
if st.session_state.eda_report:
    render_eda_report(st.session_state.eda_report)

if st.session_state.df is not None:
    render_chat_history(st.session_state.messages)  # shows text + charts inline

    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            result = run_agent_turn(get_client(api_key), st.session_state.messages, build_ctx())
        st.session_state.messages = result.messages
        st.rerun()
```

Charts are rendered via `st.image(fig_bytes)` inside `render_chat_history`. Tool calls shown in collapsed `st.expander("Code")`.

---

## Error Handling

| Scenario | Handling |
|---|---|
| Generated code crashes | `execute_python` catches exception, returns traceback in `error` field; Claude sees it and retries up to 2× |
| Agent hits max_iters | Return `"Reached iteration limit. Please rephrase."` as final text |
| Anthropic API error | Retry once with backoff in `client.py`; surface clear message on second failure |
| Bad file upload | Catch `ParserError`/`UnicodeDecodeError` in `loader.py`; show actionable message |
| Large file (>500MB) | Warn user, offer to sample to 100k rows |
| SQL mutation attempt | Regex guard in `sql_executor.py`; reject with clear error before query runs |

---

## Phase-by-Phase Build Plan

### v1 — Core (Week 1–2)

**Week 1: Data + execution layer (no LLM yet)**
- [ ] Set up project (`pyproject.toml`, `.env.example`, folder structure)
- [ ] `src/data/loader.py` — `load_tabular(file, filename) -> pd.DataFrame`
- [ ] `src/data/schema.py` — `describe_schema(df) -> SchemaContext`
- [ ] `src/execution/result.py` — `ExecutionResult` frozen dataclass
- [ ] `src/execution/python_executor.py` — `execute_python(code, df) -> ExecutionResult`
- [ ] Write `test_python_executor.py` and `test_schema.py`; hit 80% coverage on these modules
- [ ] Streamlit upload skeleton — upload file, show `df.head()` and `df.dtypes`

**Week 2: Agent loop + UI integration**
- [ ] `src/agent/client.py` — `LLMClient` wrapping `anthropic.Anthropic()`
- [ ] `src/agent/system_prompt.py` — `build_system_prompt(schema) -> str`
- [ ] `src/agent/tools.py` — `TOOL_SCHEMAS`, `dispatch_tool()`
- [ ] `src/agent/loop.py` — `run_agent_turn()` with ReAct loop
- [ ] Write `test_agent_loop.py` using `FakeLLMClient` (4 scenarios: direct answer, one tool call, tool error + retry, max_iters)
- [ ] Wire loop into `app.py` + `src/ui/chat_panel.py`
- [ ] Smoke test: Titanic CSV → "survival rate by sex" → correct table + correct numbers

**v1 is demo-ready after Week 2.**

---

### v2 — Automated EDA (Week 3)

- [ ] `src/eda/report.py` — `EDAReport` frozen dataclass (shape, nulls, distributions, correlation_matrix_png, outlier_table, narrative)
- [ ] `src/eda/auto_eda.py` — `run_auto_eda(df) -> EDAReport`
  - Shape + memory usage
  - Per-column null counts and unique counts
  - Numeric: `describe()`, histograms (sample to 100k rows if needed), IQR outlier counts
  - Categorical: top-10 value counts
  - Correlation heatmap (if ≥ 2 numeric columns)
- [ ] `src/ui/eda_panel.py` — tabbed panel: Overview / Distributions / Correlations / Outliers
- [ ] Inject EDA narrative into `build_system_prompt()` (add optional `eda_summary` param)
- [ ] Add "Suggested questions" from EDA findings (e.g., "Column `age` has 20% nulls — ask about imputation?")
- [ ] Write `test_auto_eda.py` — Titanic fixture, assert expected shape, non-zero figures, nulls detected
- [ ] Smoke test: Sales Excel → verify EDA panel renders before any user input

---

### v3 — SQL Support (Week 4)

- [ ] `src/data/db.py` — `connect_sqlite(path) -> Engine`, `connect_postgres(dsn) -> Engine`, `list_tables_with_schemas(engine) -> dict`
- [ ] `src/execution/sql_executor.py` — `execute_sql(query, engine, row_limit=1000) -> ExecutionResult`
  - Reject `INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE` via regex before execution
  - Use read-only connection mode where possible
  - Return first 1000 rows as markdown table in `stdout`
- [ ] Register `EXECUTE_SQL_TOOL` in `tools.py` only when `ctx.db_conn is not None`
- [ ] Update `build_system_prompt()` for DB context: table schemas + tool-selection guidance
- [ ] `src/ui/db_panel.py` — radio "File upload" / "Database", path/DSN input, "Test connection" button
- [ ] Bridge: pass last SQL result as `last_sql_df` in Python executor namespace for plotting
- [ ] Write `test_sql_executor.py` — in-memory SQLite: SELECT works, mutations rejected, row limit enforced
- [ ] Smoke test: Chinook SQLite → "top 5 artists by track count" → verify SQL tool used

---

## v1 → v2 → v3 Upgrade Summary

| Area | v1 | v2 adds | v3 adds |
|---|---|---|---|
| `src/eda/` | — | `auto_eda.py`, `report.py` | — |
| `src/data/` | `loader.py`, `schema.py` | — | `db.py` |
| `src/execution/` | `python_executor.py` | — | `sql_executor.py` |
| `src/agent/tools.py` | `execute_python` only | — | `execute_sql` (conditional) |
| `system_prompt.py` | schema only | + EDA summary | + table schemas + tool guidance |
| `app.py` | upload + chat | + EDA panel | + DB panel |

No existing code is deleted when upgrading — each phase is purely additive.

---

## Testing Strategy

### Coverage target: 80% on all `src/` modules except `src/ui/`

| Test file | What it covers |
|---|---|
| `test_python_executor.py` | stdout capture, figure capture, error capture, no df mutation |
| `test_schema.py` | SchemaContext fields populated correctly |
| `test_tools.py` | `dispatch_tool` routes correctly; unknown tool returns error |
| `test_agent_loop.py` | 4 scenarios with FakeLLMClient (direct, one-tool, retry, max_iters) |
| `test_auto_eda.py` | Titanic: correct shape, nulls found, figures generated |
| `test_sql_executor.py` | SELECT works, mutations blocked, row limit enforced |

CI: GitHub Actions running `pytest` on push. Add badge to README.

---

## Portfolio Polish (end of Week 4)

- [ ] README: architecture diagram, demo GIF, "what I learned" section
- [ ] 3-minute demo video recorded (Titanic + Chinook walkthrough)
- [ ] GitHub repo public, CI green
- [ ] `.env.example` committed (never `.env`)
- [ ] Document the `exec()` limitation: local-dev only; mention E2B as the production upgrade path

---

## Verification Checklist

- [ ] `streamlit run app.py` starts with one command
- [ ] Titanic CSV → "survival rate by sex" → correct numbers + chart
- [ ] Code crash → agent automatically retries and recovers
- [ ] EDA panel appears within 5s of upload for files < 100MB
- [ ] Chinook SQLite → SQL tool chosen for aggregations, Python for plotting
- [ ] `pytest --cov=src --cov-report=term-missing` shows ≥ 80% on execution + agent modules
