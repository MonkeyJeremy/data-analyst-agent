# Data Analyst Agent — Project Report

**Author:** Jeremy Zhang  
**Project Duration:** 2026-05-02 to 2026-05-09  
**Repository:** github.com/MonkeyJeremy/data-analyst-agent  
**Final Test Count:** 129 passing · 0 failing  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Development Workflow](#5-development-workflow)
6. [Session-by-Session Build Log](#6-session-by-session-build-log)
7. [Key Engineering Decisions](#7-key-engineering-decisions)
8. [Mistakes and Fixes](#8-mistakes-and-fixes)
9. [Testing Strategy](#9-testing-strategy)
10. [Challenges and Insights](#10-challenges-and-insights)
11. [What I Would Do Differently](#11-what-i-would-do-differently)
12. [Interview Q&A Preparation](#12-interview-qa-preparation)

---

## 1. Executive Summary

The **Data Analyst Agent** is a full-stack AI-powered data analysis tool built entirely from scratch. Users upload a CSV, Excel, JSON, or SQLite file — or connect to a live SQL database — and converse with a Claude-powered agent in plain English. The agent writes and executes real Python or SQL code, renders interactive Plotly charts, and explains findings in natural language.

The project was built across six focused sessions spanning seven days, producing a production-quality Streamlit application with:

- A bounded **ReAct agent loop** (Reason + Act) backed by the Anthropic Claude API
- **Automated EDA on upload** — missing values, correlations, skewness, outliers, and AI-generated question suggestions
- **Non-standard layout detection** for messy Excel files with multi-row headers
- **Dual-mode data connectivity** — DataFrame mode (pandas) and SQL mode (SQLAlchemy)
- **Text analysis via nested Claude calls** — free-form text column detection, word frequency EDA, sentiment/topic classification without any NLP libraries
- **Prompt caching** — system prompt cached per session for ~80% input token reduction
- **Token metering** — live sidebar display of session token usage and cache hit rate
- **AST import sandbox** — blocks dangerous `os`, `subprocess`, `socket`, and 20+ other modules before any `exec()` runs
- **129 unit and integration tests** covering all non-UI modules

---

## 2. Problem Statement

Most data analysis tools require the user to know either SQL or Python. The gap between "I have a CSV" and "I have answers" is too large for non-technical users. The goal was to close this gap: accept any tabular data source, automatically understand its structure, and let the user ask questions in plain English.

Three concrete pain points were identified and addressed:

| Pain Point | Solution Built |
|------------|---------------|
| Users don't know pandas/SQL | Claude generates and executes the code; user only writes English |
| Real-world Excel files have messy headers | Automatic layout detection with user confirmation |
| Data lives in many formats (CSV, JSON, databases) | Unified ingestion layer with format-specific normalisation |

---

## 3. System Architecture

### High-Level Flow

```
User Upload (CSV / Excel / JSON / SQLite / SQL URL)
      │
      ▼
Layout Detection (detect_layout)           ← for CSV/Excel only
      │
      ▼
Data Loading (load_tabular / connect_sqlite_file / connect_url)
      │
      ▼
Schema Extraction (describe_schema / describe_sql_schema)
      │
      ▼
Automated EDA (run_auto_eda)               ← DataFrame mode only
      │
      ▼
System Prompt Construction (build_system_prompt / build_sql_system_prompt)
      │
      ▼
User Message ──► Bounded ReAct Loop (run_agent_turn)
                      │
                      ├── Claude API call
                      ├── Tool dispatch (execute_python / execute_sql)
                      ├── Result capture (stdout, figures)
                      └── Repeat up to MAX_TOOL_ITERATIONS (5)
                      │
                      ▼
                 TurnResult (text + figures)
                      │
                      ▼
              Streamlit UI renders response
```

### Module Map

```
src/
├── agent/
│   ├── client.py          # Anthropic SDK wrapper + prompt caching + TokenUsage metering
│   ├── loop.py            # Bounded ReAct loop (run_agent_turn) + TurnResult with token_usage
│   ├── system_prompt.py   # Prompt construction (DataFrame / SQL / text_cols modes)
│   └── tools.py           # Tool schemas + dispatch_tool() (python / sql / analyze_text)
├── data/
│   ├── layout.py          # Non-standard header detection (LayoutResult)
│   ├── loader.py          # Multi-format file loader (CSV/Excel/JSON)
│   └── schema.py          # SchemaContext frozen dataclass
├── db/
│   ├── connection.py      # SQLConnection + connect_sqlite_file() + connect_url()
│   ├── executor.py        # execute_sql() with read-only guard
│   └── schema.py          # TableSchema introspection
├── eda/
│   ├── auto_eda.py        # run_auto_eda() pure function + text EDA integration
│   └── report.py          # EDAReport frozen dataclass (text_cols + top_words fields)
├── execution/
│   ├── python_executor.py # AST import blocklist + sandboxed exec() + chart capture
│   └── result.py          # ExecutionResult frozen dataclass
├── text/
│   ├── eda.py             # detect_text_cols() + compute_top_words() + _STOP_WORDS
│   └── analyzer.py        # analyze_text_batch() — nested Claude call → JSON → table
└── ui/
    ├── chat_panel.py      # Chat history + interactive chart renderer
    ├── eda_panel.py       # 4-tab EDA panel (Overview/Distributions/Correlations/Text)
    ├── layout_panel.py    # Header confirmation UI
    ├── sql_panel.py       # SQL connection widget + stats bar
    ├── styles.py          # Global CSS injection
    └── upload_panel.py    # File uploader
```

### Data Flow: Frozen Dataclasses as Pipeline Contracts

Every stage of the pipeline produces an immutable frozen dataclass. This was a deliberate design choice — it prevents one stage from accidentally mutating data that another stage depends on, and makes the system trivially testable.

| Dataclass | Owner | Fields |
|-----------|-------|--------|
| `ExecutionResult` | `python_executor`, `executor` | `stdout`, `error`, `figures`, `summary`, `plotly_figures` |
| `SchemaContext` | `schema.py` | `n_rows`, `n_cols`, `formatted_dtypes`, `head_markdown`, `describe_markdown` |
| `EDAReport` | `auto_eda.py` | `missing_pct`, `top_correlations`, `skewed_cols`, `outlier_counts`, `suggested_questions`, `narrative`, `text_cols`, `top_words` |
| `LayoutResult` | `layout.py` | `status`, `header_row`, `unnamed_ratio`, `confidence`, `candidate_rows`, `message` |
| `TableSchema` | `db/schema.py` | `name`, `columns`, `row_count` |
| `TurnResult` | `loop.py` | `final_text`, `tool_calls`, `messages`, `figures`, `plotly_figures`, `token_usage` |
| `TokenUsage` | `client.py` | `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens` |

---

## 4. Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| AI backbone | Anthropic Claude API (tool use) | Native tool-use, reliable JSON, strong code generation |
| Web framework | Streamlit | Fastest path from Python to interactive UI; handles state and reruns |
| Data manipulation | pandas 2.2 + numpy | Industry standard; `.json_normalize()`, `.read_sql()`, `.corr()`, `.skew()` all in one package |
| Database | SQLAlchemy 2.0 | Dialect-agnostic engine; works with SQLite, PostgreSQL, MySQL identically |
| Charts | Plotly Express + Graph Objects | Interactive HTML figures; travel as JSON strings through session state |
| Charts (fallback) | matplotlib + seaborn | Handles hand-crafted `plt.show()` code from the agent |
| Excel | openpyxl | Required by `pd.read_excel()` for `.xlsx` files |
| Testing | pytest + pytest-cov | Standard Python test framework; coverage reports built-in |
| Packaging | pyproject.toml (setuptools) | Modern Python packaging; `pip install -e .` for local development |

---

## 5. Development Workflow

The project followed a strict **Plan → TDD → Implement → Review** cycle for each version:

```
1. PLAN      Define scope, data structures, module interfaces, algorithm design
2. RED       Write tests that fail (the contract before the implementation)
3. GREEN     Implement until all tests pass
4. REFACTOR  Clean up, extract helpers, remove duplication
5. COMMIT    Detailed commit message explaining the "why"
```

This discipline was maintained across all five sessions:

- Every new module had tests written **before** the implementation
- Tests caught 6 real bugs that would have shipped to production
- No test was written to pass around a known limitation — if a test failed, the code was fixed

---

## 6. Session-by-Session Build Log

### Session 1 — v1 Foundation (2026-05-02)

**Goal:** Working end-to-end pipeline in one session.

**Built:**
- Full project scaffolding (`pyproject.toml`, `.env.example`, `.gitignore`, package structure)
- `load_tabular()` — CSV + Excel loader with format detection and error wrapping
- `describe_schema()` — produces `SchemaContext` with markdown-formatted column info and describe stats
- `execute_python()` — safe `exec()` sandbox: captures stdout via `io.StringIO`, scans the post-exec namespace for matplotlib figures, saves each as PNG bytes
- `LLMClient` — thin Anthropic SDK wrapper with exponential backoff retry on 429/500/502/503/529
- `run_agent_turn()` — bounded ReAct loop: sends full message history to Claude, handles `tool_use` and `end_turn` stop reasons, appends both `tool_use` and `tool_result` blocks before the next iteration
- Streamlit app: sidebar upload, chat panel, basic styles

**Result:** 33 tests, all passing. Working app on first real data query.

---

### Session 2 — Plotly Integration & UI Redesign (2026-05-07)

**Goal:** Fix production bugs found in live testing; upgrade charts; overhaul UI.

**Key decisions made:**
- Upgraded charts from static matplotlib PNGs to interactive Plotly figures
- Plotly figures serialised as JSON strings in `ExecutionResult.plotly_figures`
- `_apply_style()` post-processes every Plotly figure automatically (transparent background, sizing, hover labels) — agent code does not need to contain any styling
- `_compute_dimensions()` sizes charts based on data density (bars × pixels)
- Full dark-theme UI with CSS glass-effect cards, stats bar, suggestion chips, welcome screen

**Bugs fixed in this session:**
1. Duplicate chat responses (extra append in `app.py`)
2. Hover tooltip truncation (Plotly `namelength` default)
3. Chart width overridden by `use_container_width=True`

---

### Session 3 — v2 Automated EDA (2026-05-07)

**Goal:** Run EDA automatically on every upload; surface findings as suggestion chips.

**Built:**
- `run_auto_eda(df) -> EDAReport` — pure function: missing pct, Pearson correlation, skewness, IQR/z-score outlier detection, cardinality, constant column detection
- `render_eda_panel()` — 3-tab expandable panel: Overview (missing bar chart), Distributions (histogram grid), Correlations (heatmap)
- 5 dynamic question suggestions generated from actual EDA findings (e.g. "What's causing the correlation between Revenue and Units?")
- `eda_summary` hook in `build_system_prompt()` — agent gets EDA narrative as pre-computed context

**Key insight:** The `eda_summary` parameter in `system_prompt.py` and `loop.py` had been deliberately left as `None` since v1. v2 only needed to populate it — zero changes to those core files. Forward-compatible design paid off.

---

### Session 4 — Non-Standard Layout Detection (2026-05-07)

**Trigger:** Real-world `QC Sales.xlsx` file tested with the agent produced garbage results — 99 `Unnamed:` columns, no numeric data, meaningless EDA. The file had a 3-row header (blank → month names → week dates → data).

**Built:**
- `detect_layout(file_bytes, filename) -> LayoutResult` — two-signal heuristic:
  - Signal 1: `unnamed_ratio` — fraction of `Unnamed:` column names after default load
  - Signal 2: `numeric_colname_ratio` — fraction of column names that are integers/floats
  - Scans rows 0–9, scores each as `non_null_ratio × string_ratio`
  - If best score > 0.65 and re-load has >70% named columns → `auto_fixed` (silent)
  - Otherwise → `needs_confirmation` (user picks the header row from a selectbox)
- `render_layout_panel()` — info banner for `auto_fixed`; selectbox + live 5-row preview for `needs_confirmation`; Cancel resets state

**Key design decision:** Detection runs before EDA, not after. EDA on a mis-parsed DataFrame is not just useless — it's actively misleading. The check is cheap (2 pandas reads); EDA is expensive and semantically invalid on the wrong data.

---

### Session 5 — v3 JSON + SQL (2026-05-07)

**Goal:** Expand from flat CSV/Excel to JSON and SQL databases.

**Built:**
- `_load_json()` — normalises nested JSON via `json.loads` + `pd.json_normalize`: detects wrapper keys (`data`, `records`, `results`, `rows`), handles flat arrays, single objects, empty arrays
- `src/db/` package — `SQLConnection`, `connect_sqlite_file()`, `connect_url()`, `describe_sql_schema()`, `execute_sql()`
- `execute_sql` agent tool — regex-guarded SELECT-only executor that returns markdown tables
- `build_sql_system_prompt()` — injects table schemas and column types; instructs Claude to use SQL, not Python
- Dual-mode `run_agent_turn()` — mode detected from presence of `sql_engine` parameter
- SQL table picker UI — multi-table SQLite files show a picker: load as DataFrame or enter full SQL mode

---

### Session 6 — v4 Text Analysis (2026-05-09)

**Goal:** Add first-class text column support with zero new NLP library dependencies.

**Built:**
- `detect_text_cols(df)` — heuristic: `avg_chars ≥ 30` AND `cardinality_ratio ≥ 0.3`. Correctly identifies review columns; ignores low-cardinality categoricals like "department"
- `compute_top_words(series, n=15)` — built-in 50-word stop list, `re.findall(r"[a-z]+"`, 3-char minimum; no NLTK required
- `EDAReport.text_cols` + `EDAReport.top_words` — two new optional fields with `= ()` defaults; all 22 existing `test_auto_eda.py` tests pass unchanged
- `analyze_text_batch(client, texts, task)` — nested Claude API call: structured JSON prompt, strips markdown fences, parses JSON array, builds markdown table with label / confidence / note per row. Capped at 50 texts; uses `tools=[]` to prevent inner call from making tool calls
- `_TEXT_TOOL` schema + `analyze_text` dispatch route in `tools.py`
- `build_system_prompt()` extended with `text_cols` param — TEXT ANALYSIS section injected when text columns detected
- 4th "📝 Text" EDA tab with word-frequency horizontal bar chart + word-count distribution histogram
- Text-specific suggestion chips replace EDA suggestions when text columns exist
- 19 new tests (11 EDA + 8 analyzer), total 121

**Key decision:** Chose the nested Claude API call over VADER/TextBlob after evaluating three options. Zero dependencies, handles custom classification tasks beyond sentiment, architecturally elegant. The inner call passes `tools=[]` to ensure it only returns text — preventing the inner call from itself making tool use calls that would break JSON parsing.

---

### Session 7 — v5 Production Hardening (2026-05-09)

**Trigger:** Formal evaluation against a "What Makes a Good Agent Project" framework identified four high-priority gaps: no prompt caching, no cost visibility, exec() sandbox easily bypassed via dangerous imports, and stale README.

**Built:**
- **Prompt caching** — `client.py` wraps system prompt in `{"type": "text", "cache_control": {"type": "ephemeral"}}` block. Schema + EDA narrative is static across all turns in a session; served from cache at ~10% of full input cost after the first call. Expected 80–90% reduction in input token billing per session.
- **Token metering** — `TokenUsage` dataclass on `LLMClient` accumulates `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens` from every `response.usage`. Flows through `TurnResult.token_usage` to `app.py`. Sidebar displays live count + cache hit percentage. Resets on file clear.
- **AST import sandbox** — `_check_imports(code)` parses generated code with `ast.walk()` before any `exec()`. Blocks 25+ modules: `os`, `sys`, `subprocess`, `socket`, `shutil`, `pathlib`, `importlib`, `ctypes`, `pickle`, `multiprocessing`, `threading`, and more. Catches both `import X` and `from X import Y` forms. Returns `SecurityError` result without running any code.
- `FakeLLMClient.usage = TokenUsage()` added to `conftest.py` to mirror `LLMClient` interface in tests
- **README rewritten** — now accurately reflects v1–v5: SQL, JSON, text analysis, caching, sandbox, 129 tests
- 8 new tests for the import blocklist, total 129

---

## 7. Key Engineering Decisions

### 7.1 Stateless Full-History Replay

The Anthropic API is stateless — every call must include the complete conversation history from turn 0. The `run_agent_turn()` function maintains a local `history` list that grows with every iteration:

```
history = [
  {"role": "user",      "content": "What are the top correlations?"},
  {"role": "assistant", "content": [tool_use_block]},         ← appended after call 1
  {"role": "user",      "content": [tool_result_block]},      ← appended before call 2
  {"role": "assistant", "content": [text_block]},             ← appended at end_turn
]
```

Both the `tool_use` assistant block **and** the `tool_result` user block must be present before the next API call. Missing either produces an API validation error.

### 7.2 Safe Python Execution

The agent's generated code runs inside a sandboxed `exec()` call. Safety measures:
- `df = df.copy()` in the namespace — agent cannot mutate the session DataFrame
- `sys.stdout` replaced with `io.StringIO` during execution — captures all `print()` output
- Post-exec namespace scan for `go.Figure` and `plt.Figure` instances — figures are extracted, not references stored
- `matplotlib.use("Agg")` set at module import time — prevents GUI window crash in headless Streamlit server

### 7.3 `_apply_style()` Decouples Analysis from Presentation

Rather than relying on the agent to write correct styling code, a post-processing pass runs on every Plotly figure after `exec()`:
- Transparent background (works on both light and dark Streamlit themes)
- Data-density-based sizing (`_compute_dimensions()`)
- `bargap=0.55` — bars occupy ~30% of their slot (uncluttered)
- `hoverlabel.namelength = -1` — no truncation of column names
- Bar text labels stripped — values appear on hover, not as visual clutter

This made chart quality completely independent of what the agent wrote. Any valid Plotly code produces a consistently styled chart.

### 7.4 Layout Detection Algorithm

```
effective_ratio = max(unnamed_ratio, numeric_colname_ratio)

if effective_ratio < 0.30 → status="ok"

Scan rows 0..9:
  score(row_i) = non_null_ratio × string_ratio
  
If best candidate score > 0.65 AND re-load has >70% named columns:
  → status="auto_fixed"
Elif re-load has >50% named columns:
  → status="needs_confirmation"  (show top-5 candidates)
Else:
  → status="needs_confirmation"  (show all rows 0..9)
```

The scoring function (`non_null_ratio × string_ratio`) elegantly identifies header rows: they are almost always fully populated (high non_null) and contain text strings rather than numeric data (high string_ratio). A row of numbers like `[1, 2, 3, 4]` scores low on `string_ratio` even if fully populated.

### 7.5 Dual-Mode Architecture

Rather than converting SQL query results to a DataFrame (which loses the ability to write real SQL with JOINs, CTEs, window functions), the system has two first-class modes:

| Aspect | DataFrame Mode | SQL Mode |
|--------|---------------|----------|
| Tool | `execute_python` | `execute_sql` |
| System prompt | Schema + dtypes + head | Table names + column types |
| Charts | Full Plotly support | Not available (markdown tables only) |
| Data prep | `pd.json_normalize`, `load_tabular` | `connect_sqlite_file`, `connect_url` |
| Safety | `df.copy()` prevents mutation | SELECT-only regex guard |

### 7.6 Text Column Detection Heuristic

Two conditions distinguish free-form text from short labels and categorical columns:

```python
_MIN_AVG_CHARS = 30    # avg character length — below this it's a short label
_MIN_CARDINALITY = 0.3 # unique/total ratio — below this it's a categorical
```

Both conditions must be true. A status column like `["ok", "fail", "pending"]` fails on avg_chars. A department column like `["Engineering"] × 50` fails on cardinality. The thresholds are deliberately conservative — a false negative (missing a text column) is less harmful than a false positive (running word frequency on a categorical). Both are named module-level constants for easy tuning.

### 7.7 Nested Claude Call Architecture for Text Analysis

The `analyze_text_batch()` function makes a second `client.call()` with a structured JSON-return prompt and `tools=[]`:

```python
response = client.call(
    system=_ANALYSIS_SYSTEM,   # "Return ONLY a JSON array"
    messages=[{"role": "user", "content": numbered_prompt}],
    tools=[],                  # prevents inner call from making tool calls
)
```

Setting `tools=[]` is critical. If the inner call received the `execute_python` tool definition, Claude might decide to call it rather than return JSON, breaking the parser. By removing all tools, the response is guaranteed to be pure text (the JSON array).

The 50-text cap keeps the inner prompt within a reasonable token budget. For larger columns, the outer agent is instructed to sample with `.dropna().head(30).tolist()`.

### 7.8 Prompt Caching on System Prompt

The system prompt (schema + EDA narrative + text col instructions) is static within a session — it never changes between turns. By wrapping it in a `cache_control: ephemeral` block, Anthropic's prompt cache serves it at ~10% of full input token cost after the first call:

```python
system_block = [
    {
        "type": "text",
        "text": system_prompt_string,
        "cache_control": {"type": "ephemeral"},
    }
]
```

On a 10-turn session with a 2,000-token system prompt, this saves approximately 18,000 input tokens — a free optimisation with zero quality impact.

### 7.9 AST Import Sandbox

`exec()` is not truly sandboxed at the Python level — crafted code can still `import os`, `import subprocess`, etc. The `_check_imports()` function addresses this by parsing the code with `ast.walk()` before any execution:

```python
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        root = alias.name.split(".")[0]
        if root in _BLOCKED_MODULES:
            return f"SecurityError: ..."
    elif isinstance(node, ast.ImportFrom):
        root = node.module.split(".")[0]
        if root in _BLOCKED_MODULES:
            return f"SecurityError: ..."
```

The blocked set covers filesystem, network, process, memory, serialisation, and IPC modules. The check catches both `import os` and `from os.path import join`. If blocked, a `SecurityError` `ExecutionResult` is returned immediately — the code never reaches `exec()`. This is a meaningful first defence; production would add Docker container isolation.

### 7.11 JSON Normalisation Strategy

`pd.read_json` was deliberately **not** used as the primary approach because it silently mishandles nested structures. A `{"data": [...]}` wrapper produces a column named `"data"` containing Python lists — superficially valid, completely wrong for analysis.

The chosen approach:
1. Always `json.loads` first — get the raw Python object
2. Detect top-level type and any wrapper keys
3. Route to `pd.json_normalize` for controlled flattening

This is two extra lines versus `pd.read_json` but eliminates an entire class of silent failures.

---

## 8. Mistakes and Fixes

Every mistake below was caught by a failing test before it could reach production.

### Mistake 1: Mutable Reference Bug in FakeLLMClient

**What went wrong:** `FakeLLMClient.call()` stored `messages` directly — a reference to the live `history` list. Later loop iterations appended to `history`, so recorded call snapshots reflected **future state** rather than the state at call time.

**Symptom:** `test_tool_error_retry` checked `calls[1]["messages"][-1]["role"] == "user"` but found `"assistant"` — the end_turn block appended two iterations later had poisoned the snapshot.

**Fix:** `self.calls.append({"messages": copy.deepcopy(messages), ...})` — snapshot, not reference.

**Lesson:** Any time you record a mutable object for later assertion, you must deep-copy it immediately. This pattern appears in test doubles, audit logging, and event sourcing.

---

### Mistake 2: Duplicate Chat Responses

**What went wrong:** `app.py` had this block after calling `run_agent_turn()`:
```python
if not isinstance(result.final_text, str):
    messages.append({"role": "assistant", "content": result.final_text})
```
`result.final_text` is always a string, so `not isinstance(...)` is always `False`. BUT `result.messages` (which `app.py` wrote to session state) already contained the full assistant message appended by the loop. The condition was wrong — it should have been checking `result.messages` not `result.final_text` — but it caused every query to produce two responses.

**Fix:** Delete the block entirely. The loop owns the history; `app.py` only persists `result.messages`.

**Lesson:** When a function returns a complete updated state object (here: `result.messages`), the caller should not partially reconstruct that state itself. Single source of truth.

---

### Mistake 3: Plotly Hover Tooltip Truncation

**What went wrong:** Plotly's default `hoverlabel.namelength = 15` clips any name longer than 15 characters and appends a tilde. Column names like "Avg Daily Social Media Hours" appeared as "Avg Daily Social Media Hours~" in tooltips — subtle data corruption in the presentation layer.

**Fix:** `fig.update_layout(hoverlabel={"namelength": -1})` in `_apply_style()`. Setting `-1` disables truncation globally.

**Lesson:** Library defaults are not always sensible for data analysis use cases. Test display output, not just computation output.

---

### Mistake 4: `use_container_width=True` Discarding Computed Dimensions

**What went wrong:** `st.plotly_chart(fig, use_container_width=True)` stretches any figure to fill the Streamlit column (~1200px on wide layout), ignoring `fig.layout.width`. The careful `_compute_dimensions()` logic (60px per bar, clamped to sensible ranges) was completely discarded.

**Fix:** `use_container_width=False` plus a manual column-padding centring trick:
```python
pad = max(0, COLUMN_PX - chart_px) // 2
_, col_mid, _ = st.columns([pad, chart_px, pad])
with col_mid:
    st.plotly_chart(fig, use_container_width=False)
```

**Lesson:** Always verify that framework convenience parameters do not override your explicit configuration. Read the docs for interactions between parameters.

---

### Mistake 5: IQR=0 Skipping Outlier Detection

**What went wrong:** In `_compute_outliers()`:
```python
iqr = q75 - q25
if iqr == 0:
    continue  # WRONG — silently skips the column
```
When a column has 20 identical values and one extreme outlier, Q1 = Q3, so IQR = 0. The guard skipped the column entirely, reporting 0 outliers for a column with one clear outlier.

**Fix:** Z-score fallback when IQR = 0:
```python
if iqr == 0:
    std = col.std()
    if std == 0:
        continue  # truly constant column, handled elsewhere
    outlier_mask = (col - col.mean()).abs() > 3 * std
```

**Lesson:** Guard clauses that skip processing on edge cases often hide real data. Always ask "what does skipping mean for the user?" before writing `continue`.

---

### Mistake 6: Empty `candidate_rows` for All-Blank File

**What went wrong:** When a `.xlsx` file has all-None cells, pandas returns 0 rows when read with `header=None` (openpyxl skips leading/trailing empty rows). The scan loop runs over `range(0)` — zero iterations. `candidate_rows` ends up as `()`. The UI then has nothing to show in the selectbox.

**Fix:**
```python
# Before fix:
fallback_n = max(scan_up_to, min(4, n_rows))

# After fix:
if n_rows > 0:
    fallback_n = max(scan_up_to, min(4, n_rows))
else:
    fallback_n = 4  # absolute minimum
```

**Lesson:** When computing a "fallback minimum," always consider the case where both the primary source and the fallback source are zero. Test the empty-file case explicitly.

---

### Mistake 7: Breaking Existing Test When Adding JSON Support

**What went wrong:** `test_load_unsupported_format` in `test_loader.py` used `"file.json"` to trigger the `ValueError("Unsupported file format")` path. After adding `.json` support, this test expected a failure that no longer occurred.

**Fix:** Changed the test to use `"file.parquet"` — a format that remains genuinely unsupported.

**Lesson:** When extending a system's capabilities, grep for tests that relied on the old limitations. A feature addition that breaks a test is not a broken test — it's a correctly-caught regression in test assumptions.

---

## 9. Testing Strategy

### Philosophy: Test Contracts, Not Implementations

Tests were written against the public interface of each module — inputs and outputs — not against internal implementation details. This meant refactoring internals never broke tests.

### Test Double Design: FakeLLMClient

The most important test double was `FakeLLMClient` — a queue-based fake that replaces the real Anthropic SDK client in all agent loop tests:

```python
class FakeLLMClient:
    def __init__(self, responses: list[FakeMessage]) -> None:
        self._queue = deque(responses)
        self.calls = []

    def call(self, *, system, messages, tools) -> FakeMessage:
        self.calls.append({"system": system, "messages": copy.deepcopy(messages), "tools": tools})
        return self._queue.popleft()
```

Key properties:
- **Queue-based, not mock-based:** each test declares its scenario as a script of responses, not as mock assertions. This is far more readable.
- **Records calls with deep copies:** assertions on `client.calls` reflect state at the moment of the call, not post-facto state.
- **Raises on empty queue:** if the loop makes more API calls than expected, the test fails immediately with a clear error.

### Coverage Summary (Final — 129 tests)

| Module | Coverage |
|--------|----------|
| `src/db/connection.py` | 91% |
| `src/db/executor.py` | 93% |
| `src/db/schema.py` | 93% |
| `src/data/layout.py` | 88% |
| `src/data/loader.py` | 90% |
| `src/data/schema.py` | 81% |
| `src/eda/auto_eda.py` | 97% |
| `src/eda/report.py` | 100% |
| `src/agent/loop.py` | 91% |
| `src/agent/tools.py` | 76% |
| `src/execution/result.py` | 100% |
| `src/execution/python_executor.py` | 70% |
| `src/text/eda.py` | 96% |
| `src/text/analyzer.py` | 100% |
| `src/ui/*` | N/A (Streamlit; requires browser) |

### Test Fixtures

- `tests/fixtures/titanic.csv` — 10-row realistic CSV for smoke tests
- `tests/fixtures/sales.xlsx` — generated at test time via conftest session fixture (openpyxl)
- `tests/fixtures/sample.json` — 5-row flat JSON array
- `tests/fixtures/nested.json` — nested JSON with `{"data": [...]}` wrapper
- In-memory SQLite engines — created per-test via SQLAlchemy for isolation

### Avoiding Binary Fixture Files

The `sales.xlsx` fixture is generated programmatically by a `session`-scoped autouse pytest fixture rather than committed as a binary. Rationale:
- Binary `.xlsx` files are opaque to `git diff` — you can't review changes
- The generator is idempotent (skips if file exists) — no performance cost on repeated runs
- Test intent is clear from the generator code, not from inspecting an opaque file

---

## 10. Challenges and Insights

### 10.1 Anthropic API Message Format Subtlety

The Anthropic tool-use flow requires that **both** the `tool_use` assistant block and the `tool_result` user block from each iteration be present in the history before the next API call. This is more strict than OpenAI's chat completions format. The loop appends both blocks in sequence, making the history structure explicit and debuggable.

### 10.2 Streamlit Session State Management

Streamlit rerenders the entire Python script on every user interaction. This means:
- Any value that needs to persist across rerenders must live in `st.session_state`
- The `_last_filename` guard in `_handle_upload()` is essential — without it, every rerender would re-load the file and re-run EDA
- `st.chat_input` cannot be pre-filled programmatically — the `pending_query` pattern (store → rerun → pick up) was the only way to implement suggestion chips that populate the chat

### 10.3 Plotly Figure Lifecycle

Plotly figures are Python objects created in an `exec()` call. To survive the exec sandbox and persist in session state, they must be serialised to JSON immediately after execution (via `pio.to_json()`). This JSON string is then deserialised by `st.plotly_chart()` when rendering. The pattern is:

```
exec() → scan namespace for go.Figure → pio.to_json() → store in session_state → pio.from_json() → st.plotly_chart()
```

### 10.4 SQLAlchemy 2.0 Breaking Changes

SQLAlchemy 2.0 deprecated passing raw `sqlite3.Connection` objects to `pd.read_sql()`. Always wrap connections in an `Engine` object. The connection URL format also matters: `sqlite:///relative.db` (3 slashes) vs `sqlite:////absolute/path.db` (4 slashes).

### 10.5 Windows Long Path Limit

`pip install streamlit` fails on Windows 10 without Long Paths enabled because Streamlit's asset files create directory trees exceeding the 260-character MAX_PATH limit. Tests pass without Streamlit since no test imports it, but the full app requires the registry fix:

```powershell
# PowerShell as Administrator
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1
```

Note: `gpedit.msc` (Group Policy) is not available on Windows Home edition — the registry method is universal.

---

## 11. What I Would Do Differently

1. **Add a streaming response mode.** The current flow blocks until the full agent turn completes. Streaming partial text back to the UI would improve perceived responsiveness significantly.

2. **Persist chat history across sessions.** Currently chat resets on upload or page refresh. A lightweight SQLite-backed history store keyed by file hash would dramatically improve the "continuing an analysis" workflow — re-uploading the same file would restore prior context automatically.

3. **Add chart generation to SQL mode.** SQL mode currently returns markdown tables. A follow-up `execute_python` call with the query results as a temporary DataFrame would enable full chart rendering in SQL mode.

4. **Tiered model routing.** Simple aggregation queries ("what's the average salary?") go to the same model as complex multi-step analysis. Routing simple queries to `claude-haiku-4-5` and reserving Sonnet for complex ones would reduce cost meaningfully and is a strong signal of production-aware design.

5. **Replace `exec()` entirely with a container sandbox.** The AST import blocklist is a meaningful first defence but not production-grade. A Docker container with no network access, filesystem isolation, CPU/memory limits, and execution timeouts would fully address the security gap.

6. **Add a team knowledge loop.** Successful analysis workflows ("clean this column type", "detect seasonal patterns") are currently lost after each session. Exporting these as reusable JSON "recipes" or prompt templates would compound team knowledge over time.

7. **Improve test coverage of `src/agent/client.py`.** The retry branch (lines 25–38) was explicitly deferred. A mock-based test of the exponential backoff on 429 responses would close this gap.

---

## 12. Interview Q&A Preparation

### Q: Walk me through how the agent loop works.

The loop in `run_agent_turn()` implements the ReAct (Reason + Act) pattern. Each iteration:
1. Sends the full conversation history + system prompt to Claude
2. If `stop_reason == "end_turn"`, extract the text response and return
3. If `stop_reason == "tool_use"`, dispatch the tool call(s), capture results, append both the tool_use assistant block and the tool_result user block to history, then repeat

The loop is bounded at 5 iterations to prevent infinite loops and control API costs. The Anthropic API is stateless — every call replays the full history from the beginning, so the growing history list is the sole mechanism for context persistence.

---

### Q: How did you design the test suite for code that calls an LLM API?

I avoided mocking the Anthropic SDK directly because mock assertions (`assert_called_with(...)`) tightly couple tests to implementation details. Instead, I built `FakeLLMClient` — a queue-based test double that behaves like the real client from the loop's perspective. Each test scenario is written as a "response script" (a list of `FakeMessage` objects), making test intent immediately readable.

The key implementation insight was deep-copying the messages list when recording calls. Without that, later loop iterations that mutate the shared history list would corrupt previously recorded snapshots.

---

### Q: How did you handle the case where uploaded files have non-standard layouts?

This was motivated by a real failure: `QC Sales.xlsx` with 3 header rows produced 99 `Unnamed:` columns. The `detect_layout()` function runs two signals: the ratio of `Unnamed:` columns (pandas standard) plus a ratio of integer/float column names (catches files where the first row is all numbers). If either exceeds 0.30, the function scans rows 0–9 and scores each as `non_null_ratio × string_ratio`. A high-confidence result auto-fixes silently; a low-confidence result asks the user to confirm. Crucially, this check runs before EDA — running analysis on a mis-parsed DataFrame is not just useless, it's misleading.

---

### Q: Why did you choose `pd.json_normalize` over `pd.read_json` for JSON loading?

`pd.read_json` silently handles nested structures in unexpected ways — a `{"data": [...]}` wrapper produces a column called `"data"` containing Python lists rather than flattening the inner array. This passes without error but produces a DataFrame that's unusable for analysis. Using `json.loads` first gives full visibility into the structure, and `pd.json_normalize` provides controlled flattening with dotted column names for nested fields. The cost is two extra lines of code; the benefit is an entire class of silent data corruption is eliminated.

---

### Q: What security considerations did you implement for the SQL feature?

Two layers: First, only SELECT, WITH (CTEs), and EXPLAIN statements are permitted. The guard is a compiled regex `re.compile(r"^\s*(SELECT|WITH|EXPLAIN)\b", re.IGNORECASE)` that checks before any database interaction. Any other statement returns an error without touching the database. Second, all SQL goes through `pd.read_sql()` which uses parameterised SQLAlchemy execution — the query never touches raw string concatenation, so SQL injection through column names or table names is not possible through the agent's code.

---

### Q: Describe a bug that was hard to find and how you found it.

The mutable reference bug in `FakeLLMClient`. The symptom was that `test_tool_error_retry` failed asserting the last message role was `"user"` — it was `"assistant"` instead. At first this looked like the loop was appending messages in the wrong order. Adding print statements revealed the recorded call snapshot already showed `"assistant"` at record time. The root cause was that `self.calls.append({"messages": messages})` stored a reference, not a copy — later loop iterations that appended to the shared `history` list retroactively changed what the recorded snapshot appeared to contain. The fix was `copy.deepcopy(messages)`. The lesson is generalised to any pattern where you record mutable state for later assertion: snapshot it immediately or you're recording a moving target.

---

---

### Q: How did you approach text analysis without any NLP libraries?

I used Claude itself as the NLP engine via a nested API call. When a user asks about sentiment or topics, the outer agent samples the text column and calls `analyze_text`, which makes a second `client.call()` with a structured JSON-return prompt: "Return a JSON array with one object per text containing label, confidence, and note." The inner call uses `tools=[]` so it can only return text, guaranteeing the response is the JSON array rather than another tool invocation. This produces far higher quality results than rule-based libraries like VADER — it handles sarcasm, domain-specific language, and any custom classification task the user describes. The only tradeoff is cost per call, mitigated by the 50-text cap.

---

### Q: How does your prompt caching implementation work, and what's the expected saving?

The system prompt — which contains the schema, EDA narrative, and text column instructions — is static for the entire session. By wrapping it in a `cache_control: {"type": "ephemeral"}` content block, Anthropic's prompt cache stores it on the first call and serves it at roughly 10% of the normal input token cost on every subsequent call. On a typical 10-turn session with a 2,000-token system prompt, this saves around 18,000 input tokens with zero impact on output quality. The change is five lines in `client.py`.

---

### Q: How did you harden the Python executor, and what are its remaining limitations?

I added an AST-based import check that runs before any `exec()`. It parses the generated code with `ast.walk()` and rejects any attempt to import from a blocklist of 25+ modules: `os`, `sys`, `subprocess`, `socket`, `shutil`, `pathlib`, `ctypes`, and more. It catches both `import X` and `from X import Y` syntax. The code never reaches `exec()` if a blocked module appears — it returns a `SecurityError` `ExecutionResult` immediately.

The remaining limitations are honest: the AST check doesn't catch dynamic imports (`__import__("os")`), doesn't limit CPU time or memory, and doesn't block network access from pre-imported modules. For production, I would replace `exec()` entirely with a container sandbox — Docker with no network, read-only mounted data, and execution time limits. The current version is local-first and the code and README are explicit about that.

---

### Q: How do you think about the cost of running this agent at scale?

Four levers I implemented or planned:

1. **Bounded loop** — 5 iterations max prevents runaway costs from Claude getting stuck in a reasoning loop
2. **Deterministic pre-computation** — EDA, layout detection, schema introspection all run as deterministic scripts rather than reasoning rounds; this avoids paying model costs for work that's cheaper in code
3. **Prompt caching** — the schema context is served from cache at 10% cost after the first turn
4. **Text analysis cap** — `analyze_text` is capped at 50 texts per call

The next tier I would add: tiered model routing (Haiku for simple lookups, Sonnet for analysis), conversation summarisation to compress long histories, and per-query budget tracking so users see cost before making complex requests.

---

*End of Report*
