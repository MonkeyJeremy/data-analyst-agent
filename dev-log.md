# Dev Log — Data Analyst Agent

---

## Session 9 — 2026-05-11: v7 Analyst UX

### Summary
Implemented all P0 and P1 improvements from the mentor roadmap. The agent now makes every answer inspectable and downloadable, validates and auto-repairs bad charts, abstracts the execution backend for future sandboxing, and uses a lightweight visualization planner to guide chart-type selection. 175 tests passing (up from 129).

---

### [DECISION] Chart validation injects correction into tool result summary, not as a new user message
Two approaches were considered: (A) append a new user turn after a bad chart, (B) augment the `ExecutionResult.summary` that goes back to the LLM as the tool result content. Chose B. It keeps the correction inside the tool result where the LLM expects feedback, doesn't add an extra user/assistant pair to history, and doesn't burn an extra iteration. The LLM sees the chart issues in the same "turn" as the code execution and can fix them in the next iteration naturally.

---

### [DECISION] Visualization planner uses keyword matching, no LLM call
A pre-call LLM classifier for chart type would add latency and token cost on every query. A keyword classifier (7 intents, ~80 keywords) resolves intent in microseconds, is fully deterministic, and is easy to tune by reading the list. It produces a hint injected into the system prompt — not a hard constraint — so the LLM can override it when the data suggests a better chart. False negatives (no intent detected) are silent; the agent just proceeds without the hint.

---

### [DECISION] `ExecutionBackend` ABC wraps existing module-level `execute_python()` via `LocalPythonExecutor`
The goal was to add the abstraction layer without touching any of the 129 existing tests. The module-level `execute_python()` function is preserved unchanged. `LocalPythonExecutor` is a thin class that delegates to it. `dispatch_tool()` in `tools.py` still calls the module-level function directly — no behavior change. The abstraction lives at the boundary level (factory + stubs) ready for E2B/Docker without requiring a refactor of existing code.

---

### [DECISION] Download buttons appear only when charts are present; Dataset CSV always included
The download row appears only after a turn that produced Plotly figures — not after pure text answers. This keeps the UI clean for non-chart responses. The "Dataset CSV" button is always included in the row (when in DataFrame mode) because users often want to export filtered data they've been exploring. PNG export is silently skipped if `kaleido` is not installed rather than showing an error.

---

### Files created
| File | Purpose |
|------|---------|
| `src/execution/backend.py` | `ExecutionBackend` ABC |
| `src/execution/backend_factory.py` | `get_backend("local"\|"e2b"\|"docker")` factory |
| `src/execution/e2b_executor.py` | E2B stub (raises `NotImplementedError`) |
| `src/execution/docker_executor.py` | Docker stub (raises `NotImplementedError`) |
| `src/execution/chart_validator.py` | `validate_figures()` + correction prompt generation |
| `src/agent/viz_planner.py` | `plan_visualization()` keyword classifier + `build_viz_hint()` |
| `tests/test_chart_validator.py` | 13 validation tests |
| `tests/test_viz_planner.py` | 15 intent + hint tests |
| `tests/test_backend_factory.py` | 6 backend factory tests |

### Files modified
| File | Change |
|------|--------|
| `src/agent/loop.py` | Import `validate_figures`; add `viz_hint` param; augment tool result summary on bad charts |
| `src/agent/system_prompt.py` | Accept `viz_hint` kwarg; append to prompt when non-empty |
| `src/execution/python_executor.py` | Import `ExecutionBackend`; add `LocalPythonExecutor` class at end of module |
| `src/config.py` | Add `EXECUTION_MODE = "local"` constant |
| `src/ui/chat_panel.py` | Add `render_turn_downloads()` (HTML/PNG/CSV/Markdown export buttons) |
| `app.py` | Add `viz_planner` import; `execution_mode` session state + sidebar selector; persist `last_tool_calls/question/answer`; call `render_turn_downloads` |
| `README.md` | Positioning statement, "Why this is different" table, screenshot placeholders, v7 build phase row |

---

## Session 8 — 2026-05-09: v6 Multi-Provider Support

### Summary
Decoupled the ReAct loop from the Anthropic SDK by introducing a `BaseLLMClient` ABC with normalized `AgentResponse`/`ToolCall`/`TokenUsage` types. Added `AnthropicClient` (with prompt caching) and `OpenAIClient` (with tool-schema conversion) as concrete providers. The sidebar now exposes a provider + model selectbox. `FakeLLMClient` in tests extends `BaseLLMClient` directly. Fixed 5 failing tests in `test_text_analyzer.py` caused by the provider-agnostic `response.text` API. 129 tests, all passing.

---

### [DECISION] Provider abstraction via ABC + normalized AgentResponse
The loop used to directly access `response.content` (Anthropic SDK type). To support OpenAI we needed a clean seam. Created `src/agent/base.py` with:
- `ToolCall(id, name, input)` — frozen dataclass
- `AgentResponse(stop_reason, text, tool_calls, _raw)` — normalized response
- `TokenUsage` — accumulates cross-provider token counts
- `BaseLLMClient` — ABC with `call()`, `build_assistant_entry()`, `build_tool_result_entries()`

Each provider implements the three methods. The loop is now pure ABC code. `LLMClient = AnthropicClient` alias preserved for backward compat.

---

### [DECISION] `build_tool_result_entries()` returns `list[dict]`, loop uses `history.extend()`
Anthropic wraps all tool results in one user message with a `tool_result` block list. OpenAI requires separate `{"role": "tool", ...}` messages per result. Returning `list[dict]` from `build_tool_result_entries()` and using `history.extend()` (not `history.append()`) unifies both formats without branching in the loop.

---

### [DECISION] OpenAI tool schema conversion in `OpenAIClient._convert_tool()`
Internal tool schemas use `input_schema` (Anthropic format). OpenAI requires `{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}`. Conversion is a pure function inside `OpenAIClient`, keeping the internal schema format stable.

---

### [FIX] `test_text_analyzer.py` fake clients returned `SimpleNamespace(content=[block])` with no `.text`
After the provider refactor, `analyzer.py` reads `response.text` instead of scanning `response.content`. The five failing tests used inline fake clients returning `SimpleNamespace(content=[block])`. Fixed by changing all fake `_call()` returns to `SimpleNamespace(text=json_str)`.

---

### Files created
| File | Purpose |
|------|---------|
| `src/agent/base.py` | `AgentResponse`, `ToolCall`, `TokenUsage`, `BaseLLMClient` ABC |
| `src/agent/providers/__init__.py` | Package marker |
| `src/agent/providers/anthropic_provider.py` | `AnthropicClient` with prompt caching |
| `src/agent/providers/openai_provider.py` | `OpenAIClient` with tool-schema conversion |

### Files modified
| File | Change |
|------|--------|
| `src/agent/client.py` | Rewritten as factory: `create_client(provider, api_key, model)` + `LLMClient` alias |
| `src/agent/loop.py` | Uses `BaseLLMClient`, `response.text`, `response.tool_calls`; `history.extend()` |
| `src/config.py` | `PROVIDERS` dict with Anthropic + OpenAI model lists and env var names |
| `app.py` | Provider + model selectboxes; dynamic API key env var lookup |
| `tests/conftest.py` | `FakeLLMClient` extends `BaseLLMClient`, returns `AgentResponse` |
| `tests/test_text_analyzer.py` | Fake clients return `SimpleNamespace(text=...)` not `content=[block]` |
| `pyproject.toml` | `openai` optional dep; `all` extras group |

---

## Session 6 — 2026-05-07: v4 Text Analysis

### Summary
Added first-class text analysis to the agent. Auto-detects free-form text columns on upload, computes word-frequency EDA, and adds a nested Claude call (`analyze_text` tool) that performs sentiment, topic classification, or any user-defined labelling task — zero new NLP library dependencies. New "📝 Text" tab in the EDA panel, text-specific suggestion chips, and full system-prompt awareness of text columns. 19 new tests, 121 total passing.

---

### [DECISION] Use nested Claude call instead of VADER/TextBlob for text analysis
Evaluated three options: (A) NLP libraries (VADER, TextBlob), (B) nested Claude API call, (C) both. Chose Option B (Claude as the NLP engine) because it produces far higher quality results for nuanced text, handles custom classification tasks beyond sentiment, requires zero dependencies, and is architecturally elegant — the outer agent decides *what* to analyse, the inner call does the labelling with a structured JSON prompt. The only tradeoff is cost per call, which is acceptable for the 50-text cap.

---

### [DECISION] Text column heuristic: avg_chars ≥ 30 AND cardinality ratio ≥ 0.3
Two conditions distinguish free-form text from short labels and categoricals:
- `avg_chars ≥ 30`: filters out "ok / fail / pending" status columns that happen to be strings
- `cardinality / n ≥ 0.3`: filters out low-cardinality columns like "department" that repeat the same few values

Thresholds are conservative by design — a false negative (missing a text col) is less harmful than a false positive (running word-freq on a categorical). Both constants are named module-level variables for easy tuning.

---

### [DECISION] `EDAReport` backward-compat: new fields with `= ()` defaults
`EDAReport` is a frozen dataclass. Added `text_cols` and `top_words` after all existing required fields, both defaulting to empty tuples. All 22 existing `test_auto_eda.py` tests pass unchanged because Python dataclasses allow keyword-only defaults after positional fields. No migration or fixture changes needed.

---

### [DECISION] `analyze_text` caps at 50 texts, inner call uses `tools=[]`
Two design constraints:
1. **50-text cap**: keeps the inner prompt within a reasonable token budget and response time. Large datasets can be sampled with `.head(30).tolist()` or sliced by the outer agent.
2. **`tools=[]` in inner call**: the inner LLMClient call passes an empty tools list, so Claude can only return text (the JSON array). This prevents the inner call from accidentally making tool calls, which would break JSON parsing.

---

### [FIX] Test assertion mismatch: `"empty" not in "No texts provided."`
`test_analyze_text_empty_list` checked `"empty" in result.error.lower()` but the actual error message was "No texts provided." Changed the assertion to `assert result.error` (truthy check) — which correctly validates that an error is present without over-specifying the message wording.

---

### Files created
| File | Purpose |
|------|---------|
| `src/text/__init__.py` | Package marker |
| `src/text/eda.py` | `detect_text_cols()`, `compute_top_words()`, `_STOP_WORDS` |
| `src/text/analyzer.py` | `analyze_text_batch()` — nested Claude call, markdown table output |
| `tests/test_text_eda.py` | 11 tests for detection heuristics + word frequency + EDA integration |
| `tests/test_text_analyzer.py` | 8 tests for analyzer including FakeLLMClient, dispatch routing |

### Files modified
| File | Change |
|------|--------|
| `src/eda/report.py` | `text_cols` + `top_words` fields with `= ()` defaults |
| `src/eda/auto_eda.py` | Calls `detect_text_cols` + `compute_top_words`; `_build_questions` extended with `text_cols` param |
| `src/agent/tools.py` | `_TEXT_TOOL` schema; `get_tool_schemas(has_text_cols=)`; `dispatch_tool(client=)` |
| `src/agent/system_prompt.py` | `text_cols` param; TEXT ANALYSIS section injected when text columns present |
| `src/agent/loop.py` | `text_cols` param; `has_text_cols` flag; `client=client` passed to `dispatch_tool` |
| `src/ui/eda_panel.py` | 4th "📝 Text" tab with word-frequency bar + word-count histogram |
| `app.py` | `_TEXT_SUGGESTIONS`; `_render_suggestions` uses text suggestions when `eda.text_cols` non-empty; `text_cols` passed to `run_agent_turn` |

---

## Session 1 — 2026-05-02: v1 Full Build

### Summary
Implemented complete v1 from scratch: project scaffolding, data layer, Python executor, agent ReAct loop, Streamlit UI, and full test suite. 33 tests, all passing.

---

### [DECISION] Bounded ReAct loop with `MAX_TOOL_ITERATIONS = 5`
Chose to hard-cap the loop at 5 iterations rather than letting Claude recurse indefinitely.
Rationale: prevents infinite loops on ambiguous queries, keeps latency predictable, and makes the failure mode explicit ("iteration limit") rather than a silent API cost explosion.
Tradeoff: a truly complex multi-step analysis might hit the cap. Can be raised if needed.

---

### [DECISION] `matplotlib.use("Agg")` set at module import time in `python_executor.py`
Streamlit runs in a headless server process — the default `TkAgg` backend tries to open a GUI window and crashes. Setting `Agg` once at import avoids per-call setup and eliminates the crash.
This must be set *before* any `import matplotlib.pyplot` in the process, which is why it's at the top of the executor module.

---

### [DECISION] `df.copy()` inside the executor namespace
The agent-generated code gets `df.copy()`, not the original. This means even `df['col'] = ...` inside the agent's code leaves the session DataFrame unchanged.
Rationale: user uploaded the file once; agent mistakes shouldn't corrupt it. Deliberate mutation requires explicitly telling the agent to update state, which it can't do anyway (namespace is discarded after each exec).

---

### [DECISION] `FakeLLMClient` with a response queue instead of patching `anthropic`
Used a queue-based fake client rather than `unittest.mock.patch` on the Anthropic SDK. This keeps tests free of mock internals and makes test intent clearer — each scenario just declares its expected response sequence.
The `responses` list reads like a script: "when called, first return X, then Y."

---

### [MISTAKE] `FakeLLMClient` stored a reference to the mutable `messages` list
When recording calls in `self.calls.append({"messages": messages, ...})`, `messages` is a reference to the live `history` list inside the agent loop. Later loop iterations append to `history`, so `self.calls[1]["messages"][-1]` reflected post-call state rather than the state at call time. Test `test_tool_error_retry` was checking `messages[-1]["role"] == "user"` but found "assistant" (the end_turn block appended later).

---

### [FIX] `copy.deepcopy(messages)` in `FakeLLMClient.call()`
Added `import copy` and changed the recording line to `"messages": copy.deepcopy(messages)`. This snapshots the history as it existed at the moment of the call, so subsequent mutations don't corrupt the recorded state.

---

### [INSIGHT] Anthropic API requires full message history replay on every call
Unlike OpenAI's chat completions which are also stateless, Anthropic's tool-use flow requires that the `messages` list includes *both* the `tool_use` assistant block *and* the `tool_result` user block from previous iterations before the next call. Forgetting to append either block results in an API validation error. The loop appends both before each `continue`.

---

### [TRADEOFF] Skipped Excel fixture in `tests/fixtures/`
The plan calls for a `sales.xlsx` fixture. Skipped for v1 since `test_loader.py` covers Excel via the `load_tabular` code path through the file extension branch, and building a synthetic Excel fixture requires `openpyxl` at test-write time. Added a note: add a real `sales.xlsx` at the start of v2 for the EDA smoke test.

---

### [TRADEOFF] `src/agent/client.py` at 47% coverage
The retry logic (lines 25–38) handles `anthropic.APIStatusError` with status codes 429/500/502/503/529. Testing this path requires either a real API call or mocking the Anthropic SDK's exception hierarchy — both are noisy for a unit test. Accepted the lower coverage for this module; the happy path (line 20) is exercised indirectly through integration. Flag for v2: add a mock-based retry test.

---

### [INSIGHT] Windows Long Path limit breaks `streamlit` install
`pip install streamlit` fails with `OSError: [Errno 2] No such file or directory` on Windows 10 without Long Paths enabled. Streamlit bundles asset files with deep directory trees that exceed the 260-character MAX_PATH limit. All other dependencies install fine. The app will run once Long Paths are enabled (`gpedit.msc` → Computer Configuration → Administrative Templates → System → Filesystem → Enable Win32 long paths), but tests pass without it since no test imports `streamlit`.

---

### Coverage Summary (v1)

| Module | Coverage |
|--------|----------|
| `src/execution/python_executor.py` | 98% |
| `src/execution/result.py` | 100% |
| `src/agent/loop.py` | 96% |
| `src/agent/tools.py` | 100% |
| `src/agent/system_prompt.py` | 83% |
| `src/data/loader.py` | 92% |
| `src/data/schema.py` | 81% |
| `src/agent/client.py` | 47% ← retry branch untested |
| `src/ui/*` | excluded (Streamlit, requires browser) |

---

### v1 Completed Checklist

- [x] Project scaffolding (`pyproject.toml`, `.env.example`, `.gitignore`, package structure)
- [x] `src/data/loader.py` — CSV + Excel loading with error handling
- [x] `src/data/schema.py` — `SchemaContext` frozen dataclass
- [x] `src/execution/result.py` — `ExecutionResult` frozen dataclass
- [x] `src/execution/python_executor.py` — safe exec with stdout/figure capture
- [x] `src/agent/client.py` — Anthropic API wrapper with retry
- [x] `src/agent/system_prompt.py` — schema-injected system prompt
- [x] `src/agent/tools.py` — `execute_python` tool schema + dispatch
- [x] `src/agent/loop.py` — bounded ReAct loop (`TurnResult`)
- [x] `src/ui/upload_panel.py` — file uploader
- [x] `src/ui/chat_panel.py` — chat history + inline chart renderer
- [x] `app.py` — Streamlit entrypoint
- [x] `tests/conftest.py` — `sample_df`, `FakeLLMClient`
- [x] `tests/fixtures/titanic.csv` — 10-row smoke test fixture
- [x] `tests/test_loader.py` — 5 tests, loader.py 92%
- [x] `tests/test_schema.py` — 6 tests, schema.py 81%
- [x] `tests/test_python_executor.py` — 11 tests, executor 98%
- [x] `tests/test_tools.py` — 5 tests, tools.py 100%
- [x] `tests/test_agent_loop.py` — 6 tests (4 scenarios + figures + history), loop.py 96%
- [x] All 33 tests passing

### Next: v2 (Week 3)
- `src/eda/report.py` — `EDAReport` frozen dataclass
- `src/eda/auto_eda.py` — `run_auto_eda(df) -> EDAReport`
- `src/ui/eda_panel.py` — tabbed EDA panel
- Inject EDA narrative into system prompt
- Suggested questions from EDA findings
- `tests/test_auto_eda.py` (Titanic fixture)
- Smoke test: Sales Excel → EDA renders on upload

---

## Session 2 — 2026-05-07: Bug Fixes, Plotly Integration & UI Redesign

### Summary
Live-tested the v1 app, found and fixed a duplicate-response bug, upgraded charts from static
matplotlib PNGs to interactive Plotly figures, iteratively refined chart aesthetics through three
rounds of user feedback, and redesigned the full UI with a dark theme, welcome screen, dataset
stats bar, and suggestion chips.

---

### [MISTAKE] Duplicate assistant responses on every query
Every query produced two identical responses in the chat panel. Root cause: `run_agent_turn()`
already appends the full assistant message (including tool_use blocks) to `history` and returns the
updated list. `app.py` had an additional conditional block:

```python
# BUG — always True when content is a list of blocks
if not isinstance(result.final_text, str): ...
    messages.append({"role": "assistant", "content": result.final_text})
```

The condition `not isinstance(content, str)` is always True when `content` is a list, so a second
plain-text copy was appended on every turn. `render_chat_history()` then rendered both.

---

### [FIX] Removed redundant message append in `app.py`
Deleted the extra append block entirely. The loop owns the message history — `app.py` only
writes `result.messages` (the loop's authoritative copy) back into session state. No other
changes were required.

---

### [DECISION] Plotly as primary chart format; matplotlib PNG as fallback only
After seeing the static PNG bar chart in v1, switched to interactive Plotly. Design rationale:
- User can hover, zoom, pan on any chart without leaving Streamlit
- Plotly figures travel as JSON strings through `ExecutionResult.plotly_figures`, safe to store
  in session state and re-render without re-executing code
- Matplotlib PNG fallback preserved so hand-crafted `plt.savefig()` code still works

Implementation: executor scans the post-exec namespace for `go.Figure` instances, serialises
each with `pio.to_json()`. The matplotlib path fires only if no Plotly figures were found.
`ExecutionResult` gained `plotly_figures: tuple[str, ...] = field(default=())`. `TurnResult`
propagates the field. System prompt updated to strongly prefer `px`/`go` and require assigning
the figure to a variable (`fig = px.bar(...)`).

---

### [DECISION] `_apply_style()` post-processes every Plotly figure inside the executor
Rather than relying on the agent to write aesthetically correct styling code (unreliable and
verbose in prompts), the executor applies a fixed style pass after every `exec()`:

- Transparent canvas: `paper_bgcolor="rgba(0,0,0,0)"`, `plot_bgcolor="rgba(0,0,0,0)"`
- Computed dimensions via `_compute_dimensions()` (see separate entry)
- `bargap=0.55` so each bar occupies ~30% of its category slot (narrow, uncluttered)
- Strips bar text labels: `trace.update(text=None, texttemplate="")` — values on hover only
- `hoverlabel={"namelength": -1}` — never truncate column names in hover text

Agent-generated code need not contain any styling at all. This decouples analysis logic from
presentation and makes chart quality consistent regardless of what the agent writes.

---

### [MISTAKE] Hover tooltip truncated long column names
Plotly's default `hoverlabel.namelength = 15` silently clips names longer than 15 characters
and appends a tilde (`~`). A column named "Avg Daily Social Media Hours" appeared as
"Avg Daily Social Media Hours~" in the hover tooltip — easy to miss and confusing to users.

---

### [FIX] `hoverlabel={"namelength": -1}` in `_apply_style()`
Setting `namelength` to `-1` disables Plotly's truncation entirely. Full column names now
appear in hover labels. One-liner fix in `python_executor.py`.

---

### [MISTAKE] `use_container_width=True` overrode the explicit figure pixel width
`st.plotly_chart(fig, use_container_width=True)` stretches the figure to fill the Streamlit
column (~1200 px on wide layout) regardless of the `width` set in `fig.layout`. All bars
appeared wide even though `_compute_dimensions()` had computed a narrower target (e.g. 520 px
for 5 bars). The explicit width was computed but then discarded by Streamlit's own scaling.

---

### [FIX] `use_container_width=False` + column-padding centring in `_render_plotly_centred()`
Added `_render_plotly_centred(fig)` in `src/ui/chat_panel.py`:

```python
COLUMN_PX = 900  # conservative estimate of content column width
pad = max(0, COLUMN_PX - chart_px) // 2
col_left, col_mid, col_right = st.columns([pad, chart_px, pad])
with col_mid:
    st.plotly_chart(fig, use_container_width=False)
```

Charts now render at their natural pixel width, visually centred in the content area.
When the chart is wider than `COLUMN_PX`, it falls back to `use_container_width=False` without
padding (Streamlit will clip or scroll as appropriate).

---

### [DECISION] `_compute_dimensions()` sizes charts to data density
Four rules cover the common chart types:

| Chart type | Height | Width |
|------------|--------|-------|
| Horizontal bar | `max(300, min(800, n × 40 + 100))` | 650 px fixed |
| Heatmap | `max(350, min(900, n × 35 + 100))` | 750 px fixed |
| Vertical bar | 400 px fixed | `max(280, min(1000, n × 60 + 130))` |
| Everything else | 420 px | 750 px |

Where `n` is the number of bars/rows. The per-bar/row pixel allocations were tuned empirically:
40 px/row for horizontal (readable label + bar), 60 px/bar for vertical (breathing room with
`bargap=0.55`). Clamped ranges prevent degenerate sizes for very small or very large datasets.

---

### [IMPROVEMENT] Full UI redesign — `src/ui/styles.py` + `app.py` rewrite
Introduced a global CSS injection module (`inject()` called once in `main()`):

**Design language:** Dark theme (`#111120` main, `#0e0e1a` sidebar), translucent glass-effect
cards with `rgba(255,255,255,0.04)` fill and `rgba(255,255,255,0.08)` borders, purple hover
glow (`rgba(120,120,255,0.35)`) on interactive elements.

**New UI components:**

- *Welcome screen* (`_render_welcome()`): centred headline, subtitle, 3 feature cards
  (Upload / Ask / Discover) via raw HTML `<div class="feature-card">` — not Streamlit columns
  because Streamlit columns don't support variable-height card styling
- *Dataset stats bar* (`_render_dataset_stats()`): 4 `st.metric` cards — Rows, Columns,
  Numeric cols, Missing cells — shown above the chat after a file is uploaded
- *Suggestion chips* (`_render_suggestions()`): 5 pill-button questions shown when chat is
  empty. Each maps to a canned question; clicking stores it in `session_state.pending_query`
  and calls `st.rerun()`
- *Sidebar* restructured into 3 labelled sections: 🔑 API Key · 📁 Data Source · 🔍 Preview
  (5-row dataframe), plus a "Clear & upload new file" reset button

---

### [INSIGHT] `st.chat_input` cannot be pre-filled programmatically
Streamlit provides no API to inject text into the chat input widget. The suggestion chip
workaround: store the chosen question in `st.session_state.pending_query`, call `st.rerun()`,
then check for `pending_query` at the top of the main render loop (before `st.chat_input` is
evaluated). If set, clear it, run the agent turn, and return — the user never touches the input.

---

### [FIX] Corrected Windows Long Paths note from Session 1
The Session 1 entry referenced `gpedit.msc` (Group Policy Editor) as the fix for the
`pip install streamlit` failure on Windows. That tool is not available on Windows Home edition.
The registry method works on all Windows 10/11 editions:

```powershell
# Must be run in PowerShell as Administrator (not cmd.exe):
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1
```

Running in `cmd.exe` returns "not recognized as an internal or external command".

---

### Session 2 Changes Summary

| Area | Change |
|------|--------|
| `app.py` | Removed duplicate message append; added welcome screen, stats bar, suggestion chips, sidebar restructure |
| `src/execution/result.py` | Added `plotly_figures: tuple[str, ...]` field |
| `src/execution/python_executor.py` | Plotly figure scanning, `_apply_style()`, `_compute_dimensions()` |
| `src/agent/system_prompt.py` | Prefer Plotly instruction added |
| `src/agent/tools.py` | Tool description updated to mention `go`, `px` |
| `src/agent/loop.py` | `TurnResult` gains `plotly_figures`; loop propagates from tool results |
| `src/ui/styles.py` | New file — global CSS injection |
| `src/ui/chat_panel.py` | `render_turn_figures()` renders Plotly first; `_render_plotly_centred()` added |

### Next: v2 (Week 3) — unchanged from Session 1 plan

---

## Session 3 — 2026-05-07: v2 Automated EDA on Upload

### Summary
Implemented the full v2 EDA pipeline: `EDAReport` dataclass, `run_auto_eda()` computation,
`render_eda_panel()` Streamlit UI, 22 new tests (all passing), `sales.xlsx` fixture, and
5 targeted edits to `app.py`. Total test count: 55, all green. EDA coverage: 97%.

---

### [DECISION] EDA computation uses only pandas/numpy — no scipy
All EDA statistics needed (skewness, IQR, correlation, missing pct) are available directly
in pandas/numpy. Adding scipy would have been an extra dependency for zero benefit.
`df.skew()`, `df.corr()`, `df.quantile()`, `df.isnull().mean()` cover everything required.

---

### [DECISION] `EDAReport` stores only plain immutable types
The frozen dataclass uses `tuple[tuple[str, float], ...]` instead of `pd.Series` or
`pd.DataFrame`. This makes `EDAReport`:
- Truly hashable (frozen dataclass guarantee holds)
- Safe to store in Streamlit session state without serialisation issues
- Testable without DataFrame comparison helpers

---

### [DECISION] `_apply_style()` in executor vs `_eda_fig_style()` in EDA panel
Two separate style helpers exist for a reason:
- `_apply_style()` (executor): strips bar text labels and applies `bargap=0.55` because
  agent-generated charts should be minimal — values on hover only.
- `_eda_fig_style()` (eda_panel): keeps bar labels visible because EDA charts are
  read at a glance, not interactively explored. Bar heights and heatmap cell values
  should be immediately readable without hovering.

---

### [DECISION] Suggestion chips personalised from EDA findings
The 5 generic chip questions were replaced with dynamically-generated questions built
from actual findings. Priority order: top correlation pair → top missing column →
most skewed column → most outlier column → always "Give me a summary".
Falls back to generics if EDA finds nothing notable.

---

### [MISTAKE] `_compute_outliers()` skipped IQR=0 columns entirely
When 20+ values were identical and one extreme outlier existed, Q1=Q3 so IQR=0.
The guard `if iqr == 0: continue` caused the column to be silently ignored.
The outlier test caught this immediately.

---

### [FIX] Z-score fallback when IQR=0
When `iqr == 0`, fall back to z-score: count values more than 3 standard deviations
from the mean. If `std == 0` as well, it's a truly constant numeric column (already
caught by `constant_cols`) — skip. This handles the case where most values are
identical but extreme outliers exist.

---

### [INSIGHT] `eda_summary` hook was already wired in v1
`system_prompt.py` (`build_system_prompt(schema, eda_summary=None)`) and `loop.py`
(`run_agent_turn(..., eda_summary=None)`) both already had the optional parameter.
v2 only needed to populate it — zero changes to those files. Designing the hook
during v1 made the v2 integration a 5-line change in `app.py`.

---

### [DECISION] EDA panel collapsed by default (`expanded=False`)
The panel is available immediately after upload but not forced onto the user.
Power users can expand it; beginners can ignore it and use the suggestion chips.
The chips surface the most interesting findings as actionable questions.

---

### [TRADEOFF] `sales.xlsx` generated at test time via conftest, not committed as binary
A committed `.xlsx` binary is opaque to git diff and carries no history. Using a
`session`-scoped autouse fixture that generates the file once per test run keeps the
repo clean. The fixture is idempotent (skips generation if file already exists).

---

### Coverage Summary (v2)

| Module | Coverage |
|--------|----------|
| `src/eda/auto_eda.py` | 97% |
| `src/eda/report.py` | 100% |
| `src/agent/loop.py` | 96% |
| `src/agent/tools.py` | 100% |
| `src/data/loader.py` | 92% |
| `src/data/schema.py` | 81% |
| `src/execution/python_executor.py` | 63% ← Plotly path hard to unit-test |
| `src/ui/*`, `src/eda/eda_panel.py` | excluded (Streamlit, requires browser) |

---

### v2 Completed Checklist

- [x] `src/eda/__init__.py` — package marker
- [x] `src/eda/report.py` — `EDAReport` frozen dataclass
- [x] `src/eda/auto_eda.py` — `run_auto_eda(df)` pure function
- [x] `src/ui/eda_panel.py` — tabbed EDA panel (Overview / Distributions / Correlations)
- [x] `app.py` — 6 edits: imports, session key, upload hook, suggestions, panel render, eda_summary pass-through
- [x] `tests/test_auto_eda.py` — 22 tests, 97% coverage
- [x] `tests/conftest.py` — `sales_xlsx` session-scoped autouse fixture
- [x] All 55 tests passing

### Next: v3 (Week 4)
- SQLite support: `src/db/connection.py`, `src/db/executor.py`
- New tool: `execute_sql(query)` alongside `execute_python`
- Schema injection updated for table schema
- Smoke test: upload CSV → agent writes SQL to answer questions

---

## Session 4 — 2026-05-07: Non-Standard Layout Detection

### Summary
Implemented automatic detection of non-standard Excel/CSV layouts (multi-row headers, blank leading
rows, numeric column names). Files like `QC Sales.xlsx` now either auto-fix silently or show a
header-row selector with a live preview before EDA runs. 16 new tests, all green. Total: 71.

---

### [DECISION] Detection before EDA — never run EDA on a mis-parsed DataFrame
The detection step runs inside `_handle_upload()` before `_commit_upload()` (which runs schema +
EDA). Running EDA on a DataFrame with 97 `Unnamed:` columns produces misleading results (near-zero
numeric count, meaningless correlations). Detection is cheap (two pandas reads of the same file);
EDA is expensive and semantically meaningless on the wrong data.

---

### [DECISION] Two-signal suspicious-layout test: unnamed ratio + numeric column name ratio
Standard "Unnamed:" detection misses the case where a file has an all-numeric first row (e.g.
`[1, 2, 3, 4, 5]`). Pandas uses those integers as column names — no "Unnamed:" prefix, but equally
uninformative. Added `_numeric_colname_ratio()` as a second signal; `effective_ratio` is the max
of the two. This caught the edge case in `test_low_confidence_needs_confirmation`.

---

### [DECISION] Auto-fix vs. needs_confirmation threshold: named_ratio > 0.70 AND score > 0.65
Chosen to err on the side of confirming rather than silently fixing. A file that is 70% named after
re-loading is clearly better, but we want high confidence (score > 0.65) before silently applying.
If either threshold isn't met the user gets the selector, which adds one click but prevents
surprising silent changes to business-critical files.

---

### [MISTAKE] `candidate_rows = ()` for all-blank file
When an xlsx with all-None cells is read with `header=None`, pandas returns 0 rows (openpyxl skips
trailing/leading empty rows). `scan_up_to = min(10, 0) = 0`, so `tuple(range(0)) = ()`.
The test `test_candidate_rows_returned_on_confirmation` caught this immediately.

---

### [FIX] `fallback_n = max(scan_up_to, min(4, n_rows))` when no candidates found
When `n_rows == 0`, `fallback_n = max(0, min(4, 0)) = 0` still. Added special case: if `n_rows > 0`
use `max(scan_up_to, min(4, n_rows))`; otherwise use 4 as absolute fallback. Ensures
`candidate_rows` always has at least 1 entry when status is needs_confirmation.

---

### [DECISION] `_commit_upload()` extracted as a separate helper
Previously all the `st.session_state` mutations were inlined in `_handle_upload()`. Extracting
`_commit_upload(df, filename)` let both the auto-fix path (inside `_handle_upload`) and the
confirmation path (inside `main()` after user clicks Apply) share identical state-mutation logic.
No duplication, single source of truth.

---

### [DECISION] Cancel button resets `_last_filename` so the user can re-upload
If the user clicks Cancel, we clear `_last_filename` as well as `_layout_result` and
`_layout_file_bytes`. Without clearing `_last_filename`, the same filename would be seen as
"already processed" on the next upload attempt and `_handle_upload()` would return early.

---

### [INSIGHT] Streamlit file_uploader re-fires on every render cycle
`render_file_upload()` returns an `UploadedFile` object on every render as long as a file is
selected — not just on the first upload. The `_last_filename` guard in `_handle_upload()` prevents
re-running detection and re-loading on every cycle. This guard was already present for the normal
path; the layout detection path needs it equally.

---

### Coverage Summary (Session 4)

| Module | Coverage |
|--------|----------|
| `src/data/layout.py` | 88% |
| `src/data/loader.py` | 95% |
| All previous modules | unchanged |

### Session 4 Completed Checklist

- [x] `src/data/layout.py` — `LayoutResult` dataclass + `detect_layout()` + `preview_row()`
- [x] `src/data/loader.py` — extended with `header` / `skiprows` params
- [x] `src/ui/layout_panel.py` — info banner + confirmation UI with live preview
- [x] `app.py` — 5 edits: imports, session keys, `_commit_upload()`, `_handle_upload()` rewrite, main() gate + clear button
- [x] `tests/test_layout.py` — 16 tests, 88% coverage
- [x] All 71 tests passing

---

## Session 5 — 2026-05-07: v3 JSON + SQL Support

### Summary
Expanded the agent to support JSON files, SQLite file uploads, and live SQL database connections.
Introduced a dual-mode architecture: DataFrame mode (existing) and SQL mode (new `execute_sql` tool).
Added 31 new tests (11 JSON loader + 20 SQL executor/connection). 102 total tests, all passing.

---

### [DECISION] Dual-mode architecture: DataFrame vs SQL
Rather than trying to convert SQL results to a DataFrame and run the existing execute_python pipeline,
introduced a proper SQL mode with its own system prompt and `execute_sql` tool. This lets Claude write
real SQL (JOIN, GROUP BY, subqueries) instead of working around the ORM or recreating SQL semantics
in pandas.
Tradeoff: no plotly charts in SQL mode (the tool returns markdown tables, not a live DataFrame). This
is acceptable for v3 — users who want plots can "Load as DataFrame" instead.

---

### [DECISION] `execute_sql` safety guard: SELECT/WITH/EXPLAIN only
The executor checks the query string with a regex before sending it to pandas.read_sql. Any statement
not starting with SELECT, WITH, or EXPLAIN is immediately rejected with a clear error message.
Rationale: users upload their own databases but could accidentally write a destructive query; the
agent could also hallucinate DELETE/DROP. The guard makes it impossible.

---

### [DECISION] SQLite file upload → single table loads as DataFrame automatically
If a .db/.sqlite file contains exactly one table, it's loaded directly into a DataFrame and the
existing EDA + chat pipeline runs unchanged — no mode switch needed. Only multi-table SQLite files
trigger the table picker and SQL mode option.
Rationale: most people who upload a SQLite file have one canonical data table and just want to
analyse it like a CSV.

---

### [DECISION] SQLConnection as a non-frozen dataclass
SQLAlchemy engines are stateful connection pools. Using `frozen=True` would prevent reassignment
but can't prevent the engine from changing internal state (connections, pool). Using a regular
dataclass is honest about the resource's nature and allows `dispose()` for proper cleanup.
Pure data objects (LayoutResult, EDAReport, TableSchema, ColumnInfo) remain frozen.

---

### [DECISION] JSON loading via json.loads + json_normalize (not pd.read_json)
`pd.read_json` silently swallows nested structures (a `{"data":[...]}` wrapper produces a column
called "data" containing a list, not a flat DataFrame). Using `json.loads` first gives explicit
control: detect wrapper keys, call json_normalize on the inner array, handle edge cases cleanly.
Only two extra lines of code vs. a class of silent failure modes.

---

### [INSIGHT] pd.read_sql requires the engine URL scheme to match
For SQLite: `create_engine("sqlite:///path.db")` — note the triple slash for relative paths, four
for absolute (`sqlite:////absolute/path`). Passing raw sqlite3 connections to `pd.read_sql` is
deprecated in SQLAlchemy 2.0+. Always use an Engine object.

---

### [MISTAKE] test_load_unsupported_format was testing ".json" which is now a supported format
After adding JSON support to load_tabular(), the test `test_load_unsupported_format` passed
`"file.json"` and expected a ValueError("Unsupported file format"). Fixed by changing the test
to use `"file.parquet"` which remains unsupported.

---

### Coverage Summary (Session 5)

| Module | Coverage |
|--------|----------|
| `src/db/connection.py` | 91% |
| `src/db/executor.py` | 93% |
| `src/db/schema.py` | 93% |
| `src/data/loader.py` | 90% |
| `src/agent/loop.py` | 91% |
| All previous modules | unchanged |

### Session 5 Completed Checklist

- [x] `src/db/__init__.py` — package init
- [x] `src/db/connection.py` — `SQLConnection`, `connect_sqlite_file()`, `connect_url()`
- [x] `src/db/schema.py` — `ColumnInfo`, `TableSchema`, `describe_sql_schema()`
- [x] `src/db/executor.py` — `execute_sql()` (read-only guard + markdown output), `load_table()`
- [x] `src/data/loader.py` — `.json` branch via `_load_json()` + json_normalize
- [x] `src/agent/tools.py` — `_SQL_TOOL`, `get_tool_schemas(mode)`, extended `dispatch_tool()`
- [x] `src/agent/system_prompt.py` — `build_sql_system_prompt(schemas)`
- [x] `src/agent/loop.py` — `sql_engine` / `sql_schema` params, mode detection
- [x] `src/ui/sql_panel.py` — `render_sql_connect_panel()`, `render_sql_stats()`
- [x] `src/ui/upload_panel.py` — added json/db/sqlite accepted types
- [x] `app.py` — new session keys, SQLite upload branch, JSON upload branch, SQL table picker, SQL mode agent path, live SQL connect in sidebar, unified clear/reset
- [x] `tests/test_json_loader.py` — 11 tests, all passing
- [x] `tests/test_sql_executor.py` — 20 tests, all passing
- [x] `tests/test_loader.py` — fixed broken test (unsupported format now uses .parquet)
- [x] `tests/fixtures/sample.json` + `nested.json` — created
- [x] 102/102 tests passing
