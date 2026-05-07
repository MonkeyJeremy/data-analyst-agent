# Dev Log — Data Analyst Agent

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
