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
