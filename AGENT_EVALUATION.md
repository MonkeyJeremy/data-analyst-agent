# Agent Project Evaluation
_Evaluated against: "What Makes a Good Agent Project" framework (Douyin article, 2026-05-09)_

---

## Overall Scorecard

| Dimension | Score | Summary |
|-----------|-------|---------|
| Cost Awareness | B+ | Bounded loop + compact prompts; missing caching & tiered routing |
| Security Controls | A- | Good sandbox & SQL guard; exec not truly air-gapped, no audit log |
| Knowledge Accumulation | C | Strong within-session; zero cross-session or team-level persistence |
| Skill Quality | A | Scripts over reasoning, deterministic, 102 tests |

---

## 1. Cost Awareness — B+

### Strengths
- Bounded ReAct loop (`MAX_TOOL_ITERATIONS = 5` in `src/config.py`) prevents runaway API costs
- System prompt is compact: EDA narrative capped at 1,500 chars, schema as tight markdown tables
- Uses `claude-sonnet-4-6` (not Opus) — cost-appropriate model selection

### Gaps & Improvement Actions

**[ ] Add prompt caching**
- The system prompt (schema + EDA context) is identical across all turns in a session but re-sent every time
- Fix: use Anthropic cache-control headers on the system prompt block in `src/agent/client.py`
- Expected savings: 80–90% token reduction on repeated system prompt tokens

**[ ] Add token metering**
- Currently no visibility into cost per query
- Fix: log `usage.input_tokens` and `usage.output_tokens` from each API response in `src/agent/client.py`
- Extend to a running session total displayed in the Streamlit sidebar

**[ ] Implement tiered model routing**
- Simple aggregation queries ("what's the average sales?") go to the same model as complex multi-step analysis
- Fix: classify query complexity before the agent loop; route simple queries to `claude-haiku-4-5`, reserve Sonnet for complex ones
- This is explicitly called out in enterprise interviews as a production-thinking differentiator

---

## 2. Security Controls — A-

### Strengths
- Python sandbox: `exec()` runs in isolated namespace, `df.copy()` prevents DataFrame mutation, `matplotlib.use("Agg")` blocks GUI (`src/execution/python_executor.py`)
- SQL regex whitelist: blocks INSERT/UPDATE/DELETE/DROP; only SELECT/WITH/EXPLAIN permitted (`src/db/executor.py`)
- SQLAlchemy parameterized queries prevent SQL injection
- No hardcoded secrets; `.env`-based API key management
- Temp SQLite files cleaned up via `os.unlink()` on session clear (`src/db/connection.py`)

### Gaps & Improvement Actions

**[ ] Harden the exec sandbox**
- Current `exec()` is not truly air-gapped — a crafted prompt can still `import os`, `import subprocess`, etc.
- Fix options (in order of effort):
  1. Blocklist dangerous imports at parse time using `ast` module before `exec()`
  2. Use `RestrictedPython` library for a proper restricted execution environment
  3. Run code in a Docker container with no network access (production-grade)

**[ ] Add operation audit log**
- No record of what code was executed, by whom, when
- Fix: append each `execute_python` / `execute_sql` call to a structured log (timestamp, query, generated code, result summary)
- Store in a local SQLite log DB or append-only file

**[ ] Add permission isolation**
- All users currently have identical access
- Fix: introduce a simple role concept (read-only vs. full analysis) gating which tools are available

---

## 3. Knowledge Accumulation — C

This is the framework's most critical dimension for enterprise value, and the weakest area.

### Strengths
- EDA insights auto-computed on upload and persisted for the full session (`src/eda/auto_eda.py`)
- `PROJECT_REPORT.md` is excellent developer knowledge documentation
- Conversation history accumulates within a session for multi-turn coherence

### Gaps & Improvement Actions

**[ ] Cross-session memory**
- Every page refresh wipes all history and insights
- Fix: persist conversation history and EDA results to a local SQLite DB keyed by file hash
- On re-upload of the same file, restore prior context automatically

**[ ] Analysis knowledge base**
- Insights discovered in one analysis session ("Q1 sales always spike in week 2") are lost forever
- Fix: add a "Save insight" action that appends a structured note (dataset, column, finding, timestamp) to a persistent store
- Display saved insights in a sidebar panel for future sessions on the same dataset

**[ ] Team-level sharing**
- Currently a single-user tool with no knowledge sharing
- Fix: export/import insights as JSON; or add a shared insights endpoint if deployed as a multi-user app
- Even a simple "copy insight link" that encodes the finding as a URL parameter would demonstrate the thinking

---

## 4. Skill Quality — A

This is the project's strongest dimension and aligns directly with the article's best practices.

### Strengths
- **Scripts over reasoning**: the agent writes and executes real Python/SQL rather than answering from context — deterministic and verifiable
- **Frozen dataclasses as pipeline contracts** (`SchemaContext`, `EDAReport`, `ExecutionResult`, `TurnResult`) — low hallucination surface, immutable state
- **102 tests** across all non-UI modules with 81–100% per-module coverage
- `FakeLLMClient` test double enables fast, reliable unit tests without API calls
- Tool dispatch is deterministic: `dispatch_tool()` routes to the correct executor based on mode

### Minor Improvements

**[ ] Minimize reasoning rounds further**
- The `analyze_text` tool makes a nested Claude call — this is a reasoning round that could be replaced with a lighter local classifier (e.g., a fine-tuned BERT or zero-shot HuggingFace model) for common sentiment tasks
- Reserve the nested Claude call only for complex/custom classification tasks

**[ ] Index-level context exposure**
- Currently the full schema is always injected regardless of query relevance
- For wide tables (50+ columns), consider injecting only the columns most likely relevant to the query (semantic similarity pre-filter)

---

## Interview Framing Guide

### What to lead with (your strengths)
> "I bounded the agent loop to 5 iterations to control cost. The agent writes and executes real Python/SQL rather than hallucinating answers — scripts over reasoning. The execution sandbox isolates the namespace and copies the DataFrame to prevent mutation."

### What to proactively acknowledge (shows production thinking)
> "In production I'd add prompt caching on the schema context — it's static across all turns in a session, so it's a free 80% token reduction. I'd also add tiered model routing: simple aggregation queries to Haiku, complex multi-step analysis to Sonnet. And I'd add cross-session persistence so insights compound over time rather than resetting on every page load."

### The framework's key test
The article says good agent projects answer two questions from an enterprise perspective:
1. **What real problem does it solve?** → "Non-technical users can query any dataset in plain English and get executable, reproducible analysis — no SQL or Python required."
2. **How did you think about cost and security?** → See the two sections above.

---

## Prioritized Improvement Backlog

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| High | Prompt caching on system prompt | Low | Cost −80% on input tokens |
| High | Token metering in sidebar | Low | Visibility + interview talking point |
| High | Cross-session persistence (SQLite) | Medium | Knowledge accumulation |
| Medium | Tiered model routing (Haiku for simple queries) | Medium | Cost + differentiation |
| Medium | `ast`-based import blocklist in exec sandbox | Low | Security hardening |
| Medium | Operation audit log | Low | Security + observability |
| Low | Insight knowledge base (save/load findings) | Medium | Team value |
| Low | Column relevance pre-filter for wide tables | Medium | Context efficiency |
| Low | Replace nested Claude call in analyze_text | High | Skill quality + cost |
