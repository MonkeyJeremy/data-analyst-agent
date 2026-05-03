# Data Analyst Agent — Requirements & Notes

_Last updated: 2026-04-29_
_Add new requirements below as they come up._

---

## Background

Building a Conversational Data Analyst AI Agent as a portfolio project for data science job interviews.
Developer is a data science major, new to AI/LLM development.

---

## Core Requirements

### Functional
- Upload CSV or Excel files
- Ask natural language questions about the data
- Agent generates and executes real Python (pandas/matplotlib) code to answer — no guessing
- Results (text + charts) displayed inline in the UI
- Agent auto-retries on code errors (reads traceback, fixes, re-runs)

### Non-Functional
- One-command startup: `streamlit run app.py`
- Works fully offline after setup (except LLM API calls)
- Local-dev only for v1 (exec() sandbox; document this limitation)
- 80%+ test coverage on all non-UI modules

---

## Phase Requirements

### v1 — Conversational Data Analyst (Weeks 1–2)
- File upload (CSV / Excel)
- Schema injection into system prompt (columns, dtypes, head, describe)
- `execute_python` tool: runs pandas/matplotlib code against uploaded df
- Streamlit chat UI with inline chart rendering
- Tool calls shown in collapsed code expander

### v2 — Automated EDA (Week 3)
- On upload, automatically run EDA before user asks anything
- EDA covers: shape, nulls, distributions, correlations, outliers
- EDA panel rendered above chat (tabbed: Overview / Distributions / Correlations / Outliers)
- EDA narrative injected into agent system prompt
- "Suggested questions" derived from EDA findings shown to user

### v3 — SQL Support (Week 4)
- Connect to SQLite or PostgreSQL database (alternative to file upload)
- `execute_sql` tool: read-only SELECT queries, up to 1000 rows
- Agent picks correct tool (SQL for aggregations, Python for plotting)
- SQL results bridged into Python namespace for chaining (query → plot)
- Mutations (INSERT/UPDATE/DELETE/DROP/etc.) rejected with clear error

---

## Dev Log Requirement

Maintain `dev-log.md` in the project root throughout the build.
Log categories:
- `[DECISION]` — why one approach was chosen over another
- `[MISTAKE]` — bugs, wrong assumptions, things that broke
- `[FIX]` — how mistakes were resolved
- `[IMPROVEMENT]` — refactors and polish beyond the minimum
- `[INSIGHT]` — non-obvious learnings (Claude API, Streamlit, pandas, etc.)
- `[TRADEOFF]` — cases where the simpler option was chosen and why

At project completion, generate a **final report** covering:
- Architecture decisions and rationale
- Mistakes made and what they taught
- Improvements from v1 → v2 → v3
- What to do differently on a second pass
- Interview talking points

---

## Future Requirements

_Add new requirements here when they come up:_

<!-- Example format:
### Requirement Name (date added)
Description of what's needed and why.
Priority: High / Medium / Low
Phase: v1 / v2 / v3 / new phase
-->

---

## Out of Scope (for now)

- Production deployment / cloud hosting
- Multi-user support
- Authentication
- Real sandboxed code execution (e.g., E2B) — document as future upgrade
- Streaming responses (can be added as polish)
- File size > 500MB (warn and offer to sample)
