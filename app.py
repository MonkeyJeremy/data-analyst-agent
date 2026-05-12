from __future__ import annotations

import io
import os

import streamlit as st
from dotenv import load_dotenv

from src.agent.client import create_client
from src.agent.viz_planner import plan_visualization, build_viz_hint
from src.config import PROVIDERS
from src.agent.langgraph_loop import run_langgraph_turn  # LangGraph refactor (feat/langgraph)
from src.data.join_detector import detect_join_keys, JoinSuggestion
from src.data.layout import detect_layout
from src.data.loader import load_tabular
from src.data.registry import DataFrameRegistry
from src.data.schema import describe_schema
from src.db.connection import connect_sqlite_file
from src.db.executor import load_table
from src.db.schema import describe_sql_schema
from src.eda.auto_eda import run_auto_eda
from src.ui.chat_panel import render_chat_history, render_turn_figures, render_turn_downloads
from src.ui.eda_panel import render_eda_panel
from src.ui.layout_panel import _CANCEL_SENTINEL, render_layout_panel
from src.ui.sql_panel import render_sql_connect_panel, render_sql_stats
from src.ui.styles import inject
from src.ui.upload_panel import render_file_upload

load_dotenv()

st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Suggested questions per dataset (generic; agent will personalise) ────────
_DEFAULT_SUGGESTIONS = [
    "What does this dataset look like? Give me a summary.",
    "Show the distribution of each numeric column.",
    "Are there any missing values? Which columns are affected?",
    "What are the top correlations between numeric columns?",
    "Plot the most interesting relationship you can find.",
]

_SQL_SUGGESTIONS = [
    "What tables are in this database and how many rows does each have?",
    "Show me 10 sample rows from each table.",
    "What columns are available in each table?",
    "Find the top 10 records by the most relevant numeric column.",
    "Are there any obvious relationships between the tables?",
]

_TEXT_SUGGESTIONS = [
    "What is the overall sentiment in the text column?",
    "What are the most common topics or themes?",
    "Find examples of strongly negative feedback.",
    "Classify each entry as positive, negative, or neutral.",
    "What words appear most frequently?",
]


def _init_session_state() -> None:
    import uuid as _uuid  # local to avoid polluting module namespace
    defaults: dict = {
        "df": None,
        "schema": None,
        "eda": None,                 # EDAReport computed on upload
        # Multi-dataframe support
        "registry": DataFrameRegistry(),
        "join_suggestions": [],
        "manual_joins": [],       # list[JoinSuggestion] — user-defined, persist across uploads
        "active_df_name": None,      # name of df shown in EDA panel
        # LangGraph: stable thread ID for MemorySaver checkpointing within session
        "lg_thread_id": str(_uuid.uuid4()),
        "messages": [],
        "last_figures": (),
        "last_plotly_figures": (),
        "last_tool_calls": (),       # ToolCallRecord tuple from most recent turn
        "last_question": "",         # user question from most recent turn
        "last_answer": "",           # agent answer from most recent turn
        "_last_filename": None,
        "pending_query": None,       # set by suggestion chips
        "_layout_result": None,      # LayoutResult from last detection
        "_layout_file_bytes": None,  # raw bytes held for user confirmation
        "_pending_upload_bytes": None,  # bytes of file awaiting layout confirm
        "_pending_upload_name": None,   # filename awaiting layout confirm
        # SQL mode
        "sql_connection": None,      # SQLConnection | None
        "sql_tables": None,          # list[str] | None — multi-table picker
        "_mode": "dataframe",        # "dataframe" | "sql"
        # Execution mode
        "execution_mode": "local",   # "local" | "e2b" | "docker"
        # Token metering
        "session_input_tokens": 0,
        "session_output_tokens": 0,
        "session_cache_read_tokens": 0,
        "session_cache_write_tokens": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _get_api_key(sidebar_key: str, provider: str) -> str | None:
    key_env = PROVIDERS.get(provider, {}).get("key_env", "ANTHROPIC_API_KEY")
    return sidebar_key.strip() or os.getenv(key_env) or None


# ── Sidebar ──────────────────────────────────────────────────────────────────

def _render_join_builder(registry: "DataFrameRegistry") -> None:
    """Sidebar UI for manually defining a relationship between two tables."""
    if registry.count() < 2:
        return

    with st.expander("➕ Define Relationship", expanded=False):
        names = registry.names()

        col1, col2 = st.columns(2)
        left_table = col1.selectbox("Left table", names, key="jb_left_table")
        right_options = [n for n in names if n != left_table]
        right_table = col2.selectbox("Right table", right_options, key="jb_right_table")

        left_entry = registry.get(left_table)
        right_entry = registry.get(right_table)
        if left_entry is None or right_entry is None:
            return

        left_cols = list(left_entry.df.columns)
        right_cols = list(right_entry.df.columns)

        col3, col4 = st.columns(2)
        left_col = col3.selectbox("Left column", left_cols, key="jb_left_col")
        right_col = col4.selectbox("Right column", right_cols, key="jb_right_col")

        join_type = st.radio(
            "Join type", ["inner", "left", "right", "outer"],
            horizontal=True, key="jb_join_type"
        )

        if st.button("Add relationship", key="jb_add"):
            new_join = JoinSuggestion(
                left_table=left_table,
                left_col=left_col,
                right_table=right_table,
                right_col=right_col,
                match_rate=1.0,
                join_type=join_type,
                source="manual",
            )
            # Avoid exact duplicates
            existing = st.session_state.manual_joins
            already = any(
                j.left_table == new_join.left_table and
                j.left_col == new_join.left_col and
                j.right_table == new_join.right_table and
                j.right_col == new_join.right_col
                for j in existing
            )
            if not already:
                st.session_state.manual_joins = existing + [new_join]
                st.rerun()


def _render_relationships_panel(
    manual_joins: list,
    auto_joins: list,
) -> None:
    """Show all current relationships (manual + auto) with remove option for manual ones."""
    all_joins = manual_joins + auto_joins
    if not all_joins:
        return

    label = f"🔗 Relationships ({len(all_joins)})"
    with st.expander(label, expanded=bool(manual_joins)):
        if manual_joins:
            st.caption("User-defined")
            for i, j in enumerate(manual_joins):
                col1, col2 = st.columns([5, 1])
                col1.markdown(
                    f"`{j.left_table}.{j.left_col}` → "
                    f"`{j.right_table}.{j.right_col}` "
                    f"({j.join_type})"
                )
                if col2.button("✕", key=f"rm_join_{i}"):
                    updated = list(st.session_state.manual_joins)
                    updated.pop(i)
                    st.session_state.manual_joins = updated
                    st.rerun()

        if auto_joins:
            st.caption("Auto-detected")
            for j in auto_joins:
                pct = int(j.match_rate * 100)
                st.markdown(
                    f"`{j.left_table}.{j.left_col}` → "
                    f"`{j.right_table}.{j.right_col}` "
                    f"({pct}% match, {j.join_type})"
                )


def _render_sidebar() -> tuple[str, str, str]:
    """Render sidebar and return (api_key_input, provider, model)."""
    with st.sidebar:
        st.markdown("## 📊 Data Analyst Agent")
        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Provider + model selection ────────────────────────────────────────
        st.markdown("#### 🔑 API Configuration")
        provider = st.selectbox(
            "Provider",
            options=list(PROVIDERS.keys()),
            key="_provider",
            label_visibility="collapsed",
        )
        provider_cfg = PROVIDERS[provider]

        model = st.selectbox(
            "Model",
            options=provider_cfg["models"],
            index=0,
            key=f"_model_{provider}",
            label_visibility="collapsed",
        )

        api_key_input = st.text_input(
            f"{provider} API Key",
            type="password",
            placeholder=provider_cfg["key_placeholder"],
            label_visibility="collapsed",
            help=(
                f"Leave blank to use the {provider_cfg['key_env']} "
                "environment variable."
            ),
        )

        st.markdown("#### ⚙️ Execution Mode")
        execution_mode = st.selectbox(
            "Execution mode",
            options=["local", "e2b", "docker"],
            index=0,
            key="_execution_mode",
            label_visibility="collapsed",
            help="Local runs code in-process. E2B/Docker require extra setup.",
        )
        st.session_state.execution_mode = execution_mode
        st.markdown("<br>", unsafe_allow_html=True)

        # File upload (multi-file)
        st.markdown("#### 📁 Data Source")
        uploaded_files = render_file_upload()

        if uploaded_files:
            for uploaded in uploaded_files:
                _handle_upload(uploaded)

        # Loaded DataFrames list
        registry: DataFrameRegistry = st.session_state.registry
        if not registry.is_empty():
            st.markdown("**Loaded tables:**")
            for entry in registry.entries():
                is_active = st.session_state.get("active_df_name") == entry.name
                col_btn, col_remove = st.columns([4, 1])
                # Clicking the name toggles the table view in the main area
                label = f"{'▼' if is_active else '▶'} `{entry.name}` {entry.schema.n_rows:,}×{entry.schema.n_cols}"
                if col_btn.button(
                    label,
                    key=f"_view_{entry.name}",
                    use_container_width=True,
                    help="Click to view / hide table data",
                ):
                    st.session_state.active_df_name = None if is_active else entry.name
                    st.rerun()
                if col_remove.button("✕", key=f"_rm_{entry.name}", help=f"Remove {entry.name}"):
                    registry.remove(entry.name)
                    # Re-sync backward-compat keys
                    new_primary = registry.primary()
                    if new_primary is not None:
                        st.session_state.df = new_primary.df
                        st.session_state.schema = new_primary.schema
                        st.session_state.eda = new_primary.eda
                        st.session_state.active_df_name = new_primary.name
                    else:
                        st.session_state.df = None
                        st.session_state.schema = None
                        st.session_state.eda = None
                        st.session_state.active_df_name = None
                    st.session_state.join_suggestions = detect_join_keys(registry)
                    st.rerun()

        # Manual relationship builder + combined relationships panel
        _render_join_builder(registry)
        _render_relationships_panel(
            st.session_state.manual_joins,
            st.session_state.join_suggestions,
        )

        # SQL connect (live database)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🗄️ Or connect to SQL database")
        conn = render_sql_connect_panel()
        if conn is not None:
            # Dispose old connection if any
            old = st.session_state.get("sql_connection")
            if old is not None:
                old.dispose()
            st.session_state.sql_connection = conn
            st.session_state.sql_tables = None
            st.session_state._mode = "sql"
            st.session_state.messages = []
            st.session_state.last_figures = ()
            st.session_state.last_plotly_figures = ()
            st.session_state.last_tool_calls = ()
            st.session_state.last_question = ""
            st.session_state.last_answer = ""
            st.session_state.df = None
            st.session_state.schema = None
            st.session_state.eda = None
            st.session_state["_last_filename"] = "SQL connection"
            st.rerun()

        # Token usage meter
        _render_token_meter()

        # Reset button
        if (
            st.session_state.df is not None
            or st.session_state.get("sql_connection") is not None
        ):
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️  Clear & upload new file", use_container_width=True):
                _reset_all_state()
                st.rerun()

    return api_key_input, provider, model


def _render_token_meter() -> None:
    """Show a compact token-usage summary in the sidebar."""
    inp = st.session_state.get("session_input_tokens", 0)
    out = st.session_state.get("session_output_tokens", 0)
    cached = st.session_state.get("session_cache_read_tokens", 0)
    if inp == 0 and out == 0:
        return  # nothing to show yet

    total = inp + out
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### 📊 Session tokens")
    col_i, col_o = st.columns(2)
    col_i.metric("Input", f"{inp:,}")
    col_o.metric("Output", f"{out:,}")
    if cached:
        pct = round(cached / max(inp, 1) * 100)
        st.caption(f"⚡ {cached:,} tokens from cache ({pct}% cache hit rate)")
    st.caption(f"Total this session: **{total:,}**")


def _reset_all_state() -> None:
    """Reset all data/chat state back to the initial empty condition."""
    # Dispose SQL connection before clearing
    conn = st.session_state.get("sql_connection")
    if conn is not None:
        conn.dispose()

    for key in (
        "df", "schema", "eda", "_last_filename",
        "_layout_result", "_layout_file_bytes",
        "_pending_upload_bytes", "_pending_upload_name",
        "sql_connection", "sql_tables",
        "active_df_name",
    ):
        st.session_state[key] = None

    st.session_state.registry = DataFrameRegistry()
    st.session_state.join_suggestions = []
    st.session_state.manual_joins = []
    st.session_state.messages = []
    st.session_state.last_figures = ()
    st.session_state.last_plotly_figures = ()
    st.session_state.last_tool_calls = ()
    st.session_state.last_question = ""
    st.session_state.last_answer = ""
    st.session_state.pending_query = None
    st.session_state._mode = "dataframe"
    st.session_state.session_input_tokens = 0
    st.session_state.session_output_tokens = 0
    st.session_state.session_cache_read_tokens = 0
    st.session_state.session_cache_write_tokens = 0


def _build_join_suggestion_message(
    registry: "DataFrameRegistry",
    auto_joins: list,
    manual_joins: list,
) -> str:
    """Build a proactive assistant message about detected/existing relationships.

    Shown in the chat whenever a second (or later) file is uploaded.
    Describes what was detected automatically; if manual joins already exist,
    acknowledges them instead.
    """
    names = registry.names()
    table_summary = ", ".join(f"**{n}**" for n in names)

    lines: list[str] = [
        f"I now have {registry.count()} tables loaded: {table_summary}.\n",
    ]

    if manual_joins:
        lines.append(
            f"You've defined {len(manual_joins)} relationship(s) manually — "
            "I'll use those when joining.\n"
        )
    elif auto_joins:
        lines.append(
            f"I automatically detected {len(auto_joins)} relationship(s) "
            "between these tables:\n"
        )
        for j in auto_joins:
            pct = int(j.match_rate * 100)
            lines.append(
                f"- `{j.left_table}.{j.left_col}` → "
                f"`{j.right_table}.{j.right_col}` "
                f"({pct}% value overlap — suggested **{j.join_type}** join)"
            )
        lines.append("")
        # Show a ready-to-use merge example for the top suggestion
        top = auto_joins[0]
        lines.append("Here's how to merge them:")
        lines.append(f"```python\nmerged = {top.example_code()}\n```")
        lines.append(
            "\nYou can ask me to analyse the combined dataset directly, "
            "or define your own relationships in the sidebar."
        )
    else:
        lines.append(
            "I couldn't detect obvious join keys automatically. "
            "You can define relationships manually in the **➕ Define Relationship** "
            "panel in the sidebar, or just ask me questions about each table individually."
        )

    return "\n".join(lines)


def _commit_upload(df: object, filename: str) -> None:
    """Add a successfully loaded DataFrame to the registry and refresh derived state.

    Multiple calls accumulate DataFrames; the registry grows with each upload.
    Backward-compat keys (``df``, ``schema``, ``eda``) are kept in sync with
    the registry's primary entry so that existing single-df code paths work
    without modification.
    """
    registry: DataFrameRegistry = st.session_state.registry
    entry = registry.add(filename, df)

    # Recompute join suggestions across all loaded DataFrames
    st.session_state.join_suggestions = detect_join_keys(registry)

    # Sync backward-compat keys from the primary entry
    primary = registry.primary()
    st.session_state.df = primary.df
    st.session_state.schema = primary.schema
    st.session_state.eda = primary.eda
    st.session_state.active_df_name = primary.name

    # Reset conversation on first upload (empty registry before this call had
    # count == 1 now, so clear only when this is the very first file).
    if registry.count() == 1:
        st.session_state.messages = []
        st.session_state.last_figures = ()
        st.session_state.last_plotly_figures = ()
        st.session_state.last_tool_calls = ()
        st.session_state.last_question = ""
        st.session_state.last_answer = ""
        st.session_state.pending_query = None

    # When a second (or later) file is added, inject a proactive assistant
    # message describing auto-detected relationships and suggesting merges.
    # Only fires when there are no user-defined manual joins (they already
    # know the schema). Replaces any previous auto-suggestion so it stays fresh.
    if registry.count() >= 2:
        auto_joins = st.session_state.join_suggestions
        manual_joins = st.session_state.get("manual_joins", [])
        suggestion_msg = _build_join_suggestion_message(registry, auto_joins, manual_joins)
        messages: list = st.session_state.messages
        # Remove any previous auto-suggestion (identified by sentinel prefix)
        messages = [m for m in messages if not m.get("_auto_join_hint")]
        messages.append({
            "role": "assistant",
            "content": suggestion_msg,
            "_auto_join_hint": True,   # sentinel — stripped before sending to LLM
        })
        st.session_state.messages = messages

    st.session_state["_last_filename"] = filename
    st.session_state._mode = "dataframe"
    # Clear any SQL state
    old_conn = st.session_state.get("sql_connection")
    if old_conn is not None:
        old_conn.dispose()
    st.session_state.sql_connection = None
    st.session_state.sql_tables = None


def _handle_upload(uploaded: object) -> None:
    # Skip if this file is already loaded (check registry filenames list)
    registry: DataFrameRegistry = st.session_state.registry
    if uploaded.name in registry.filenames():
        return
    # Also skip if awaiting layout confirmation for this exact file
    if st.session_state.get("_pending_upload_name") == uploaded.name:
        return

    try:
        file_bytes = uploaded.read()
        uploaded.seek(0)  # reset so pandas can read it too
        name_lower = uploaded.name.lower()

        # ── SQLite path ───────────────────────────────────────────────────────
        if name_lower.endswith((".db", ".sqlite")):
            conn = connect_sqlite_file(file_bytes)
            st.session_state["_last_filename"] = uploaded.name
            st.session_state._layout_result = None
            if len(conn.tables) == 1:
                # Single table → load as DataFrame (no mode switch needed)
                df = load_table(conn.engine, conn.tables[0])
                conn.dispose()
                _commit_upload(df, uploaded.name)
            else:
                # Multiple tables → show picker in main()
                st.session_state.sql_connection = conn
                st.session_state.sql_tables = list(conn.tables)
            return

        # ── JSON path ─────────────────────────────────────────────────────────
        if name_lower.endswith(".json"):
            df = load_tabular(io.BytesIO(file_bytes), uploaded.name)
            st.session_state["_last_filename"] = uploaded.name
            st.session_state._layout_result = None
            _commit_upload(df, uploaded.name)
            return

        # ── CSV / Excel path — run layout detection ───────────────────────────
        result = detect_layout(file_bytes, uploaded.name)
        st.session_state._layout_result = result
        st.session_state._layout_file_bytes = file_bytes
        st.session_state["_last_filename"] = uploaded.name
        st.session_state._pending_upload_name = uploaded.name
        st.session_state._pending_upload_bytes = file_bytes

        if result.status == "needs_confirmation":
            # Don't load yet — main() will render the confirmation panel
            return

        # auto_fixed or ok: load with detected header row
        df = load_tabular(
            io.BytesIO(file_bytes), uploaded.name, header=result.header_row
        )
        st.session_state._pending_upload_name = None
        st.session_state._pending_upload_bytes = None
        _commit_upload(df, uploaded.name)

    except ValueError as exc:
        st.error(str(exc))


# ── Welcome screen ────────────────────────────────────────────────────────────

def _render_welcome() -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align:center;font-size:2.4rem;margin-bottom:0.3rem'>"
        "📊 Data Analyst Agent</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;opacity:0.5;font-size:1rem;margin-bottom:2.5rem'>"
        "Ask questions about your data in plain English. "
        "Powered by Claude + real Python execution.</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    cards = [
        ("📤", "Upload", "CSV, Excel, JSON or SQLite files — no size limit for most datasets."),
        ("💬", "Ask", "Type a question in plain English. The agent writes and runs real code."),
        ("📈", "Discover", "Get tables, interactive charts, and clear explanations instantly."),
    ]
    for col, (icon, title, desc) in zip((c1, c2, c3), cards):
        with col:
            st.markdown(
                f"""<div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;opacity:0.4;font-size:0.9rem'>"
        "⬅️  Upload a file or connect to a SQL database in the sidebar.</p>",
        unsafe_allow_html=True,
    )


# ── Dataset stats bar ─────────────────────────────────────────────────────────

def _render_dataset_stats() -> None:
    schema = st.session_state.schema
    df = st.session_state.df
    filename = st.session_state.get("_last_filename", "dataset")
    registry: DataFrameRegistry = st.session_state.registry

    if registry.count() > 1:
        st.markdown(f"### 📂 {registry.count()} tables loaded")
        for entry in registry.entries():
            st.caption(
                f"`{entry.name}` ({entry.filename}) — "
                f"{entry.schema.n_rows:,} rows × {entry.schema.n_cols} cols"
            )
        # Show join suggestions if any
        join_suggestions = st.session_state.get("join_suggestions", [])
        if join_suggestions:
            with st.expander("🔗 Detected join keys", expanded=False):
                for j in join_suggestions:
                    pct = int(j.match_rate * 100)
                    st.code(j.example_code(), language="python")
                    st.caption(f"{pct}% value overlap — suggested `{j.join_type}` join")
    else:
        st.markdown(f"### 📂 `{filename}`")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{schema.n_rows:,}")
    m2.metric("Columns", schema.n_cols)
    m3.metric("Numeric cols", int((df.dtypes != "object").sum()))
    m4.metric("Missing cells", f"{int(df.isnull().sum().sum()):,}")

    st.markdown("<br>", unsafe_allow_html=True)


# ── SQL table picker ──────────────────────────────────────────────────────────

def _render_sql_table_picker() -> None:
    """Let the user choose a table and how to open it (DataFrame vs SQL mode)."""
    conn = st.session_state.sql_connection
    if conn is None:
        return

    st.markdown("### 🗄️ Multiple tables found")
    st.caption(
        f"The uploaded database contains {len(conn.tables)} tables. "
        "Choose one to explore, or enter SQL mode to query across all tables."
    )

    table = st.selectbox(
        "Select table:",
        options=st.session_state.sql_tables,
        key="_sql_table_picker",
    )

    col_df, col_sql = st.columns([1, 1])
    with col_df:
        if st.button("📊 Load as DataFrame", use_container_width=True, type="primary"):
            try:
                df = load_table(conn.engine, table)
                conn.dispose()
                st.session_state.sql_connection = None
                st.session_state.sql_tables = None
                _commit_upload(df, table)
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load table '{table}': {exc}")

    with col_sql:
        if st.button("🔍 SQL Mode (all tables)", use_container_width=True):
            st.session_state._mode = "sql"
            st.session_state.sql_tables = None
            st.rerun()


# ── Suggestion chips ──────────────────────────────────────────────────────────

def _render_suggestions() -> None:
    """Show clickable example questions. Sets session_state.pending_query on click."""
    if st.session_state.get("_mode") == "sql":
        questions = _SQL_SUGGESTIONS
    else:
        eda = st.session_state.get("eda")
        if eda and eda.text_cols:
            questions = _TEXT_SUGGESTIONS
        elif eda:
            questions = list(eda.suggested_questions)
        else:
            questions = _DEFAULT_SUGGESTIONS

    st.markdown(
        "<p style='opacity:0.45;font-size:0.82rem;margin-bottom:0.5rem'>"
        "💡 Try asking…</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(questions))
    for col, question in zip(cols, questions):
        with col:
            st.markdown('<div class="suggestion-btn">', unsafe_allow_html=True)
            if st.button(question, key=f"sug_{question[:30]}"):
                st.session_state.pending_query = question
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)


# ── Agent turn ────────────────────────────────────────────────────────────────

def _run_query(prompt: str, api_key: str, provider: str, model: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("🤔 Analysing…"):
        try:
            client = create_client(provider=provider, api_key=api_key, model=model)
            viz_hint = build_viz_hint(plan_visualization(prompt))

            # Strip UI-only sentinel keys before sending to the LLM
            llm_messages = [
                {k: v for k, v in m.items() if k != "_auto_join_hint"}
                for m in st.session_state.messages
            ]

            thread_id = st.session_state.get("lg_thread_id")
            if st.session_state.get("_mode") == "sql":
                sql_conn = st.session_state.sql_connection
                sql_schema = describe_sql_schema(sql_conn.engine)
                result = run_langgraph_turn(
                    messages=llm_messages,
                    sql_engine=sql_conn.engine,
                    sql_schema=sql_schema,
                    thread_id=thread_id,
                )
            else:
                eda = st.session_state.get("eda")
                text_cols = tuple(eda.text_cols) if eda is not None else ()
                combined_joins = (
                    st.session_state.manual_joins
                    + st.session_state.get("join_suggestions", [])
                )
                result = run_langgraph_turn(
                    messages=llm_messages,
                    registry=st.session_state.registry,
                    join_suggestions=combined_joins,
                    text_cols=text_cols,
                    viz_hint=viz_hint,
                    client=client,
                    thread_id=thread_id,
                )
            st.session_state.messages = result.messages
            st.session_state.last_figures = result.figures
            st.session_state.last_plotly_figures = result.plotly_figures
            st.session_state.last_tool_calls = result.tool_calls
            st.session_state.last_question = prompt
            st.session_state.last_answer = result.final_text
            # Accumulate token usage for the sidebar meter
            if result.token_usage is not None:
                st.session_state.session_input_tokens += result.token_usage.input_tokens
                st.session_state.session_output_tokens += result.token_usage.output_tokens
                st.session_state.session_cache_read_tokens += result.token_usage.cache_read_tokens
                st.session_state.session_cache_write_tokens += result.token_usage.cache_write_tokens
        except Exception as exc:  # noqa: BLE001
            st.error(f"Agent error: {exc}")
            st.session_state.messages.pop()
            return
    st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    inject()
    _init_session_state()

    api_key_input, provider, model = _render_sidebar()

    # ── Layout confirmation / auto-fix banner ────────────────────────────────
    layout_result = st.session_state.get("_layout_result")
    if layout_result is not None:
        if layout_result.status == "needs_confirmation":
            # Show header selector — block normal content until resolved
            confirmed_df = render_layout_panel(
                layout_result,
                st.session_state._layout_file_bytes or b"",
                st.session_state.get("_last_filename", ""),
            )
            if confirmed_df is _CANCEL_SENTINEL:
                # User cancelled — reset upload state entirely
                for key in ("_layout_result", "_layout_file_bytes", "_last_filename"):
                    st.session_state[key] = None
                st.rerun()
            elif confirmed_df is not None:
                _commit_upload(confirmed_df, st.session_state["_last_filename"])
                st.session_state._layout_result = None
                st.rerun()
            else:
                # Waiting for user to click Apply
                if st.session_state.df is None:
                    return  # nothing else to show
        elif layout_result.status == "auto_fixed":
            # Non-blocking banner shown inline above the stats bar
            render_layout_panel(layout_result, b"", "")

    # ── SQL table picker (multi-table SQLite upload) ──────────────────────────
    if st.session_state.sql_tables:
        _render_sql_table_picker()
        return

    # ── Welcome screen ────────────────────────────────────────────────────────
    is_sql_mode = st.session_state.get("_mode") == "sql"
    has_data = st.session_state.df is not None or (
        is_sql_mode and st.session_state.sql_connection is not None
    )
    if not has_data:
        _render_welcome()
        return

    # ── Stats bar ─────────────────────────────────────────────────────────────
    if is_sql_mode:
        render_sql_stats(
            st.session_state.sql_connection,
            st.session_state.get("_last_filename", "database"),
        )
    else:
        _render_dataset_stats()

    # Table view panel — shown when user clicks a table name in the sidebar
    if not is_sql_mode:
        registry: DataFrameRegistry = st.session_state.registry
        active_name = st.session_state.get("active_df_name")
        active_entry = registry.get(active_name) if active_name else None
        if active_entry is not None:
            hdr_col, close_col = st.columns([6, 1])
            hdr_col.markdown(
                f"### 📋 `{active_entry.name}` "
                f"<span style='font-size:0.85rem;color:gray;'>"
                f"{active_entry.schema.n_rows:,} rows × {active_entry.schema.n_cols} cols"
                f"</span>",
                unsafe_allow_html=True,
            )
            if close_col.button("✕ Close", key="_close_table_view"):
                st.session_state.active_df_name = None
                st.rerun()
            st.dataframe(
                active_entry.df,
                use_container_width=True,
                hide_index=False,
            )
            st.divider()

    # EDA panel — DataFrame mode only
    if not is_sql_mode:
        registry: DataFrameRegistry = st.session_state.registry
        if registry.count() > 1:
            chosen_entry = registry.get(st.session_state.get("active_df_name")) or registry.primary()
            if chosen_entry is not None:
                render_eda_panel(chosen_entry.eda, chosen_entry.df)
        elif st.session_state.eda is not None:
            render_eda_panel(st.session_state.eda, st.session_state.df)

    # Show suggestions only when chat is empty
    if not st.session_state.messages:
        _render_suggestions()

    render_chat_history(st.session_state.messages)

    if st.session_state.last_plotly_figures or st.session_state.last_figures:
        render_turn_figures(
            st.session_state.last_figures,
            st.session_state.last_plotly_figures,
        )
        render_turn_downloads(
            plotly_figures=st.session_state.last_plotly_figures,
            question=st.session_state.get("last_question", ""),
            answer=st.session_state.get("last_answer", ""),
            tool_calls=st.session_state.get("last_tool_calls", ()),
            df=st.session_state.df,
        )

    # Handle a suggestion chip click (fires before chat_input check)
    if st.session_state.pending_query:
        pending = st.session_state.pending_query
        st.session_state.pending_query = None
        api_key = _get_api_key(api_key_input, provider)
        if not api_key:
            key_env = PROVIDERS.get(provider, {}).get("key_env", "ANTHROPIC_API_KEY")
            st.error(f"No API key found. Enter one in the sidebar or set {key_env}.")
            return
        _run_query(pending, api_key, provider, model)
        return

    if prompt := st.chat_input("Ask a question about your data…"):
        api_key = _get_api_key(api_key_input, provider)
        if not api_key:
            key_env = PROVIDERS.get(provider, {}).get("key_env", "ANTHROPIC_API_KEY")
            st.error(f"No API key found. Enter one in the sidebar or set {key_env}.")
            return
        _run_query(prompt, api_key, provider, model)


if __name__ == "__main__":
    main()
