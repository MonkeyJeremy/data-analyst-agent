from __future__ import annotations

import io
import os

import streamlit as st
from dotenv import load_dotenv

from src.agent.client import LLMClient
from src.agent.loop import run_agent_turn
from src.data.layout import detect_layout
from src.data.loader import load_tabular
from src.data.schema import describe_schema
from src.db.connection import connect_sqlite_file
from src.db.executor import load_table
from src.db.schema import describe_sql_schema
from src.eda.auto_eda import run_auto_eda
from src.ui.chat_panel import render_chat_history, render_turn_figures
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
    defaults: dict = {
        "df": None,
        "schema": None,
        "eda": None,                 # EDAReport computed on upload
        "messages": [],
        "last_figures": (),
        "last_plotly_figures": (),
        "_last_filename": None,
        "pending_query": None,       # set by suggestion chips
        "_layout_result": None,      # LayoutResult from last detection
        "_layout_file_bytes": None,  # raw bytes held for user confirmation
        # SQL mode
        "sql_connection": None,      # SQLConnection | None
        "sql_tables": None,          # list[str] | None — multi-table picker
        "_mode": "dataframe",        # "dataframe" | "sql"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _get_api_key(sidebar_key: str) -> str | None:
    return sidebar_key.strip() or os.getenv("ANTHROPIC_API_KEY") or None


# ── Sidebar ──────────────────────────────────────────────────────────────────

def _render_sidebar() -> str:
    """Render sidebar and return the API key string (may be empty)."""
    with st.sidebar:
        st.markdown("## 📊 Data Analyst Agent")
        st.markdown("<hr>", unsafe_allow_html=True)

        # API key
        st.markdown("#### 🔑 API Key")
        api_key_input = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            label_visibility="collapsed",
            help="Leave blank to use the ANTHROPIC_API_KEY environment variable.",
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # File upload
        st.markdown("#### 📁 Data Source")
        uploaded = render_file_upload()

        if uploaded is not None:
            _handle_upload(uploaded)

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
            st.session_state.df = None
            st.session_state.schema = None
            st.session_state.eda = None
            st.session_state["_last_filename"] = "SQL connection"
            st.rerun()

        # Dataset preview (DataFrame mode only)
        if st.session_state.df is not None:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("#### 🔍 Preview")
            st.dataframe(
                st.session_state.df.head(5),
                use_container_width=True,
                hide_index=False,
            )

        # Reset button
        if (
            st.session_state.df is not None
            or st.session_state.get("sql_connection") is not None
        ):
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️  Clear & upload new file", use_container_width=True):
                _reset_all_state()
                st.rerun()

    return api_key_input


def _reset_all_state() -> None:
    """Reset all data/chat state back to the initial empty condition."""
    # Dispose SQL connection before clearing
    conn = st.session_state.get("sql_connection")
    if conn is not None:
        conn.dispose()

    for key in (
        "df", "schema", "eda", "_last_filename",
        "_layout_result", "_layout_file_bytes",
        "sql_connection", "sql_tables",
    ):
        st.session_state[key] = None

    st.session_state.messages = []
    st.session_state.last_figures = ()
    st.session_state.last_plotly_figures = ()
    st.session_state.pending_query = None
    st.session_state._mode = "dataframe"


def _commit_upload(df: object, filename: str) -> None:
    """Persist a successfully loaded DataFrame and run schema + EDA."""
    schema = describe_schema(df)
    st.session_state.df = df
    st.session_state.schema = schema
    st.session_state.eda = run_auto_eda(df)
    st.session_state.messages = []
    st.session_state.last_figures = ()
    st.session_state.last_plotly_figures = ()
    st.session_state["_last_filename"] = filename
    st.session_state.pending_query = None
    st.session_state._mode = "dataframe"
    # Clear any SQL state
    old_conn = st.session_state.get("sql_connection")
    if old_conn is not None:
        old_conn.dispose()
    st.session_state.sql_connection = None
    st.session_state.sql_tables = None


def _handle_upload(uploaded: object) -> None:
    # Skip if this file is already loaded (or awaiting confirmation)
    if st.session_state.get("_last_filename") == uploaded.name:
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

        if result.status == "needs_confirmation":
            # Don't load yet — main() will render the confirmation panel
            return

        # auto_fixed or ok: load with detected header row
        df = load_tabular(
            io.BytesIO(file_bytes), uploaded.name, header=result.header_row
        )
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

def _run_query(prompt: str, api_key: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("🤔 Analysing…"):
        try:
            if st.session_state.get("_mode") == "sql":
                sql_conn = st.session_state.sql_connection
                sql_schema = describe_sql_schema(sql_conn.engine)
                result = run_agent_turn(
                    client=LLMClient(api_key=api_key),
                    messages=st.session_state.messages,
                    sql_engine=sql_conn.engine,
                    sql_schema=sql_schema,
                )
            else:
                eda = st.session_state.get("eda")
                eda_summary = eda.narrative if eda is not None else None
                text_cols = tuple(eda.text_cols) if eda is not None else ()
                result = run_agent_turn(
                    client=LLMClient(api_key=api_key),
                    messages=st.session_state.messages,
                    df=st.session_state.df,
                    schema=st.session_state.schema,
                    eda_summary=eda_summary,
                    text_cols=text_cols,
                )
            st.session_state.messages = result.messages
            st.session_state.last_figures = result.figures
            st.session_state.last_plotly_figures = result.plotly_figures
        except Exception as exc:  # noqa: BLE001
            st.error(f"Agent error: {exc}")
            st.session_state.messages.pop()
            return
    st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    inject()
    _init_session_state()

    api_key_input = _render_sidebar()

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

    # EDA panel — DataFrame mode only
    if not is_sql_mode and st.session_state.eda is not None:
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

    # Handle a suggestion chip click (fires before chat_input check)
    if st.session_state.pending_query:
        pending = st.session_state.pending_query
        st.session_state.pending_query = None
        api_key = _get_api_key(api_key_input)
        if not api_key:
            st.error("No API key found. Enter one in the sidebar or set ANTHROPIC_API_KEY.")
            return
        _run_query(pending, api_key)
        return

    if prompt := st.chat_input("Ask a question about your data…"):
        api_key = _get_api_key(api_key_input)
        if not api_key:
            st.error("No API key found. Enter one in the sidebar or set ANTHROPIC_API_KEY.")
            return
        _run_query(prompt, api_key)


if __name__ == "__main__":
    main()
