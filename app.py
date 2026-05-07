from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from src.agent.client import LLMClient
from src.agent.loop import run_agent_turn
from src.data.loader import load_tabular
from src.data.schema import describe_schema
from src.eda.auto_eda import run_auto_eda
from src.ui.chat_panel import render_chat_history, render_turn_figures
from src.ui.eda_panel import render_eda_panel
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


def _init_session_state() -> None:
    defaults: dict = {
        "df": None,
        "schema": None,
        "eda": None,             # EDAReport computed on upload
        "messages": [],
        "last_figures": (),
        "last_plotly_figures": (),
        "_last_filename": None,
        "pending_query": None,   # set by suggestion chips
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

        # Dataset preview
        if st.session_state.df is not None:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("#### 🔍 Preview")
            st.dataframe(
                st.session_state.df.head(5),
                use_container_width=True,
                hide_index=False,
            )

            # Reset button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️  Clear & upload new file", use_container_width=True):
                for key in ("df", "schema", "eda", "messages", "last_figures",
                            "last_plotly_figures", "_last_filename"):
                    st.session_state[key] = None if key not in ("messages",) else []
                st.session_state.last_figures = ()
                st.session_state.last_plotly_figures = ()
                st.rerun()

    return api_key_input


def _handle_upload(uploaded: object) -> None:
    try:
        df = load_tabular(uploaded, uploaded.name)
        schema = describe_schema(df)
        if (
            st.session_state.df is None
            or st.session_state.get("_last_filename") != uploaded.name
        ):
            st.session_state.df = df
            st.session_state.schema = schema
            st.session_state.eda = run_auto_eda(df)
            st.session_state.messages = []
            st.session_state.last_figures = ()
            st.session_state.last_plotly_figures = ()
            st.session_state["_last_filename"] = uploaded.name
            st.session_state.pending_query = None
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
        ("📤", "Upload", "Drop in any CSV or Excel file — no size limit for most datasets."),
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
        "⬅️  Upload a CSV or Excel file in the sidebar to get started.</p>",
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


# ── Suggestion chips ──────────────────────────────────────────────────────────

def _render_suggestions() -> None:
    """Show clickable example questions. Sets session_state.pending_query on click.

    Uses EDA-derived questions when available; falls back to generic defaults.
    """
    eda = st.session_state.get("eda")
    questions = list(eda.suggested_questions) if eda else _DEFAULT_SUGGESTIONS

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
            eda_summary = (
                st.session_state.eda.narrative
                if st.session_state.get("eda") is not None
                else None
            )
            result = run_agent_turn(
                client=LLMClient(api_key=api_key),
                messages=st.session_state.messages,
                df=st.session_state.df,
                schema=st.session_state.schema,
                eda_summary=eda_summary,
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

    if st.session_state.df is None:
        _render_welcome()
        return

    _render_dataset_stats()

    # EDA panel — collapsed by default, available immediately after upload
    if st.session_state.eda is not None:
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
