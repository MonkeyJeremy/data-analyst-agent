from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from src.agent.client import LLMClient
from src.agent.loop import run_agent_turn
from src.data.loader import load_tabular
from src.data.schema import describe_schema
from src.ui.chat_panel import render_chat_history, render_turn_figures
from src.ui.upload_panel import render_file_upload

load_dotenv()

st.set_page_config(page_title="Data Analyst Agent", layout="wide")


def _init_session_state() -> None:
    defaults = {
        "df": None,
        "schema": None,
        "messages": [],
        "last_figures": (),
        "last_plotly_figures": (),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _get_api_key(sidebar_key: str) -> str | None:
    return sidebar_key.strip() or os.getenv("ANTHROPIC_API_KEY") or None


def main() -> None:
    _init_session_state()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        api_key_input = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            help="Leave blank to use the ANTHROPIC_API_KEY environment variable.",
        )
        uploaded = render_file_upload()

        if uploaded is not None:
            try:
                df = load_tabular(uploaded, uploaded.name)
                schema = describe_schema(df)
                # Only reset if a new file is uploaded
                if (
                    st.session_state.df is None
                    or getattr(st.session_state.df, "shape", None) != df.shape
                    or st.session_state.get("_last_filename") != uploaded.name
                ):
                    st.session_state.df = df
                    st.session_state.schema = schema
                    st.session_state.messages = []
                    st.session_state.last_figures = ()
                    st.session_state["_last_filename"] = uploaded.name
                    st.success(
                        f"Loaded **{uploaded.name}** — "
                        f"{schema.n_rows:,} rows × {schema.n_cols} columns"
                    )
            except ValueError as exc:
                st.error(str(exc))

        if st.session_state.df is not None:
            st.divider()
            st.caption("Dataset preview")
            st.dataframe(st.session_state.df.head(), use_container_width=True)

    # ── Main area ──────────────────────────────────────────────────────────────
    st.title("Data Analyst Agent")

    if st.session_state.df is None:
        st.info("Upload a CSV or Excel file in the sidebar to get started.")
        return

    render_chat_history(st.session_state.messages)

    # Render charts from the most recent turn below the last message
    if st.session_state.last_plotly_figures or st.session_state.last_figures:
        render_turn_figures(
            st.session_state.last_figures,
            st.session_state.last_plotly_figures,
        )

    if prompt := st.chat_input("Ask a question about your data…"):
        api_key = _get_api_key(api_key_input)
        if not api_key:
            st.error(
                "No API key found. Enter one in the sidebar or set ANTHROPIC_API_KEY."
            )
            return

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking…"):
            try:
                result = run_agent_turn(
                    client=LLMClient(api_key=api_key),
                    messages=st.session_state.messages,
                    df=st.session_state.df,
                    schema=st.session_state.schema,
                )
                st.session_state.messages = result.messages
                st.session_state.last_figures = result.figures
                st.session_state.last_plotly_figures = result.plotly_figures

            except Exception as exc:  # noqa: BLE001
                st.error(f"Agent error: {exc}")
                # Remove the user message so the user can retry
                st.session_state.messages.pop()
                return

        st.rerun()


if __name__ == "__main__":
    main()
