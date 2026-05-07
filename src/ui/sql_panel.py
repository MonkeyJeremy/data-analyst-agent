"""Sidebar SQL connection UI and inline SQL-mode stats bar."""
from __future__ import annotations

import streamlit as st

from src.db.connection import SQLConnection, connect_url


def render_sql_connect_panel() -> SQLConnection | None:
    """Render a connection-string input + Connect button in the sidebar.

    Returns
    -------
    SQLConnection | None
        A live connection if the user just clicked "Connect" successfully,
        otherwise ``None``.
    """
    conn_str = st.text_input(
        "SQL Connection String",
        placeholder="postgresql://user:pass@host/db   or   sqlite:///path.db",
        type="password",
        label_visibility="collapsed",
        key="_sql_conn_input",
        help=(
            "Supported: PostgreSQL, MySQL, SQLite, and any SQLAlchemy-compatible URL. "
            "Example: postgresql://admin:secret@localhost:5432/mydb"
        ),
    )
    if st.button("🔌 Connect", key="_sql_connect_btn", use_container_width=True):
        if not conn_str.strip():
            st.error("Please enter a connection string.")
            return None
        try:
            with st.spinner("Connecting…"):
                conn = connect_url(conn_str)
            st.success(
                f"Connected — {len(conn.tables)} table(s) found "
                f"({conn.dialect.upper()})"
            )
            return conn
        except ValueError as exc:
            st.error(f"Connection failed: {exc}")
    return None


def render_sql_stats(conn: SQLConnection, filename: str) -> None:
    """Render the SQL-mode header bar (equivalent to the DataFrame stats bar).

    Parameters
    ----------
    conn:
        Active SQL connection.
    filename:
        Display name for the data source (uploaded filename or URL label).
    """
    st.markdown(f"### 🗄️ `{filename}`")

    m1, m2 = st.columns(2)
    m1.metric("Tables", len(conn.tables))
    m2.metric("Dialect", conn.dialect.upper())

    if conn.tables:
        table_list = " · ".join(f"`{t}`" for t in conn.tables)
        st.caption(f"Tables: {table_list}")

    st.markdown("<br>", unsafe_allow_html=True)
