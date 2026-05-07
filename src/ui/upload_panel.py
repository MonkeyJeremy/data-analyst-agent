from __future__ import annotations

import streamlit as st


def render_file_upload() -> object | None:
    """Render the file uploader. Returns the uploaded file object or None."""
    return st.file_uploader(
        "Upload a data file",
        type=["csv", "xlsx", "xls", "json", "db", "sqlite"],
        label_visibility="collapsed",
        help="Supported: .csv · .xlsx · .xls · .json · .db · .sqlite",
    )
