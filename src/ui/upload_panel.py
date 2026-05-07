from __future__ import annotations

import streamlit as st


def render_file_upload() -> object | None:
    """Render the file uploader. Returns the uploaded file object or None."""
    return st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
        help="Supported: .csv · .xlsx · .xls",
    )
