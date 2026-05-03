from __future__ import annotations

import streamlit as st


def render_file_upload() -> object | None:
    """Render the file uploader in the sidebar.

    Returns the uploaded file object or None if nothing is uploaded.
    """
    st.header("Data Source")
    uploaded = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: .csv, .xlsx, .xls",
    )
    return uploaded
