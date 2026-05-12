from __future__ import annotations

import streamlit as st


def render_file_upload() -> list | None:
    """Render the multi-file uploader.

    Returns a list of uploaded file objects (may be empty), or ``None`` if
    nothing has been uploaded yet.
    """
    result = st.file_uploader(
        "Upload data files",
        type=["csv", "xlsx", "xls", "json", "db", "sqlite"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Supported: .csv · .xlsx · .xls · .json · .db · .sqlite — upload multiple files to analyse them together.",
    )
    # st.file_uploader with accept_multiple_files returns [] when nothing is
    # uploaded; normalise to None so callers can use `if uploaded_files:`.
    return result if result else None
