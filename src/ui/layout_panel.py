"""Layout confirmation / info panel for non-standard file headers."""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src.data.layout import LayoutResult, preview_row
from src.data.loader import load_tabular


def render_layout_panel(
    result: LayoutResult,
    file_bytes: bytes,
    filename: str,
) -> pd.DataFrame | None:
    """Render the layout detection UI.

    Parameters
    ----------
    result:
        Detection result from detect_layout().
    file_bytes:
        Raw file bytes — needed to re-load with the user's selected header row.
        Pass ``b""`` for the banner-only (auto_fixed) path.
    filename:
        Original filename.

    Returns
    -------
    pd.DataFrame | None
        The confirmed DataFrame when the user clicks "Apply", otherwise None.
        Always returns None for status="auto_fixed" (no confirmation needed).
    """
    if result.status == "auto_fixed":
        st.info(f"📐 {result.message}")
        return None

    if result.status != "needs_confirmation":
        return None

    # ── Confirmation panel ────────────────────────────────────────────────────
    st.warning(f"📐 {result.message}")

    # Build selectbox options
    options = list(result.candidate_rows) if result.candidate_rows else list(range(5))

    def _row_label(i: int) -> str:
        if not file_bytes:
            return f"Row {i + 1}"
        preview = preview_row(file_bytes, filename, i, n_cells=5)
        return f"Row {i + 1} — {preview}"

    selected = st.selectbox(
        "Select the header row:",
        options=options,
        index=0,
        format_func=_row_label,
        key="_layout_header_selectbox",
    )

    # Live preview of how the data would look with the selected header
    if file_bytes and selected is not None:
        try:
            df_preview = load_tabular(
                io.BytesIO(file_bytes), filename, header=selected
            )
            st.caption(
                f"Preview — {len(df_preview):,} rows × {len(df_preview.columns)} columns "
                f"({df_preview.select_dtypes('number').shape[1]} numeric)"
            )
            st.dataframe(df_preview.head(5), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not preview with row {selected + 1} as header: {exc}")

    col_apply, col_cancel = st.columns([1, 4])
    with col_apply:
        if st.button("✅ Apply", key="_layout_apply_btn", type="primary"):
            if file_bytes:
                try:
                    df = load_tabular(
                        io.BytesIO(file_bytes), filename, header=selected
                    )
                    return df
                except Exception as exc:
                    st.error(f"Failed to load: {exc}")
    with col_cancel:
        if st.button("❌ Cancel upload", key="_layout_cancel_btn"):
            # Signal caller to reset upload state
            return _CANCEL_SENTINEL  # type: ignore[return-value]

    return None


# Sentinel object for cancel — caller checks `is _CANCEL_SENTINEL`
_CANCEL_SENTINEL = object()
