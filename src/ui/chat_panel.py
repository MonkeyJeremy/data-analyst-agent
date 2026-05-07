from __future__ import annotations

import json

import streamlit as st


def render_chat_history(messages: list[dict]) -> None:
    """Render the full chat history including inline charts and code expanders."""
    for msg in messages:
        role = msg["role"]

        # Skip tool_result user messages — they're internal API scaffolding
        content = msg.get("content", "")
        if isinstance(content, list):
            _render_structured_content(role, content)
        else:
            if content:
                with st.chat_message(role):
                    st.markdown(content)


def _render_structured_content(role: str, blocks: list) -> None:
    """Handle messages whose content is a list of typed blocks."""
    # Collect text and figures for display
    text_parts: list[str] = []
    tool_uses: list[dict] = []

    for block in blocks:
        if isinstance(block, dict):
            btype = block.get("type", "")
            if btype == "text":
                text_parts.append(block["text"])
            elif btype == "tool_use":
                tool_uses.append(block)
            # tool_result blocks are silently skipped
        else:
            # anthropic SDK typed objects
            btype = getattr(block, "type", "")
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_uses.append(
                    {"name": block.name, "input": dict(block.input), "id": block.id}
                )

    if not text_parts and not tool_uses:
        return

    with st.chat_message(role):
        if text_parts:
            st.markdown("\n".join(text_parts))
        for tu in tool_uses:
            purpose = tu.get("input", {}).get("purpose", "")
            code = tu.get("input", {}).get("code", "")
            label = f"Code — {purpose}" if purpose else "Code"
            with st.expander(label, expanded=False):
                st.code(code, language="python")


def render_turn_figures(
    figures: tuple[bytes, ...],
    plotly_figures: tuple[str, ...] = (),
) -> None:
    """Render charts produced during the latest agent turn.

    Plotly figures (interactive) are rendered first; matplotlib PNGs are the fallback.
    Charts are centred at their natural pixel width rather than stretched to fill the column.
    """
    for fig_json in plotly_figures:
        try:
            import plotly.io as pio
            fig = pio.from_json(fig_json)
            _render_plotly_centred(fig)
        except Exception:
            st.warning("Could not render interactive chart.")

    for fig_bytes in figures:
        st.image(fig_bytes, use_container_width=False)


def _render_plotly_centred(fig: object) -> None:
    """Render a Plotly figure at its natural width, centred in the column.

    Streamlit has no native centering option for st.plotly_chart, so we read
    the width from the figure layout and pad with empty columns on each side.
    The main Streamlit content column is ~730 px wide (layout="wide" gives
    more, but we use a conservative estimate so maths stays simple).
    """
    import plotly.graph_objects as go

    COLUMN_PX = 900  # conservative estimate of available content width

    layout_width = getattr(getattr(fig, "layout", None), "width", None)
    chart_px = int(layout_width) if layout_width else COLUMN_PX

    if chart_px >= COLUMN_PX:
        # Chart fills the column — no padding needed
        st.plotly_chart(fig, use_container_width=False)
        return

    # Build column ratios so the chart sits in the centre
    pad = max(0, COLUMN_PX - chart_px) // 2
    left_w = pad
    right_w = pad
    mid_w = chart_px

    if left_w < 20:
        st.plotly_chart(fig, use_container_width=False)
        return

    col_left, col_mid, col_right = st.columns([left_w, mid_w, right_w])
    with col_mid:
        st.plotly_chart(fig, use_container_width=False)
