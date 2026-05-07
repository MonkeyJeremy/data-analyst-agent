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
    """
    for fig_json in plotly_figures:
        try:
            import plotly.io as pio
            fig = pio.from_json(fig_json)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("Could not render interactive chart.")

    for fig_bytes in figures:
        st.image(fig_bytes, use_container_width=True)
