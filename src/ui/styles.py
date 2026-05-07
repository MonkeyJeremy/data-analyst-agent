"""Global CSS injected once per Streamlit session."""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* ── Hide Streamlit chrome ───────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0e0e1a;
    border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* ── Main content area ───────────────────────────────────────────────────── */
.stApp { background: #111120; }
.block-container { padding-top: 2rem !important; }

/* ── Metric cards ────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 1rem 1.25rem !important;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(120,120,255,0.35);
}
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; opacity: 0.6; }
[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700; }

/* ── Chat messages ───────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 0.25rem 0.5rem;
    margin-bottom: 0.25rem;
}

/* ── Chat input ──────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    background: rgba(255,255,255,0.04) !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(120,120,255,0.5) !important;
    box-shadow: 0 0 0 2px rgba(100,100,255,0.12) !important;
}

/* ── Suggestion chips ────────────────────────────────────────────────────── */
.suggestion-btn button {
    border-radius: 20px !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    background: rgba(255,255,255,0.04) !important;
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.82rem !important;
    padding: 0.3rem 0.9rem !important;
    height: auto !important;
    white-space: normal !important;
    text-align: left !important;
    transition: all 0.18s ease;
    width: 100%;
}
.suggestion-btn button:hover {
    background: rgba(100,100,255,0.15) !important;
    border-color: rgba(120,120,255,0.4) !important;
    color: #fff !important;
}

/* ── Feature cards (welcome screen) ─────────────────────────────────────── */
.feature-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem 1.25rem;
    text-align: center;
    height: 100%;
    transition: border-color 0.2s, background 0.2s;
}
.feature-card:hover {
    background: rgba(255,255,255,0.06);
    border-color: rgba(120,120,255,0.35);
}
.feature-icon { font-size: 2rem; margin-bottom: 0.6rem; }
.feature-title { font-size: 1rem; font-weight: 600; margin-bottom: 0.4rem; color: #e0e0ff; }
.feature-desc  { font-size: 0.82rem; opacity: 0.55; line-height: 1.5; }

/* ── Expander (code blocks) ──────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    background: rgba(0,0,0,0.2) !important;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── Success / error alerts ──────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 10px !important; }
</style>
"""


def inject() -> None:
    """Call once at the top of main() to apply global styles."""
    st.markdown(_CSS, unsafe_allow_html=True)
