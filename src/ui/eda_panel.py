"""EDA panel — collapsible Streamlit section rendered after file upload."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.eda.report import EDAReport
from src.ui.chat_panel import _render_plotly_centred


def render_eda_panel(eda: EDAReport, df: pd.DataFrame) -> None:
    """Render the automated EDA expander (collapsed by default).

    Shows three tabs — Overview, Distributions, Correlations.
    """
    with st.expander("📊 Automated EDA", expanded=False):
        tab_overview, tab_dist, tab_corr = st.tabs(
            ["Overview", "Distributions", "Correlations"]
        )
        with tab_overview:
            _render_overview(eda)
        with tab_dist:
            _render_distributions(eda, df)
        with tab_corr:
            _render_correlations(eda, df)


# ── Overview tab ──────────────────────────────────────────────────────────────

def _render_overview(eda: EDAReport) -> None:
    # Missing-values chart
    if eda.missing_pct:
        cols_m = [c for c, _ in eda.missing_pct]
        pcts_m = [p for _, p in eda.missing_pct]
        fig = px.bar(
            x=pcts_m,
            y=cols_m,
            orientation="h",
            labels={"x": "% missing", "y": "Column"},
            title="Missing Values (%)",
        )
        _eda_fig_style(fig, height=max(200, len(cols_m) * 35 + 80), width=550)
        _render_plotly_centred(fig)
    else:
        st.success("No missing values ✅")

    # Data-quality bullets
    items: list[str] = []
    if eda.outlier_counts:
        top = ", ".join(f"**{c}** ({n})" for c, n in eda.outlier_counts[:3])
        items.append(f"IQR outliers — {top}")
    if eda.constant_cols:
        items.append(f"Constant columns (1 unique value) — {', '.join(f'**{c}**' for c in eda.constant_cols)}")
    if eda.high_cardinality_cols:
        items.append(f"High-cardinality columns — {', '.join(f'**{c}**' for c in eda.high_cardinality_cols)}")
    if eda.skewed_cols:
        top_sk = ", ".join(f"**{c}** (skew={s:.2f})" for c, s in eda.skewed_cols[:3])
        items.append(f"Skewed numeric columns — {top_sk}")

    if items:
        st.markdown("**Data quality notes:**")
        for item in items:
            st.markdown(f"- {item}")


# ── Distributions tab ─────────────────────────────────────────────────────────

def _render_distributions(eda: EDAReport, df: pd.DataFrame) -> None:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.info("No numeric columns to plot.")
        return

    display_cols = num_cols[:9]  # cap at 9
    grid_cols = st.columns(3)

    for idx, col in enumerate(display_cols):
        with grid_cols[idx % 3]:
            fig = px.histogram(
                df,
                x=col,
                nbins=30,
                title=col,
                labels={col: col, "count": ""},
            )
            _eda_fig_style(fig, height=220, width=260)
            st.plotly_chart(fig, use_container_width=False)


# ── Correlations tab ──────────────────────────────────────────────────────────

def _render_correlations(eda: EDAReport, df: pd.DataFrame) -> None:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns for a correlation heatmap.")
        return

    corr = df[num_cols].corr().round(2)
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Matrix",
        text_auto=True,
    )
    n = len(num_cols)
    size = max(300, min(700, n * 55 + 100))
    _eda_fig_style(fig, height=size, width=size)
    _render_plotly_centred(fig)

    if eda.top_correlations:
        st.markdown("**Top correlations (|r| > 0.30):**")
        for a, b, r in eda.top_correlations[:5]:
            bar = "🟩" if r > 0 else "🟥"
            st.markdown(f"- {bar} `{a}` × `{b}` — r = **{r:.3f}**")


# ── Shared style helper ───────────────────────────────────────────────────────

def _eda_fig_style(fig: go.Figure, height: int, width: int) -> None:
    """Minimal style for EDA figures — transparent bg + fixed size.

    Deliberately does NOT strip bar labels or apply bargap — EDA charts
    show values directly rather than hover-only like the agent charts.
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        autosize=False,
        height=height,
        width=width,
        margin={"t": 40, "b": 30, "l": 50, "r": 20},
        font={"size": 11},
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
