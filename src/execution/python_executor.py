import io
import traceback
from contextlib import redirect_stderr, redirect_stdout

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

from src.execution.result import ExecutionResult

matplotlib.use("Agg")  # non-interactive backend; must be set before any plt use


def execute_python(code: str, df: pd.DataFrame) -> ExecutionResult:
    """Execute code against a copy of df in a sandboxed namespace.

    Pre-imported names available in the namespace:
      df, pd, np, plt, sns, go (plotly.graph_objects), px (plotly.express).
    stdout and stderr are captured.
    Plotly figures (preferred) are serialised to JSON; matplotlib figures fall
    back to PNG bytes. The original df is never mutated.
    """
    import plotly.express as px  # local import keeps top-level fast

    namespace: dict = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "go": go,
        "px": px,
    }
    stdout_buf = io.StringIO()
    mpl_figures: list[bytes] = []
    plotly_figures: list[str] = []
    error: str | None = None

    plt.close("all")

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
            exec(compile(code, "<agent>", "exec"), namespace)  # noqa: S102

        # Collect Plotly figures from namespace (scan all values)
        for val in namespace.values():
            if isinstance(val, go.Figure):
                _apply_style(val)
                plotly_figures.append(pio.to_json(val))

        # Fall back to matplotlib only if no Plotly figures were produced
        if not plotly_figures:
            for fig_num in plt.get_fignums():
                buf = io.BytesIO()
                plt.figure(fig_num).savefig(buf, format="png", bbox_inches="tight", dpi=100)
                mpl_figures.append(buf.getvalue())

    except Exception:
        error = traceback.format_exc(limit=5)
    finally:
        plt.close("all")

    total_charts = len(plotly_figures) + len(mpl_figures)
    return ExecutionResult(
        stdout=stdout_buf.getvalue(),
        error=error,
        figures=tuple(mpl_figures),
        plotly_figures=tuple(plotly_figures),
        summary=_build_summary(stdout_buf.getvalue(), total_charts, error),
    )


def _apply_style(fig: go.Figure) -> None:
    """Mutate a Plotly figure in-place with transparent background and auto-sized height.

    - Transparent canvas so the chart blends with any Streamlit theme.
    - Height is computed from the data so dense charts aren't squashed and
      sparse charts don't waste vertical space.
    """
    height = _compute_height(fig)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",   # outer canvas transparent
        plot_bgcolor="rgba(0,0,0,0)",    # inner plot area transparent
        autosize=True,                   # width fills the Streamlit column
        height=height,
        margin={"t": 60, "b": 60, "l": 60, "r": 30},
        font={"size": 13},
        hoverlabel={"font_size": 13},
    )
    # Keep grid lines subtle so they're visible on both light and dark themes
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.3)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.3)")


def _compute_height(fig: go.Figure) -> int:
    """Return a height in pixels that fits the figure's data.

    Rules (applied in order):
    - Horizontal bar chart: 40px per category, min 300, max 800
    - Heatmap: 35px per y-tick, min 350, max 900
    - Everything else: 420px (compact default that suits most line/scatter/bar charts)
    """
    traces = fig.data
    if not traces:
        return 420

    first = traces[0]

    # Horizontal bar chart
    if getattr(first, "orientation", None) == "h":
        y_vals = getattr(first, "y", None)
        n = len(y_vals) if y_vals is not None else 5
        return max(300, min(800, n * 40 + 80))

    # Heatmap
    if isinstance(first, go.Heatmap):
        y_vals = getattr(first, "y", None)
        n = len(y_vals) if y_vals is not None else 8
        return max(350, min(900, n * 35 + 80))

    # Vertical bar — scale slightly with category count
    if isinstance(first, go.Bar):
        x_vals = getattr(first, "x", None)
        n = len(x_vals) if x_vals is not None else 5
        return max(350, min(600, 300 + n * 15))

    return 420


def _build_summary(stdout: str, chart_count: int, error: str | None) -> str:
    if error:
        lines = error.strip().splitlines()
        truncated = "\n".join(lines[-20:]) if len(lines) > 20 else error.strip()
        return f"ERROR:\n{truncated}"

    parts: list[str] = []
    if stdout.strip():
        out = stdout.strip()
        if len(out) > 2000:
            out = out[:2000] + "\n... (truncated)"
        parts.append(out)
    if chart_count:
        parts.append(f"[{chart_count} interactive chart(s) generated and displayed to the user]")
    if not parts:
        parts.append("Code executed successfully with no output.")
    return "\n".join(parts)
