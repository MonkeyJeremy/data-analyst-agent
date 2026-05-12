import ast
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

from src.execution.backend import ExecutionBackend
from src.execution.result import ExecutionResult

matplotlib.use("Agg")  # non-interactive backend; must be set before any plt use

# ── Security: blocked imports ─────────────────────────────────────────────────
# Modules that could access the filesystem, network, processes, or memory.
# Checked via AST parse before exec() so the code never runs.
_BLOCKED_MODULES: frozenset[str] = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "importlib", "ctypes", "pickle", "marshal", "shelve",
    "builtins", "__builtin__", "tempfile", "glob", "fnmatch",
    "multiprocessing", "threading", "asyncio", "concurrent",
    "pty", "tty", "termios", "fcntl", "signal", "resource",
    "mmap", "struct", "cffi", "sysconfig",
})


def _check_imports(code: str) -> str | None:
    """Return an error string if *code* imports a blocked module, else ``None``.

    Uses :mod:`ast` to inspect the parse tree — the code is never executed.
    Also catches :exc:`SyntaxError` and returns it as an error string so the
    caller gets a clean message rather than an exception.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BLOCKED_MODULES:
                    return (
                        f"SecurityError: import '{alias.name}' is not permitted "
                        "in the analysis sandbox."
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _BLOCKED_MODULES:
                    return (
                        f"SecurityError: 'from {node.module} import ...' is not "
                        "permitted in the analysis sandbox."
                    )
    return None


def execute_python(code: str, dataframes: dict[str, pd.DataFrame]) -> ExecutionResult:
    """Execute code against copies of all dataframes in a sandboxed namespace.

    Pre-imported names available in the namespace:
      All DataFrames by their registered names, plus ``df`` as alias for the
      primary table (for backward compatibility).
      Also available: pd, np, plt, sns, go (plotly.graph_objects), px (plotly.express).
    stdout and stderr are captured.
    Plotly figures (preferred) are serialised to JSON; matplotlib figures fall
    back to PNG bytes. The original dataframes are never mutated.
    """
    import plotly.express as px  # local import keeps top-level fast

    namespace: dict = {name: df.copy() for name, df in dataframes.items()}
    namespace.update({"pd": pd, "np": np, "plt": plt, "sns": sns, "go": go, "px": px})
    stdout_buf = io.StringIO()
    mpl_figures: list[bytes] = []
    plotly_figures: list[str] = []
    error: str | None = None

    plt.close("all")

    # ── Security pre-check ────────────────────────────────────────────────────
    import_error = _check_imports(code)
    if import_error:
        return ExecutionResult(
            stdout="",
            error=import_error,
            figures=(),
            plotly_figures=(),
            summary=f"ERROR:\n{import_error}",
        )

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
    """Mutate a Plotly figure in-place:

    - Transparent canvas (blends with any Streamlit theme)
    - Width computed from bar count so each bar has a fixed pixel footprint
    - Height computed from category count so charts are never squashed
    - Bar text labels removed — values appear only on hover
    """
    height, width = _compute_dimensions(fig)

    # autosize=False + explicit width gives a fixed-size canvas; Streamlit will
    # centre it if narrower than the column.
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        autosize=False,
        width=width,
        height=height,
        margin={"t": 60, "b": 60, "l": 70, "r": 30},
        font={"size": 13},
        hoverlabel={"font_size": 13, "namelength": -1},  # -1 = never truncate hover text
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.3)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.3)")

    # Narrow bars: large gap so bars occupy ~30% of their slot
    has_bars = any(isinstance(t, go.Bar) for t in fig.data)
    if has_bars:
        fig.update_layout(bargap=0.55, bargroupgap=0.1)

    # Strip bar labels — show values on hover only
    for trace in fig.data:
        if isinstance(trace, go.Bar):
            trace.update(text=None, texttemplate="")


def _compute_dimensions(fig: go.Figure) -> tuple[int, int]:
    """Return (height, width) in pixels sized to the figure's data.

    Bar charts:
      Vertical   — width = 120px × n_bars + margins; height fixed at 420
      Horizontal — height = 40px × n_bars + margins; width fixed at 650
    Heatmap      — height = 35px × n_rows + margins; width fixed at 750
    Everything else — 420 × 750 (reasonable default, autosize via Streamlit)
    """
    traces = fig.data
    if not traces:
        return 420, 750

    first = traces[0]

    # Horizontal bar chart
    if getattr(first, "orientation", None) == "h":
        y_vals = getattr(first, "y", None)
        n = len(y_vals) if y_vals is not None else 5
        return max(300, min(800, n * 40 + 100)), 650

    # Heatmap
    if isinstance(first, go.Heatmap):
        y_vals = getattr(first, "y", None)
        n = len(y_vals) if y_vals is not None else 8
        return max(350, min(900, n * 35 + 100)), 750

    # Vertical bar — 60 px per bar keeps bars narrow; bargap set separately
    if isinstance(first, go.Bar):
        x_vals = getattr(first, "x", None)
        n = len(x_vals) if x_vals is not None else 5
        width = max(280, min(1000, n * 60 + 130))
        return 400, width

    # Scatter, line, pie, etc. — use a comfortable fixed size
    return 420, 750


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


class LocalPythonExecutor(ExecutionBackend):
    """In-process Python executor with AST-level import sandboxing.

    Delegates to :func:`execute_python` so all existing safety logic is reused.
    """

    @property
    def name(self) -> str:
        return "local"

    def execute(self, code: str, dataframes: dict[str, pd.DataFrame]) -> ExecutionResult:
        return execute_python(code, dataframes)
