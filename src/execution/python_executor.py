import io
import traceback
from contextlib import redirect_stderr, redirect_stdout

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.execution.result import ExecutionResult

matplotlib.use("Agg")  # non-interactive backend; must be set before any plt use


def execute_python(code: str, df: pd.DataFrame) -> ExecutionResult:
    """Execute code against a copy of df in a sandboxed namespace.

    Pre-imported names available in the namespace: df, pd, np, plt, sns.
    stdout and stderr are captured. All matplotlib figures are serialised to
    PNG bytes. The original df is never mutated.
    """
    namespace: dict = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
    }
    stdout_buf = io.StringIO()
    figures: list[bytes] = []
    error: str | None = None

    plt.close("all")

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
            exec(compile(code, "<agent>", "exec"), namespace)  # noqa: S102

        for fig_num in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(fig_num).savefig(buf, format="png", bbox_inches="tight", dpi=100)
            figures.append(buf.getvalue())
    except Exception:
        error = traceback.format_exc(limit=5)
    finally:
        plt.close("all")

    return ExecutionResult(
        stdout=stdout_buf.getvalue(),
        error=error,
        figures=tuple(figures),
        summary=_build_summary(stdout_buf.getvalue(), figures, error),
    )


def _build_summary(stdout: str, figures: list[bytes], error: str | None) -> str:
    if error:
        # Truncate long tracebacks so Claude's context doesn't overflow
        lines = error.strip().splitlines()
        truncated = "\n".join(lines[-20:]) if len(lines) > 20 else error.strip()
        return f"ERROR:\n{truncated}"

    parts: list[str] = []
    if stdout.strip():
        out = stdout.strip()
        if len(out) > 2000:
            out = out[:2000] + "\n... (truncated)"
        parts.append(out)
    if figures:
        parts.append(f"[{len(figures)} chart(s) generated and displayed to the user]")
    if not parts:
        parts.append("Code executed successfully with no output.")
    return "\n".join(parts)
