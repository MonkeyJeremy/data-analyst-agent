from __future__ import annotations

import pandas as pd

from src.execution.python_executor import execute_python
from src.execution.result import ExecutionResult

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "execute_python",
        "description": (
            "Execute Python code against the user's DataFrame (variable: `df`). "
            "pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns) are "
            "pre-imported. Use print() for text output; use plt.show() or create a "
            "figure for charts. Do NOT re-read the file. Do NOT use input() or network calls."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Valid Python code to execute.",
                },
                "purpose": {
                    "type": "string",
                    "description": "One-sentence explanation shown to the user.",
                },
            },
            "required": ["code", "purpose"],
        },
    }
]


def dispatch_tool(name: str, tool_input: dict, df: pd.DataFrame) -> ExecutionResult:
    """Route a tool call to its implementation.

    Returns an ExecutionResult (error field set for unknown tools).
    """
    if name == "execute_python":
        code = tool_input.get("code", "")
        return execute_python(code, df)

    return ExecutionResult(
        stdout="",
        error=f"Unknown tool: '{name}'",
        figures=(),
        summary=f"ERROR: Unknown tool '{name}'",
    )
