from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    error: str | None
    figures: tuple[bytes, ...]        # PNG bytes, one per matplotlib figure (fallback)
    summary: str                      # short string sent back to Claude as tool_result
    plotly_figures: tuple[str, ...] = field(default=())  # Plotly JSON strings (preferred)
