from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    error: str | None
    figures: tuple[bytes, ...]  # PNG bytes, one per matplotlib figure
    summary: str                # short string sent back to Claude as tool_result
