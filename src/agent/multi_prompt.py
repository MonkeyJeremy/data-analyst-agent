"""System prompt builder for multi-dataframe sessions."""
from __future__ import annotations

from src.data.join_detector import JoinSuggestion
from src.data.registry import DataFrameRegistry

_MAX_EDA_CHARS = 500


def build_multi_dataframe_prompt(
    registry: DataFrameRegistry,
    join_suggestions: list[JoinSuggestion],
    text_cols_by_table: dict[str, tuple[str, ...]] | None = None,
    viz_hint: str = "",
) -> str:
    entries = registry.entries()
    primary = registry.primary()
    text_cols_by_table = text_cols_by_table or {}

    lines: list[str] = [
        "You are an expert data analyst. "
        "You have access to the following DataFrames in your Python environment:\n",
    ]

    for entry in entries:
        alias = " (`df` is an alias for this table)" if entry is primary else ""
        lines.append(
            f"**`{entry.name}`**{alias} — "
            f"{entry.schema.n_rows:,} rows × {entry.schema.n_cols} columns"
        )
        lines.append(entry.schema.formatted_dtypes)
        lines.append(f"Sample:\n{entry.schema.head_markdown}\n")
        if entry.eda.narrative:
            narrative = entry.eda.narrative[:_MAX_EDA_CHARS]
            if len(entry.eda.narrative) > _MAX_EDA_CHARS:
                narrative += "..."
            lines.append(f"*EDA:* {narrative}\n")
        text_cols = text_cols_by_table.get(entry.name, ())
        if text_cols:
            lines.append(
                f"Free-text columns (use `analyze_text` tool): "
                f"{', '.join(f'`{c}`' for c in text_cols)}\n"
            )

    if join_suggestions:
        lines.append("## Relationships\n")
        for j in join_suggestions:
            if j.source == "manual":
                tag = "user-defined"
            else:
                tag = f"{int(j.match_rate * 100)}% match"
            lines.append(
                f"- `{j.left_table}.{j.left_col}` → "
                f"`{j.right_table}.{j.right_col}` "
                f"({tag}, `{j.join_type}` join)"
            )
        lines.append("\nExample join:")
        lines.append(f"```python\nmerged = {join_suggestions[0].example_code()}\n```\n")

    if viz_hint:
        lines.append(f"Visualisation hint: {viz_hint}\n")

    lines.append(
        "## Instructions\n"
        "- Use `execute_python` for all analysis and visualisation.\n"
        "- Reference each DataFrame by its exact name shown above.\n"
        "- `df` is always an alias for the first-loaded table.\n"
        "- Use Plotly (`px`, `go`) for charts; matplotlib as fallback only.\n"
        "- After each code result, explain findings in plain language.\n"
    )

    return "\n".join(lines)
