"""Post-generation chart validation for Plotly figures.

Validates a rendered figure for common quality issues and returns a
correction prompt that can be injected back into the agent loop so the
LLM can regenerate an improved chart.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    issues: list[str]
    correction_prompt: str = ""


_CORRECTION_TEMPLATE = """\
The generated chart has the following issues: {issues}.
Please regenerate the chart fixing these problems:
- If too many categories: use the top 15 by value and group the rest as "Other".
- If missing axis labels: add descriptive axis titles matching the data.
- If unsorted dates: sort the data by the date column before plotting.
- If a pie chart has too many slices: switch to a bar chart or use top 8 slices + "Other".
- If the title is missing or generic: set a descriptive title that answers the user's question."""

_GENERIC_TITLES = frozenset({"chart", "figure", "plot", "graph", ""})


def validate_figures(plotly_json_list: tuple[str, ...]) -> ValidationResult:
    """Validate a tuple of Plotly JSON strings.

    Returns a single :class:`ValidationResult` covering all figures.
    """
    all_issues: list[str] = []

    for fig_json in plotly_json_list:
        try:
            fig_dict = json.loads(fig_json)
        except (ValueError, TypeError):
            continue
        all_issues.extend(_check_figure(fig_dict))

    if not all_issues:
        return ValidationResult(valid=True, issues=[])

    issues_text = "; ".join(all_issues)
    prompt = _CORRECTION_TEMPLATE.format(issues=issues_text)
    return ValidationResult(valid=False, issues=all_issues, correction_prompt=prompt)


def _check_figure(fig: dict) -> list[str]:
    issues: list[str] = []
    data = fig.get("data", [])
    layout = fig.get("layout", {})

    if not data:
        issues.append("chart is empty (no traces)")
        return issues

    for trace in data:
        trace_type = trace.get("type", "")
        x_vals = trace.get("x") or []
        y_vals = trace.get("y") or []

        if trace_type not in ("pie", "heatmap") and len(x_vals) == 0 and len(y_vals) == 0:
            issues.append("chart trace has no data points")

        if trace_type == "bar":
            n_cats = max(len(x_vals), len(y_vals))
            if n_cats > 30:
                issues.append(
                    f"bar chart has {n_cats} categories (>30); use top 15 + 'Other'"
                )

        if trace_type == "pie":
            labels = trace.get("labels") or []
            if len(labels) > 8:
                issues.append(
                    f"pie chart has {len(labels)} slices (>8); switch to bar or use top 8 + 'Other'"
                )

    has_cartesian = any(t.get("type", "") not in ("pie",) for t in data)
    if has_cartesian:
        xaxis = layout.get("xaxis", {})
        yaxis = layout.get("yaxis", {})
        x_title = xaxis.get("title") or {}
        x_text = x_title.get("text", "") if isinstance(x_title, dict) else str(x_title)
        y_title = yaxis.get("title") or {}
        y_text = y_title.get("text", "") if isinstance(y_title, dict) else str(y_title)
        if not x_text.strip():
            issues.append("x-axis label is missing")
        if not y_text.strip():
            issues.append("y-axis label is missing")

    title_obj = layout.get("title") or {}
    title_text = title_obj.get("text", "") if isinstance(title_obj, dict) else str(title_obj)
    if title_text.strip().lower() in _GENERIC_TITLES:
        issues.append("chart title is missing or too generic")

    return issues
