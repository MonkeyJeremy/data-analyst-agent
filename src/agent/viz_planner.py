"""Visualization intent classifier.

Inspects the user's question and returns a :class:`VizPlan` describing
what kind of chart (if any) would best answer it.  Uses keyword matching
only — zero LLM calls, zero latency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VIZ_INTENT_MAP: dict[str, str] = {
    "trend": "line chart (x = time, y = metric; sort by date ascending)",
    "distribution": "histogram or box plot",
    "relationship": "scatter plot",
    "ranking": "horizontal bar chart sorted descending",
    "composition": "stacked bar chart (use pie only if ≤ 6 categories)",
    "correlation": "heatmap",
    "comparison": "grouped bar chart",
}

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "trend": [
        "over time", "by month", "by year", "by week", "by day", "by quarter",
        "trend", "trends", "time series", "timeline", "monthly", "yearly",
        "weekly", "daily", "quarterly", "growth", "over the", "across time",
    ],
    "distribution": [
        "distribution", "spread", "histogram", "how are", "range",
        "percentile", "quartile", "variance", "std", "standard deviation",
        "normal", "skew", "skewed", "outlier", "outliers",
    ],
    "relationship": [
        "vs ", "versus", "correlation between", "relationship between",
        "scatter", "related to", "affect", "impact", "influence",
        "correlate", "association",
    ],
    "ranking": [
        "top ", "top-", "bottom ", "rank", "ranked", "ranking", "most ",
        "least ", "highest", "lowest", "largest", "smallest", "best",
        "worst", "leader", "leaderboard",
    ],
    "composition": [
        "breakdown", "share", "proportion", "percent of", "percentage of",
        "pie", "composition", "make up", "made up", "consists", "portion",
        "fraction",
    ],
    "correlation": [
        "corr", "heatmap", "heat map", "correlation matrix",
        "which columns correlate", "correlation between all",
    ],
    "comparison": [
        "compare", "comparison", "difference between",
        "side by side", "compared to", "relative to", "against",
    ],
}


@dataclass(frozen=True)
class VizPlan:
    intent: str | None
    recommended_chart: str | None
    output_type: Literal["chart", "table", "both"]
    reasoning: str


def plan_visualization(question: str) -> VizPlan:
    """Return a :class:`VizPlan` for the given *question* via keyword matching."""
    q = question.lower()
    output_type = _infer_output_type(q)

    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return VizPlan(
                intent=intent,
                recommended_chart=VIZ_INTENT_MAP[intent],
                output_type=output_type,
                reasoning=f"Question matches '{intent}' pattern.",
            )

    return VizPlan(
        intent=None,
        recommended_chart=None,
        output_type=output_type,
        reasoning="No specific visualization intent detected.",
    )


def _infer_output_type(q: str) -> Literal["chart", "table", "both"]:
    chart_words = {"chart", "plot", "graph", "visuali", "show me a", "draw"}
    table_words = {"table", "list", "show me the", "what are the", "count", "how many", "sum of"}
    has_chart = any(w in q for w in chart_words)
    has_table = any(w in q for w in table_words)
    if has_chart and has_table:
        return "both"
    if has_chart:
        return "chart"
    if has_table:
        return "table"
    return "both"


def build_viz_hint(plan: VizPlan) -> str:
    """Return a system-prompt snippet from *plan*, or empty string if no intent."""
    if plan.intent is None:
        return ""
    return (
        f"\nVISUALIZATION HINT:\n"
        f"The user's question suggests a '{plan.intent}' analysis.\n"
        f"Recommended chart type: {plan.recommended_chart}.\n"
        f"Preferred output: {plan.output_type}.\n"
        f"If generating a chart, follow this recommendation unless the data clearly calls for something different."
    )
