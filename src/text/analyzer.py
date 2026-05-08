"""Nested-Claude text labelling — no external NLP libraries required.

The outer agent decides *what* to analyse; this module calls Claude again with
a focused labelling prompt and returns a structured markdown table.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from src.execution.result import ExecutionResult

if TYPE_CHECKING:
    pass  # LLMClient imported lazily to avoid circular imports

_MAX_TEXTS = 50

_ANALYSIS_SYSTEM = (
    "You are a text analysis expert. "
    "Always return a valid JSON array exactly as instructed. "
    "No prose, no markdown fences."
)


def analyze_text_batch(
    client: Any,
    texts: list[str],
    task: str,
) -> ExecutionResult:
    """Analyse up to *_MAX_TEXTS* texts via a nested Claude call.

    Parameters
    ----------
    client:
        An :class:`~src.agent.client.LLMClient` instance (or compatible fake).
    texts:
        List of strings to label.  Capped at :data:`_MAX_TEXTS`.
    task:
        Natural-language description of what to analyse, e.g. ``"sentiment"``.

    Returns
    -------
    ExecutionResult
        ``summary`` contains a markdown table; ``error`` is ``None`` on success.
    """
    texts = [str(t) for t in texts[:_MAX_TEXTS]]

    if not texts:
        return ExecutionResult(
            stdout="",
            error="No texts provided.",
            figures=(),
            summary="ERROR: empty input.",
        )

    numbered = "\n".join(f"{i + 1}. {t[:500]}" for i, t in enumerate(texts))
    prompt = (
        f"Task: {task}\n\n"
        f"Texts:\n{numbered}\n\n"
        "Return a JSON array with one object per text (same order):\n"
        '  {"index": <1-based int>, "label": "<result>", '
        '"confidence": "high"|"medium"|"low", "note": "<≤10 words>"}\n\n'
        "Return ONLY the JSON array."
    )

    try:
        response = client.call(
            system=_ANALYSIS_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            tools=[],
        )
        raw = "\n".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()

        # Strip any accidental markdown fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("` \n")

        results: list[dict] = json.loads(raw)

    except Exception as exc:  # noqa: BLE001
        return ExecutionResult(
            stdout="",
            error=str(exc),
            figures=(),
            summary=f"ERROR: {exc}",
        )

    # ── Build markdown table ──────────────────────────────────────────────────
    header = "| # | Text (preview) | Label | Confidence | Note |"
    sep = "|---|----------------|-------|------------|------|"
    rows: list[str] = []
    for i, (orig, r) in enumerate(zip(texts, results)):
        preview = orig[:45].replace("|", "\\|") + ("…" if len(orig) > 45 else "")
        rows.append(
            f"| {i + 1} | {preview} | {r.get('label', '')} "
            f"| {r.get('confidence', '')} | {r.get('note', '')} |"
        )

    table = "\n".join([header, sep] + rows)
    summary = f"Analysed {len(texts)} texts — task: *{task}*\n\n{table}"

    if len(texts) == _MAX_TEXTS:
        summary += (
            f"\n\n*Note: capped at {_MAX_TEXTS} texts. "
            "Call again with a different slice for more.*"
        )

    return ExecutionResult(stdout=summary, error=None, figures=(), summary=summary)
