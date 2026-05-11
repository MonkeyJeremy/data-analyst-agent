from __future__ import annotations

from src.data.schema import SchemaContext


def build_system_prompt(
    schema: SchemaContext,
    eda_summary: str | None = None,
    text_cols: tuple[str, ...] = (),
    viz_hint: str = "",
) -> str:
    """Build the system prompt for DataFrame mode (execute_python tool)."""
    prompt = f"""\
You are a senior data analyst pair-programming with a user.

YOUR DATAFRAME:
- Variable: `df`
- Shape: {schema.n_rows} rows × {schema.n_cols} columns
- Columns and dtypes:
{schema.formatted_dtypes}

FIRST 5 ROWS:
{schema.head_markdown}

NUMERIC SUMMARY:
{schema.describe_markdown}

INSTRUCTIONS:
1. For any computation or aggregation, call execute_python. Never guess numbers.
2. After tool results return, interpret them in plain English for the user.
3. For charts, use plotly express (px) or plotly graph_objects (go) — assign the figure to a variable (e.g. `fig = px.bar(...)`). Always set a descriptive title and axis labels. Plotly is strongly preferred over matplotlib for all visualisations.
4. If a question is ambiguous, ask for clarification instead of guessing.
5. One analytical step per tool call.
6. If a tool returns an error, read the traceback, fix the code, and retry (max 2 retries).
7. Never reference column names that are not listed above.\
"""

    if eda_summary:
        prompt += f"\n\nPRE-COMPUTED EDA INSIGHTS:\n{eda_summary[:4000]}"

    if text_cols:
        cols_str = ", ".join(f"'{c}'" for c in text_cols)
        prompt += f"""

TEXT ANALYSIS:
The following columns contain free-form text: {cols_str}
You have access to the analyze_text tool. Use it when the user asks about sentiment,
topics, tone, intent, or any text classification task.
Workflow: sample the column first with df['col'].dropna().head(30).tolist(), then call
analyze_text with those texts and a clear task description."""

    if viz_hint:
        prompt += viz_hint

    return prompt


def build_sql_system_prompt(schemas: tuple) -> str:
    """Build the system prompt for SQL mode (execute_sql tool).

    Parameters
    ----------
    schemas:
        Tuple of :class:`~src.db.schema.TableSchema` objects describing the
        connected database.

    Returns
    -------
    str
        System prompt instructing the agent to use ``execute_sql``.
    """
    # Build table descriptions
    table_lines: list[str] = []
    for ts in schemas:
        col_desc = ", ".join(
            f"{c.name} ({c.dtype})" for c in ts.columns
        )
        row_info = f"{ts.row_count:,} rows" if ts.row_count >= 0 else "unknown rows"
        table_lines.append(f"  • {ts.name} ({row_info}): {col_desc}")

    tables_block = "\n".join(table_lines) if table_lines else "  (no tables found)"

    prompt = f"""\
You are a senior data analyst pair-programming with a user.

CONNECTED DATABASE — {len(schemas)} table(s):
{tables_block}

INSTRUCTIONS:
1. For any query, aggregation, or data retrieval, call execute_sql. Never guess values.
2. Only SELECT, WITH (CTEs), and EXPLAIN statements are permitted — the tool will block anything else.
3. After tool results return, interpret them in plain English for the user.
4. If the user asks for a chart or visualisation, describe what the data shows and suggest they export to CSV if they need a visual.
5. If a question is ambiguous, ask for clarification instead of guessing.
6. One analytical query per tool call.
7. If a tool returns an error, read the message, fix the SQL, and retry (max 2 retries).
8. Never reference table or column names that are not listed above.
9. Use proper SQL quoting for table and column names (double-quotes for identifiers).
10. Do not attempt to modify, insert, delete, or drop any data or schema objects.\
"""
    return prompt
