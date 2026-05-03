from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SchemaContext:
    n_rows: int
    n_cols: int
    formatted_dtypes: str   # markdown table: column | dtype
    head_markdown: str      # df.head().to_markdown()
    describe_markdown: str  # df.describe().to_markdown()


def describe_schema(df: pd.DataFrame) -> SchemaContext:
    dtype_rows = "\n".join(
        f"| {col} | {dtype} |" for col, dtype in df.dtypes.items()
    )
    formatted_dtypes = f"| Column | dtype |\n|--------|-------|\n{dtype_rows}"

    try:
        head_md = df.head().to_markdown(index=False)
    except Exception:
        head_md = df.head().to_string()

    try:
        desc_md = df.describe(include="all").to_markdown()
    except Exception:
        desc_md = df.describe(include="all").to_string()

    return SchemaContext(
        n_rows=len(df),
        n_cols=len(df.columns),
        formatted_dtypes=formatted_dtypes,
        head_markdown=head_md or "",
        describe_markdown=desc_md or "",
    )
