"""DataFrameRegistry — ordered collection of loaded DataFrames with their metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from src.data.schema import SchemaContext, describe_schema
from src.eda.auto_eda import run_auto_eda
from src.eda.report import EDAReport


def _to_identifier(filename: str) -> str:
    """Convert a filename to a valid Python identifier.

    "Sales Data 2024.csv" → "sales_data_2024"
    """
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    identifier = re.sub(r"[^a-zA-Z0-9]", "_", stem).lower()
    identifier = re.sub(r"_+", "_", identifier).strip("_")
    if not identifier or identifier[0].isdigit():
        identifier = "df_" + identifier
    return identifier or "df"


@dataclass(frozen=True)
class DataFrameEntry:
    name: str
    df: pd.DataFrame
    schema: SchemaContext
    eda: EDAReport
    filename: str


class DataFrameRegistry:
    """Ordered dict of all loaded DataFrames, keyed by safe identifier name."""

    def __init__(self) -> None:
        self._entries: dict[str, DataFrameEntry] = {}

    def add(self, filename: str, df: pd.DataFrame) -> DataFrameEntry:
        base_name = _to_identifier(filename)
        name = base_name
        counter = 1
        while name in self._entries:
            name = f"{base_name}_{counter}"
            counter += 1
        entry = DataFrameEntry(
            name=name,
            df=df,
            schema=describe_schema(df),
            eda=run_auto_eda(df),
            filename=filename,
        )
        self._entries[name] = entry
        return entry

    def remove(self, name: str) -> None:
        self._entries.pop(name, None)

    def get(self, name: str) -> DataFrameEntry | None:
        return self._entries.get(name)

    def names(self) -> list[str]:
        return list(self._entries.keys())

    def entries(self) -> list[DataFrameEntry]:
        return list(self._entries.values())

    def primary(self) -> DataFrameEntry | None:
        return next(iter(self._entries.values()), None)

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def count(self) -> int:
        return len(self._entries)

    def as_namespace(self) -> dict[str, pd.DataFrame]:
        """All dfs by name + df = primary for backward compat."""
        ns: dict[str, pd.DataFrame] = {e.name: e.df.copy() for e in self._entries.values()}
        if primary := self.primary():
            ns["df"] = primary.df.copy()
        return ns

    def filenames(self) -> list[str]:
        return [e.filename for e in self._entries.values()]
