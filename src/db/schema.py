"""SQL schema introspection — produces TableSchema objects for system-prompt injection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import inspect, text


@dataclass(frozen=True)
class ColumnInfo:
    """Name and SQL type of one column."""

    name: str
    dtype: str


@dataclass(frozen=True)
class TableSchema:
    """Schema information for one database table."""

    name: str
    columns: tuple[ColumnInfo, ...]
    row_count: int


def describe_sql_schema(engine: Any) -> tuple[TableSchema, ...]:
    """Inspect *engine* and return a :class:`TableSchema` for every table.

    Parameters
    ----------
    engine:
        A SQLAlchemy Engine (any supported dialect).

    Returns
    -------
    tuple[TableSchema, ...]
        One entry per table.  Row counts are queried individually; if a count
        query fails the count is reported as -1.
    """
    insp = inspect(engine)
    table_names: list[str] = insp.get_table_names()

    schemas: list[TableSchema] = []
    for table_name in table_names:
        raw_columns = insp.get_columns(table_name)
        columns = tuple(
            ColumnInfo(name=col["name"], dtype=str(col["type"]))
            for col in raw_columns
        )

        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text(f'SELECT COUNT(*) FROM "{table_name}"')
                )
                row_count = int(result.scalar() or 0)
        except Exception:  # noqa: BLE001
            row_count = -1

        schemas.append(TableSchema(name=table_name, columns=columns, row_count=row_count))

    return tuple(schemas)
