"""SQL database connection helpers.

Supports:
- SQLite file upload (bytes → temp file → SQLAlchemy engine)
- Live SQL connection via URL string (PostgreSQL, MySQL, SQLite, etc.)
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import create_engine, inspect, text


@dataclass
class SQLConnection:
    """An active connection to a SQL database.

    Not frozen because SQLAlchemy engines are stateful resources.

    Attributes
    ----------
    engine:
        SQLAlchemy Engine for running queries.
    dialect:
        Database dialect name, e.g. "sqlite", "postgresql", "mysql".
    tables:
        Names of all tables visible in the default schema.
    _temp_path:
        Path to the temp file written for SQLite file uploads.
        ``None`` for URL-based connections.
    """

    engine: Any
    dialect: str
    tables: tuple[str, ...]
    _temp_path: str | None = field(default=None, repr=False)

    def dispose(self) -> None:
        """Release the engine connection pool and remove any temp files."""
        try:
            self.engine.dispose()
        except Exception:  # noqa: BLE001
            pass
        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.unlink(self._temp_path)
            except Exception:  # noqa: BLE001
                pass


def connect_sqlite_file(file_bytes: bytes) -> SQLConnection:
    """Write *file_bytes* to a temp file and open a SQLite connection.

    Parameters
    ----------
    file_bytes:
        Raw bytes of a .db / .sqlite file.

    Returns
    -------
    SQLConnection
        Live connection with ``dialect="sqlite"`` and the table list populated.
    """
    fd, path = tempfile.mkstemp(suffix=".db")
    try:
        os.write(fd, file_bytes)
    finally:
        os.close(fd)

    engine = create_engine(f"sqlite:///{path}")
    insp = inspect(engine)
    tables = tuple(insp.get_table_names())
    return SQLConnection(engine=engine, dialect="sqlite", tables=tables, _temp_path=path)


def connect_url(connection_string: str) -> SQLConnection:
    """Create a SQLAlchemy engine from *connection_string* and verify connectivity.

    Parameters
    ----------
    connection_string:
        A SQLAlchemy-compatible URL, e.g.
        ``"sqlite:///path/to/db.sqlite"`` or
        ``"postgresql://user:pass@host/db"``.

    Returns
    -------
    SQLConnection
        Live connection with dialect and tables populated.

    Raises
    ------
    ValueError
        If the connection string is empty or if the connection attempt fails.
    """
    if not connection_string.strip():
        raise ValueError("Connection string cannot be empty.")

    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise ValueError(f"Could not connect: {exc}") from exc

    insp = inspect(engine)
    tables = tuple(insp.get_table_names())
    dialect: str = engine.dialect.name  # type: ignore[assignment]
    return SQLConnection(engine=engine, dialect=dialect, tables=tables)
