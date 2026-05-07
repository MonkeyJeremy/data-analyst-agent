"""Tests for src/db/executor.py and src/db/connection.py."""
from __future__ import annotations

import io
import sqlite3
import tempfile
import os

import pytest
from sqlalchemy import create_engine, text

from src.db.connection import SQLConnection, connect_sqlite_file, connect_url
from src.db.executor import execute_sql, load_table
from src.db.schema import describe_sql_schema, TableSchema, ColumnInfo


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sqlite_engine():
    """In-memory SQLite engine with an 'employees' table and a 'departments' table."""
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE TABLE employees "
            "(id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary REAL)"
        ))
        conn.execute(text(
            "INSERT INTO employees VALUES "
            "(1, 'Alice', 'Engineering', 85000.0), "
            "(2, 'Bob', 'Marketing', 62000.0), "
            "(3, 'Carol', 'Engineering', 92000.0)"
        ))
        conn.execute(text(
            "CREATE TABLE departments "
            "(id INTEGER PRIMARY KEY, name TEXT, budget REAL)"
        ))
        conn.execute(text(
            "INSERT INTO departments VALUES "
            "(1, 'Engineering', 500000.0), "
            "(2, 'Marketing', 200000.0)"
        ))
        conn.commit()
    return engine


@pytest.fixture
def sqlite_file_bytes():
    """Bytes of a real SQLite file with one table."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE items (id INTEGER, label TEXT, value REAL)")
    conn.execute("INSERT INTO items VALUES (1, 'x', 3.14)")
    conn.execute("INSERT INTO items VALUES (2, 'y', 2.72)")
    conn.commit()
    conn.close()
    with open(path, "rb") as f:
        data = f.read()
    os.unlink(path)
    return data


# ── execute_sql ───────────────────────────────────────────────────────────────

class TestExecuteSql:
    def test_select_returns_markdown(self, sqlite_engine):
        result = execute_sql(sqlite_engine, "SELECT * FROM employees")
        assert result.error is None
        assert "Alice" in result.summary
        assert "Bob" in result.summary
        # Should contain a markdown table separator
        assert "|" in result.summary

    def test_select_count(self, sqlite_engine):
        result = execute_sql(sqlite_engine, "SELECT COUNT(*) FROM employees")
        assert result.error is None
        assert "3" in result.summary

    def test_select_with_where(self, sqlite_engine):
        result = execute_sql(sqlite_engine, "SELECT name FROM employees WHERE salary > 80000")
        assert result.error is None
        assert "Alice" in result.summary
        assert "Carol" in result.summary
        assert "Bob" not in result.summary

    def test_bad_table_name(self, sqlite_engine):
        result = execute_sql(sqlite_engine, "SELECT * FROM nonexistent_table")
        assert result.error is not None
        assert "ERROR" in result.summary

    def test_non_select_blocked_drop(self, sqlite_engine):
        result = execute_sql(sqlite_engine, "DROP TABLE employees")
        assert result.error is not None
        assert "ERROR" in result.summary
        # Table should still exist
        check = execute_sql(sqlite_engine, "SELECT COUNT(*) FROM employees")
        assert check.error is None

    def test_non_select_blocked_insert(self, sqlite_engine):
        result = execute_sql(sqlite_engine, "INSERT INTO employees VALUES (99, 'Hacker', 'X', 0)")
        assert result.error is not None
        assert "SELECT" in result.summary or "blocked" in result.summary.lower()

    def test_empty_result_set(self, sqlite_engine):
        result = execute_sql(sqlite_engine, "SELECT * FROM employees WHERE salary > 999999")
        assert result.error is None
        assert "0" in result.summary or "no rows" in result.summary.lower()

    def test_cte_with_allowed(self, sqlite_engine):
        query = "WITH top AS (SELECT * FROM employees) SELECT * FROM top"
        result = execute_sql(sqlite_engine, query)
        assert result.error is None
        assert "Alice" in result.summary


# ── describe_sql_schema ───────────────────────────────────────────────────────

class TestDescribeSqlSchema:
    def test_returns_correct_tables(self, sqlite_engine):
        schemas = describe_sql_schema(sqlite_engine)
        names = {s.name for s in schemas}
        assert "employees" in names
        assert "departments" in names

    def test_column_info_populated(self, sqlite_engine):
        schemas = describe_sql_schema(sqlite_engine)
        emp = next(s for s in schemas if s.name == "employees")
        col_names = {c.name for c in emp.columns}
        assert "id" in col_names
        assert "name" in col_names
        assert "salary" in col_names

    def test_row_counts(self, sqlite_engine):
        schemas = describe_sql_schema(sqlite_engine)
        emp = next(s for s in schemas if s.name == "employees")
        dept = next(s for s in schemas if s.name == "departments")
        assert emp.row_count == 3
        assert dept.row_count == 2

    def test_frozen_dataclass(self, sqlite_engine):
        schemas = describe_sql_schema(sqlite_engine)
        s = schemas[0]
        assert isinstance(s, TableSchema)
        with pytest.raises(Exception):  # frozen dataclass — assignment should fail
            s.row_count = 999  # type: ignore[misc]


# ── connect_sqlite_file ───────────────────────────────────────────────────────

class TestConnectSqliteFile:
    def test_returns_sql_connection(self, sqlite_file_bytes):
        conn = connect_sqlite_file(sqlite_file_bytes)
        assert conn.dialect == "sqlite"
        assert "items" in conn.tables
        conn.dispose()

    def test_engine_is_queryable(self, sqlite_file_bytes):
        conn = connect_sqlite_file(sqlite_file_bytes)
        result = execute_sql(conn.engine, "SELECT COUNT(*) FROM items")
        assert result.error is None
        assert "2" in result.summary
        conn.dispose()

    def test_dispose_removes_temp_file(self, sqlite_file_bytes):
        conn = connect_sqlite_file(sqlite_file_bytes)
        path = conn._temp_path
        assert path is not None
        assert os.path.exists(path)
        conn.dispose()
        assert not os.path.exists(path)


# ── connect_url ───────────────────────────────────────────────────────────────

class TestConnectUrl:
    def test_connect_bad_url_raises(self):
        with pytest.raises(ValueError):
            connect_url("postgresql://nonexistent_host:5432/db")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            connect_url("")

    def test_connect_sqlite_url(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn_str = f"sqlite:///{db_path}"
        # Create the db first
        e = create_engine(conn_str)
        with e.connect() as c:
            c.execute(text("CREATE TABLE t (x INTEGER)"))
            c.commit()
        e.dispose()

        conn = connect_url(conn_str)
        assert conn.dialect == "sqlite"
        assert "t" in conn.tables
        conn.dispose()


# ── load_table ────────────────────────────────────────────────────────────────

class TestLoadTable:
    def test_loads_all_rows(self, sqlite_engine):
        df = load_table(sqlite_engine, "employees")
        assert len(df) == 3
        assert "name" in df.columns

    def test_loads_departments(self, sqlite_engine):
        df = load_table(sqlite_engine, "departments")
        assert len(df) == 2
        assert "budget" in df.columns
