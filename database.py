#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: database.py
#############################

"""Small DB-API compatibility layer for SQLite and PostgreSQL.

SQLite remains the deterministic test/development backend.  When DATABASE_URL
is set, PostgreSQL is used instead, and question-mark placeholders are converted
to psycopg's `%s` form.  Keeping this adapter narrow avoids coupling the domain
stores to an ORM while preserving their existing parameterized SQL.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from app_logging import bac_log_kv
from runtime_config import DATABASE_URL


class DatabaseConnection:
    """Expose the subset of DB-API methods used by the two persistence modules."""

    def __init__(self, connection: Any, backend: str) -> None:
        self._connection = connection
        self.backend = backend

    def _sql(self, statement: str) -> str:
        if self.backend == "postgresql":
            return statement.replace("?", "%s")
        return statement

    def execute(self, statement: str, parameters: Iterable[Any] | None = None):
        return self._connection.execute(
            self._sql(statement),
            tuple(parameters or ()),
        )

    def executemany(self, statement: str, parameter_rows: Iterable[Iterable[Any]]):
        prepared_rows = [tuple(row) for row in parameter_rows]
        if self.backend == "postgresql":
            cursor = self._connection.cursor()
            cursor.executemany(self._sql(statement), prepared_rows)
            return cursor
        return self._connection.executemany(self._sql(statement), prepared_rows)

    def executescript(self, script: str) -> None:
        if self.backend == "sqlite":
            self._connection.executescript(script)
            return

        # The schema scripts contain ordinary DDL statements and no procedural
        # bodies, so splitting on semicolons is safe and works with psycopg.
        for statement in script.split(";"):
            if statement.strip():
                self._connection.execute(statement)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


@contextmanager
def database_connection(
    default_sqlite_path: str | Path,
    explicit_sqlite_path: str | Path | None = None,
):
    """Open the configured database and commit or roll back one unit of work.

    An explicit path always selects SQLite.  Tests rely on this rule to isolate
    each case even if a developer happens to have DATABASE_URL set globally.
    """
    if DATABASE_URL and explicit_sqlite_path is None:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as ex:  # pragma: no cover - production dependency guard
            raise RuntimeError(
                "DATABASE_URL requires the 'psycopg[binary]' package."
            ) from ex

        raw_connection = psycopg.connect(
            DATABASE_URL,
            row_factory=dict_row,
            connect_timeout=10,
        )
        connection = DatabaseConnection(raw_connection, "postgresql")
    else:
        path = Path(explicit_sqlite_path or default_sqlite_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_connection = sqlite3.connect(path, timeout=30)
        raw_connection.row_factory = sqlite3.Row
        raw_connection.execute("PRAGMA journal_mode=WAL")
        raw_connection.execute("PRAGMA busy_timeout=30000")
        connection = DatabaseConnection(raw_connection, "sqlite")

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def configured_database_backend(explicit_sqlite_path: str | Path | None = None) -> str:
    """Return a safe backend label for health captions and BAC_LOG output."""
    backend = "postgresql" if DATABASE_URL and explicit_sqlite_path is None else "sqlite"
    bac_log_kv("database.backend", backend=backend)
    return backend
