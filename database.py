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

PostgreSQL connections come from one process-wide `psycopg_pool` pool, so a
chart render no longer pays a new TCP/TLS/auth handshake per store call.
`ensure_schema` runs each store's idempotent DDL once per process and target
instead of before every read and write.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from app_logging import bac_log_kv
from runtime_config import DATABASE_POOL_SIZE, DATABASE_URL


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


_POSTGRES_POOL: Any | None = None
_POSTGRES_POOL_LOCK = threading.Lock()


def _postgres_pool() -> Any | None:
    """Return the process-wide pool, or None when psycopg_pool is unavailable."""
    global _POSTGRES_POOL
    if _POSTGRES_POOL is not None:
        return _POSTGRES_POOL
    try:
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
    except ImportError:
        return None
    with _POSTGRES_POOL_LOCK:
        if _POSTGRES_POOL is None:
            _POSTGRES_POOL = ConnectionPool(
                DATABASE_URL,
                min_size=1,
                max_size=DATABASE_POOL_SIZE,
                kwargs={"row_factory": dict_row, "connect_timeout": 10},
                # Validate idle connections so a PostgreSQL restart does not
                # surface as a failed request on the next checkout.
                check=ConnectionPool.check_connection,
                name="stock-market",
                open=True,
            )
            bac_log_kv("database.pool", status="opened", max_size=DATABASE_POOL_SIZE)
    return _POSTGRES_POOL


def _uses_postgres(explicit_sqlite_path: str | Path | None) -> bool:
    return bool(DATABASE_URL) and explicit_sqlite_path is None


@contextmanager
def database_connection(
    default_sqlite_path: str | Path,
    explicit_sqlite_path: str | Path | None = None,
):
    """Open the configured database and commit or roll back one unit of work.

    An explicit path always selects SQLite.  Tests rely on this rule to isolate
    each case even if a developer happens to have DATABASE_URL set globally.
    """
    if _uses_postgres(explicit_sqlite_path):
        pool = _postgres_pool()
        if pool is not None:
            # The pool commits on a clean exit, rolls back on an exception,
            # and returns the connection for reuse either way.
            with pool.connection() as raw_connection:
                yield DatabaseConnection(raw_connection, "postgresql")
            return

        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as ex:  # pragma: no cover - production dependency guard
            raise RuntimeError(
                "DATABASE_URL requires the 'psycopg[binary,pool]' package."
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


_INITIALIZED_SCHEMAS: set[tuple[str, ...]] = set()
_SCHEMA_LOCK = threading.Lock()


def ensure_schema(
    store_name: str,
    default_sqlite_path: str | Path,
    explicit_sqlite_path: str | Path | None,
    initialize: Callable[[], object],
) -> None:
    """Run a store's idempotent DDL once per process and database target.

    A SQLite file that disappears while the process runs (for example a
    cleared ``data/`` directory) is initialized again on its next use.
    """
    if _uses_postgres(explicit_sqlite_path):
        key: tuple[str, ...] = (store_name, "postgresql", DATABASE_URL)
        sqlite_path: Path | None = None
    else:
        sqlite_path = Path(explicit_sqlite_path or default_sqlite_path).resolve()
        key = (store_name, "sqlite", str(sqlite_path))

    def already_initialized() -> bool:
        return key in _INITIALIZED_SCHEMAS and (
            sqlite_path is None or sqlite_path.exists()
        )

    if already_initialized():
        return
    with _SCHEMA_LOCK:
        if already_initialized():
            return
        initialize()
        _INITIALIZED_SCHEMAS.add(key)
