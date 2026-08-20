from __future__ import annotations

from typing import Any

import sqlalchemy
from sqlalchemy import event
from sqlalchemy.dialects.sqlite import insert

from server.common.path import DATABASE_PATH
from server.configurations import DatabaseSettings
from server.repositories.database.base import RepositoryBase
from server.repositories.database.utils import normalize_sqlite_path

###############################################################################
class SQLiteRepository(RepositoryBase):
    SQLITE_MAX_VARIABLES = 900

    # -------------------------------------------------------------------------
    def __init__(
        self,
        settings: DatabaseSettings,
        *,
        enforce_foreign_keys: bool = True,
        begin_immediate: bool = False,
    ) -> None:
        self.db_path = normalize_sqlite_path(DATABASE_PATH)
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        engine = sqlalchemy.create_engine(
            f"sqlite:///{self.db_path}",
            future=True,
            connect_args={
                "autocommit": False,
                "timeout": settings.connect_timeout,
            },
        )
        if enforce_foreign_keys:
            event.listen(engine, "connect", self.enable_foreign_keys)
        if begin_immediate:
            event.listen(engine, "begin", self.begin_immediate)
        super().__init__(engine, settings.insert_batch_size)

    # -------------------------------------------------------------------------
    @staticmethod
    def enable_foreign_keys(dbapi_connection: Any, connection_record: Any) -> None:
        previous_autocommit = getattr(dbapi_connection, "autocommit", None)
        if previous_autocommit is not None:
            dbapi_connection.autocommit = True
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()
            if previous_autocommit is not None:
                dbapi_connection.autocommit = previous_autocommit

    # -------------------------------------------------------------------------
    @staticmethod
    def begin_immediate(connection: Any) -> None:
        # Python's modern sqlite3 transaction mode opens an initial deferred
        # transaction as soon as a connection is created.  Temporarily switch
        # the DB-API connection to autocommit so the explicit writer lock can
        # replace that deferred transaction without a nested-BEGIN error.
        dbapi_connection = connection.connection.driver_connection
        previous_autocommit = dbapi_connection.autocommit
        dbapi_connection.autocommit = True
        try:
            dbapi_connection.execute("BEGIN IMMEDIATE")
        finally:
            dbapi_connection.autocommit = previous_autocommit

    # -------------------------------------------------------------------------
    def _insert(self, table, records, *, ignore_duplicates: bool) -> None:  # type: ignore[no-untyped-def]
        with self.session_factory() as session:
            try:
                for batch in self._batches(records):
                    statement = insert(table).values(batch)
                    if ignore_duplicates:
                        statement = statement.on_conflict_do_nothing()
                    session.execute(statement)
                session.commit()
            except Exception:
                session.rollback()
                raise

    # -------------------------------------------------------------------------
    def _upsert(self, table, records, conflict_columns) -> None:  # type: ignore[no-untyped-def]
        with self.session_factory() as session:
            try:
                for batch in self._batches(records):
                    statement = insert(table).values(batch)
                    updates = {column: getattr(statement.excluded, column) for column in batch[0] if column not in conflict_columns}
                    statement = statement.on_conflict_do_update(index_elements=conflict_columns, set_=updates) if updates else statement.on_conflict_do_nothing(index_elements=conflict_columns)
                    session.execute(statement)
                session.commit()
            except Exception:
                session.rollback()
                raise
