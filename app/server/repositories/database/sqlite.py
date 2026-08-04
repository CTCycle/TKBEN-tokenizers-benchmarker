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
    def __init__(self, settings: DatabaseSettings) -> None:
        self.db_path = normalize_sqlite_path(DATABASE_PATH)
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        engine = sqlalchemy.create_engine(f"sqlite:///{self.db_path}", future=True)
        event.listen(engine, "connect", self.enable_foreign_keys)
        super().__init__(engine, settings.insert_batch_size)

    # -------------------------------------------------------------------------
    @staticmethod
    def enable_foreign_keys(dbapi_connection: Any, connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
        finally:
            cursor.close()

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
