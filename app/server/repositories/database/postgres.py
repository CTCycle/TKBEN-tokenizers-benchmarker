from __future__ import annotations

import urllib.parse
from typing import Any

import sqlalchemy
from sqlalchemy.dialects.postgresql import insert

from server.configurations import DatabaseSettings
from server.repositories.database.base import RepositoryBase

###############################################################################
class PostgresRepository(RepositoryBase):

    # -------------------------------------------------------------------------
    def __init__(self, settings: DatabaseSettings) -> None:
        if not settings.host or not settings.database_name or not settings.username:
            raise ValueError("PostgreSQL host, database name, and username are required")
        engine_name = (settings.engine or "").lower()
        if engine_name != "postgresql+psycopg":
            raise ValueError(f"Unsupported database engine: {settings.engine}")
        username = urllib.parse.quote_plus(settings.username)
        password = urllib.parse.quote_plus(settings.password or "")
        connect_args: dict[str, Any] = {"connect_timeout": settings.connect_timeout}
        if settings.ssl:
            connect_args["sslmode"] = "require"
            if settings.ssl_ca:
                connect_args["sslrootcert"] = settings.ssl_ca
        engine = sqlalchemy.create_engine(f"{engine_name}://{username}:{password}@{settings.host}:{settings.port or 5432}/{settings.database_name}", future=True, connect_args=connect_args, pool_pre_ping=True)
        super().__init__(engine, settings.insert_batch_size)

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
