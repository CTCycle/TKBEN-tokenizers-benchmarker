from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any, Protocol

from server.common.path import DATABASE_PATH
from server.common.utils.logger import logger
from server.configurations import DatabaseSettings, get_server_settings
from server.repositories.database.postgres import PostgresRepository
from server.repositories.database.sqlite import SQLiteRepository
from server.repositories.database.utils import normalize_sqlite_path


###############################################################################
class DatabaseBackend(Protocol):
    engine: Any
    session_factory: Any

    # -------------------------------------------------------------------------
    def insert_records(self, table_name: str, records: list[Mapping[str, Any]], *, ignore_duplicates: bool = False) -> None: ...

    # -------------------------------------------------------------------------
    def upsert_records(self, table_name: str, records: list[Mapping[str, Any]], conflict_columns: list[str]) -> None: ...

    # -------------------------------------------------------------------------
    def get_distinct_values(self, table_name: str, column: str) -> list[Any]: ...

    # -------------------------------------------------------------------------
    def validate_schema(self) -> None: ...


###############################################################################
def build_sqlite_backend(settings: DatabaseSettings) -> DatabaseBackend:
    return SQLiteRepository(settings, initialize_schema=not Path(normalize_sqlite_path(DATABASE_PATH)).exists())


###############################################################################
def build_postgres_backend(settings: DatabaseSettings) -> DatabaseBackend:
    return PostgresRepository(settings)


###############################################################################
class TKBENDatabase:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.settings = get_server_settings().database
        if self.settings.embedded_database:
            self.backend = build_sqlite_backend(self.settings)
        else:
            if (self.settings.engine or "").lower() != "postgresql+psycopg":
                raise ValueError(f"Unsupported database engine: {self.settings.engine}")
            self.backend = build_postgres_backend(self.settings)
        logger.info("Initialized database backend")

    # -------------------------------------------------------------------------
    @property
    def engine(self) -> Any:
        return self.backend.engine

    # -------------------------------------------------------------------------
    @property
    def db_path(self) -> str | None:
        return getattr(self.backend, "db_path", None)


###############################################################################
@cache
def get_database() -> TKBENDatabase:
    return TKBENDatabase()
