from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Protocol

import pandas as pd

from TKBEN.server.common.constants import DATABASE_FILENAME, RESOURCES_PATH
from TKBEN.server.configurations import DatabaseSettings, get_server_settings
from TKBEN.server.repositories.database.postgres import PostgresRepository
from TKBEN.server.repositories.database.sqlite import SQLiteRepository
from TKBEN.server.repositories.database.utils import normalize_sqlite_path
from TKBEN.server.common.utils.logger import logger


###############################################################################
class DatabaseBackend(Protocol):
    db_path: str | None
    engine: Any

    # -------------------------------------------------------------------------
    def load_from_database(self, table_name: str) -> pd.DataFrame: ...

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None: ...

    # -------------------------------------------------------------------------
    def insert_dataframe(
        self, df: pd.DataFrame, table_name: str, ignore_duplicates: bool = True
    ) -> None: ...


BackendFactory = Callable[[DatabaseSettings], DatabaseBackend]


# -----------------------------------------------------------------------------
def build_sqlite_backend(settings: DatabaseSettings) -> DatabaseBackend:
    db_path = normalize_sqlite_path(os.path.join(RESOURCES_PATH, DATABASE_FILENAME))
    should_initialize_schema = not os.path.exists(db_path)
    return SQLiteRepository(settings, initialize_schema=should_initialize_schema)


# -----------------------------------------------------------------------------
def build_postgres_backend(settings: DatabaseSettings) -> DatabaseBackend:
    return PostgresRepository(settings)


BACKEND_FACTORIES: dict[str, BackendFactory] = {
    "sqlite": build_sqlite_backend,
    "postgres": build_postgres_backend,
}


###############################################################################
class TKBENDatabase:
    def __init__(self) -> None:
        self.settings = get_server_settings().database
        self.backend = self._build_backend(self.settings.embedded_database)

    # -------------------------------------------------------------------------
    def _build_backend(self, is_embedded: bool) -> DatabaseBackend:
        if is_embedded:
            backend_name = "sqlite"
        else:
            engine_name = (self.settings.engine or "").lower()
            if engine_name != "postgresql+psycopg":
                raise ValueError(f"Unsupported database engine: {self.settings.engine}")
            backend_name = "postgres"
        normalized_name = backend_name.lower()
        logger.info("Initializing %s database backend", backend_name)
        if normalized_name not in BACKEND_FACTORIES:
            raise ValueError(f"Unsupported database engine: {backend_name}")
        factory = BACKEND_FACTORIES[normalized_name]
        return factory(self.settings)

    # -------------------------------------------------------------------------
    @property
    def db_path(self) -> str | None:
        return getattr(self.backend, "db_path", None)

    # -------------------------------------------------------------------------
    def load_from_database(self, table_name: str) -> pd.DataFrame:
        return self.backend.load_from_database(table_name)

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        self.backend.upsert_into_database(df, table_name)

    # -------------------------------------------------------------------------
    def insert_dataframe(
        self, df: pd.DataFrame, table_name: str, ignore_duplicates: bool = True
    ) -> None:
        """Insert DataFrame rows in batches (append mode, no delete)."""
        self.backend.insert_dataframe(df, table_name, ignore_duplicates)


database = TKBENDatabase()
