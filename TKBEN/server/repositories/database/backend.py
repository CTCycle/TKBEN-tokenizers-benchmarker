from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import pandas as pd

from TKBEN.server.configurations import DatabaseSettings, server_settings
from TKBEN.server.repositories.database.postgres import PostgresRepository
from TKBEN.server.repositories.database.sqlite import SQLiteRepository
from TKBEN.server.repositories.database.utils import normalize_postgres_engine
from TKBEN.server.common.utils.logger import logger


###############################################################################
class DatabaseBackend(Protocol):
    db_path: str | None
    engine: Any

    # -------------------------------------------------------------------------
    def load_from_database(self, table_name: str) -> pd.DataFrame: ...

    # -------------------------------------------------------------------------
    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None: ...

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None: ...

    def bulk_replace_by_key(
        self, df: pd.DataFrame, table_name: str, key_column: str, key_value: str
    ) -> None: ...

    # -------------------------------------------------------------------------
    def delete_by_key(
        self, table_name: str, key_column: str, key_value: str
    ) -> None: ...

    # -------------------------------------------------------------------------
    def insert_dataframe(
        self, df: pd.DataFrame, table_name: str, ignore_duplicates: bool = True
    ) -> None: ...


BackendFactory = Callable[[DatabaseSettings], DatabaseBackend]


# -----------------------------------------------------------------------------
def build_sqlite_backend(settings: DatabaseSettings) -> DatabaseBackend:
    return SQLiteRepository(settings)

# -----------------------------------------------------------------------------
def build_postgres_backend(settings: DatabaseSettings) -> DatabaseBackend:
    return PostgresRepository(settings)


BACKEND_FACTORIES: dict[str, BackendFactory] = {
    "sqlite": build_sqlite_backend,
    "postgres": build_postgres_backend,
}


###############################################################################
class TKBENWebappDatabase:
    def __init__(self) -> None:
        self.settings = server_settings.database
        self.backend = self._build_backend(self.settings.embedded_database)

    # -------------------------------------------------------------------------
    def _build_backend(self, is_embedded: bool) -> DatabaseBackend:
        if is_embedded:
            backend_name = "sqlite"
        else:
            normalized_engine = normalize_postgres_engine(self.settings.engine).lower()
            backend_name = "postgres" if normalized_engine.startswith("postgresql") else normalized_engine
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
    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        self.backend.save_into_database(df, table_name)

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        self.backend.upsert_into_database(df, table_name)

    def bulk_replace_by_key(
        self, df: pd.DataFrame, table_name: str, key_column: str, key_value: str
    ) -> None:
        self.backend.bulk_replace_by_key(df, table_name, key_column, key_value)

    # -------------------------------------------------------------------------
    def delete_by_key(
        self, table_name: str, key_column: str, key_value: str
    ) -> None:
        """Delete all rows matching the given key value."""
        self.backend.delete_by_key(table_name, key_column, key_value)

    # -------------------------------------------------------------------------
    def insert_dataframe(
        self, df: pd.DataFrame, table_name: str, ignore_duplicates: bool = True
    ) -> None:
        """Insert DataFrame rows in batches (append mode, no delete)."""
        self.backend.insert_dataframe(df, table_name, ignore_duplicates)


database = TKBENWebappDatabase()


