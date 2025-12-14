from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import pandas as pd

from TKBEN.server.utils.configurations import DatabaseSettings, server_settings
from TKBEN.server.utils.logger import logger
from TKBEN.server.database.postgres import PostgresRepository
from TKBEN.server.database.schema import Base
from TKBEN.server.database.sqlite import SQLiteRepository


###############################################################################
class DatabaseBackend(Protocol):
    db_path: str | None
    engine: Any

    # -------------------------------------------------------------------------
    def load_from_database(self, table_name: str) -> pd.DataFrame: ...

    # -------------------------------------------------------------------------
    def load_paginated(
        self, table_name: str, offset: int, limit: int
    ) -> pd.DataFrame: ...

    # -------------------------------------------------------------------------
    def get_table_names(self) -> list[str]: ...

    # -------------------------------------------------------------------------
    def get_column_count(self, table_name: str) -> int: ...

    # -------------------------------------------------------------------------
    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None: ...

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None: ...

    # -------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int: ...

    # -------------------------------------------------------------------------
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
        backend_name = "sqlite" if is_embedded else (self.settings.engine or "postgres")
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
    def load_paginated(
        self, table_name: str, offset: int = 0, limit: int = 1000
    ) -> pd.DataFrame:
        return self.backend.load_paginated(table_name, offset, limit)

    # -------------------------------------------------------------------------
    def get_table_names(self) -> list[str]:
        return self.backend.get_table_names()

    # -------------------------------------------------------------------------
    def get_column_count(self, table_name: str) -> int:
        return self.backend.get_column_count(table_name)

    # -------------------------------------------------------------------------
    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        self.backend.save_into_database(df, table_name)

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        self.backend.upsert_into_database(df, table_name)

    # -------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        return self.backend.count_rows(table_name)

    # -------------------------------------------------------------------------
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

