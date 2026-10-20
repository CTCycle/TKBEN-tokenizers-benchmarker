# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache
from threading import RLock
from typing import Any, Protocol

from server.common.utils.logger import logger
from server.configurations import DatabaseSettings, ServerSettings, get_server_settings
from server.repositories.database.postgres import PostgresRepository
from server.repositories.database.sqlite import SQLiteRepository

###############################################################################
class DatabaseBackend(Protocol):
    engine: Any
    session_factory: Any

    # -------------------------------------------------------------------------
    def insert_records(
        self,
        table_name: str,
        records: Sequence[Mapping[str, Any]],
        *,
        ignore_duplicates: bool = False,
    ) -> None: ...

    # -------------------------------------------------------------------------
    def upsert_records(
        self,
        table_name: str,
        records: Sequence[Mapping[str, Any]],
        conflict_columns: list[str],
    ) -> None: ...

###############################################################################
def build_sqlite_backend(settings: DatabaseSettings) -> DatabaseBackend:
    return SQLiteRepository(settings)

###############################################################################
def build_postgres_backend(settings: DatabaseSettings) -> DatabaseBackend:
    return PostgresRepository(settings)

###############################################################################
class TKBENDatabase:

    # -------------------------------------------------------------------------
    def __init__(
        self, settings: DatabaseSettings | ServerSettings | None = None
    ) -> None:
        self._lock = RLock()
        self.settings = (
            settings.database
            if isinstance(settings, ServerSettings)
            else settings or get_server_settings().database
        )
        if self.settings.embedded_database:
            self.backend = build_sqlite_backend(self.settings)
        else:
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

    def activate(self, candidate: "TKBENDatabase") -> tuple[DatabaseSettings, DatabaseBackend]:
        """Atomically point all shared repositories at a validated backend."""
        with self._lock:
            previous = (self.settings, self.backend)
            self.settings = candidate.settings
            self.backend = candidate.backend
            return previous

    def restore(self, previous: tuple[DatabaseSettings, DatabaseBackend]) -> None:
        with self._lock:
            self.settings, self.backend = previous

###############################################################################
@cache
def get_database() -> TKBENDatabase:
    return TKBENDatabase()
