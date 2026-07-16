from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy.engine import Engine

from server.repositories.database.backend import TKBENDatabase, get_database

###############################################################################
class DataRepositoryQueries:

    # -------------------------------------------------------------------------
    def __init__(self, database: TKBENDatabase | None = None) -> None:
        self.database = database or get_database()

    # -------------------------------------------------------------------------
    @property
    def engine(self) -> Engine:
        return self.database.backend.engine

    # -------------------------------------------------------------------------
    def insert_records(self, table_name: str, records: Sequence[Mapping[str, Any]], *, ignore_duplicates: bool = False) -> None:
        self.database.backend.insert_records(table_name, records, ignore_duplicates=ignore_duplicates)

    # -------------------------------------------------------------------------
    def upsert_records(self, table_name: str, records: Sequence[Mapping[str, Any]], conflict_columns: list[str]) -> None:
        self.database.backend.upsert_records(table_name, records, conflict_columns)

    # -------------------------------------------------------------------------
    def get_distinct_values(self, table_name: str, column: str) -> list[Any]:
        return self.database.backend.get_distinct_values(table_name, column)
