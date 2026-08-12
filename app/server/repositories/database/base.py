from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from sqlalchemy.orm import sessionmaker

from server.repositories.schemas.models import Base

###############################################################################
class RepositoryBase:
    IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    # -------------------------------------------------------------------------
    def __init__(self, engine: Any, insert_batch_size: int) -> None:
        self.engine = engine
        self.session_factory = sessionmaker(bind=engine, future=True)
        self.insert_batch_size = max(1, int(insert_batch_size))

    # -------------------------------------------------------------------------
    def sanitize_identifier(self, name: str) -> str:
        if not self.IDENTIFIER_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid SQL identifier: {name}")
        return name

    # -------------------------------------------------------------------------
    def get_table(self, table_name: str):
        self.sanitize_identifier(table_name)
        try:
            return Base.metadata.tables[table_name]
        except KeyError as exc:
            raise ValueError(f"Unknown canonical table: {table_name}") from exc

    # -------------------------------------------------------------------------
    def _batches(self, records: list[Mapping[str, Any]]) -> Iterable[list[Mapping[str, Any]]]:
        for start in range(0, len(records), self.insert_batch_size):
            yield records[start : start + self.insert_batch_size]

    # -------------------------------------------------------------------------
    def insert_records(self, table_name: str, records: Sequence[Mapping[str, Any]], *, ignore_duplicates: bool = False) -> None:
        if not records:
            return
        self._insert(self.get_table(table_name), records, ignore_duplicates=ignore_duplicates)

    # -------------------------------------------------------------------------
    def _insert(self, table, records, *, ignore_duplicates: bool) -> None:  # type: ignore[no-untyped-def]
        raise NotImplementedError

    # -------------------------------------------------------------------------
    def upsert_records(self, table_name: str, records: Sequence[Mapping[str, Any]], conflict_columns: list[str]) -> None:
        if not records:
            return
        self._upsert(self.get_table(table_name), records, conflict_columns)

    # -------------------------------------------------------------------------
    def _upsert(self, table, records, conflict_columns):  # type: ignore[no-untyped-def]
        raise NotImplementedError
