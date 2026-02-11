from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from TKBEN.server.repositories.database.backend import TKBENWebappDatabase, database


###############################################################################
class DataRepositoryQueries:
    def __init__(self, db: TKBENWebappDatabase = database) -> None:
        self.database = db

    # -------------------------------------------------------------------------
    @property
    def engine(self) -> Engine:
        return self.database.backend.engine

    # -------------------------------------------------------------------------
    def load_table(self, table_name: str) -> pd.DataFrame:
        return self.database.load_from_database(table_name)

    # -------------------------------------------------------------------------
    def save_table(self, dataset: pd.DataFrame, table_name: str) -> None:
        self.database.save_into_database(dataset, table_name)

    # -------------------------------------------------------------------------
    def upsert_table(self, dataset: pd.DataFrame, table_name: str) -> None:
        self.database.upsert_into_database(dataset, table_name)

    # -------------------------------------------------------------------------
    def insert_table(
        self, dataset: pd.DataFrame, table_name: str, ignore_duplicates: bool = True
    ) -> None:
        self.database.insert_dataframe(
            dataset,
            table_name,
            ignore_duplicates=ignore_duplicates,
        )

    def delete_by_key(self, table_name: str, key_column: str, key_value: str) -> None:
        self.database.delete_by_key(table_name, key_column, key_value)

    # -------------------------------------------------------------------------
    def bulk_replace_by_key(
        self, dataset: pd.DataFrame, table_name: str, key_column: str, key_value: str
    ) -> None:
        self.database.bulk_replace_by_key(dataset, table_name, key_column, key_value)

    # -------------------------------------------------------------------------
    def get_distinct_values(self, table_name: str, column: str) -> list[Any]:
        return self.database.backend.get_distinct_values(table_name, column)
