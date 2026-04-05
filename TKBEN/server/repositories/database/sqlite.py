from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd
import sqlalchemy
from sqlalchemy import MetaData, Table, UniqueConstraint, event, inspect, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from TKBEN.server.configurations import DatabaseSettings
from TKBEN.server.repositories.schemas.models import Base
from TKBEN.server.common.constants import RESOURCES_PATH, DATABASE_FILENAME
from TKBEN.server.repositories.database.utils import normalize_sqlite_path
from TKBEN.server.common.utils.logger import logger


###############################################################################
class SQLiteRepository:
    IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    SQLITE_MAX_VARIABLES = 999

    def __init__(
        self, settings: DatabaseSettings, initialize_schema: bool = False
    ) -> None:
        self.db_path: str | None = normalize_sqlite_path(
            os.path.join(RESOURCES_PATH, DATABASE_FILENAME)
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.engine: Engine = sqlalchemy.create_engine(
            f"sqlite:///{self.db_path}", echo=False, future=True
        )
        event.listen(self.engine, "connect", self.enable_foreign_keys)
        self.session = sessionmaker(bind=self.engine, future=True)
        self.insert_batch_size = settings.insert_batch_size
        if initialize_schema:
            Base.metadata.create_all(self.engine, checkfirst=True)

    # -------------------------------------------------------------------------
    def enable_foreign_keys(self, dbapi_connection, connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def sanitize_identifier(self, name: str) -> str:
        if not self.IDENTIFIER_PATTERN.match(name):
            raise ValueError(f"Invalid SQL identifier: {name}")
        return name

    # -------------------------------------------------------------------------
    def relation_exists(self, conn: Any, relation_name: str) -> bool:
        inspector = inspect(conn)
        if inspector.has_table(relation_name):
            return True
        return relation_name in inspector.get_view_names()

    # -------------------------------------------------------------------------
    def get_table_class(self, table_name: str) -> Any:
        for cls in Base.__subclasses__():
            if getattr(cls, "__tablename__", None) == table_name:
                return cls
        raise ValueError(f"No table class found for name {table_name}")

    # -------------------------------------------------------------------------
    def ensure_table_schema(self, table_name: str) -> None:
        table_cls = self.get_table_class(table_name)
        expected_cols = set(table_cls.__table__.columns.keys())
        with self.engine.begin() as conn:
            inspector = inspect(conn)
            if not inspector.has_table(table_name):
                return
            existing_cols = {
                column["name"] for column in inspector.get_columns(table_name)
            }
            if existing_cols == expected_cols:
                return
            raise RuntimeError(
                "Schema mismatch for table '"
                f"{table_name}"
                "'. Runtime auto-recreate is disabled; run schema initialization/migration before startup. "
                f"Existing columns: {sorted(existing_cols)}; expected columns: {sorted(expected_cols)}"
            )

    # -------------------------------------------------------------------------
    def _effective_batch_size(self, row_sample: dict[str, Any] | None) -> int:
        if not row_sample:
            return self.insert_batch_size
        columns_per_row = max(1, len(row_sample))
        sqlite_cap = max(1, self.SQLITE_MAX_VARIABLES // columns_per_row)
        return max(1, min(self.insert_batch_size, sqlite_cap))

    # -------------------------------------------------------------------------
    def upsert_dataframe(self, df: pd.DataFrame, table_cls) -> None:
        table = table_cls.__table__
        session = self.session()
        try:
            unique_cols: list[str] = []
            for uc in table.constraints:
                if isinstance(uc, UniqueConstraint):
                    unique_cols = list(uc.columns.keys())
                    break
            if not unique_cols:
                unique_cols = [column.name for column in table.primary_key.columns]
            if not unique_cols:
                raise ValueError(f"No unique constraint found for {table_cls.__name__}")
            records = df.to_dict(orient="records")
            batch_size = self._effective_batch_size(records[0] if records else None)
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                if not batch:
                    continue
                stmt = insert(table).values(batch)
                update_cols = {
                    col: getattr(stmt.excluded, col)  # type: ignore[attr-defined]
                    for col in batch[0]
                    if col not in unique_cols
                }
                # If all columns are part of unique constraint, use DO NOTHING
                if update_cols:
                    stmt = stmt.on_conflict_do_update(
                        index_elements=unique_cols, set_=update_cols
                    )
                else:
                    stmt = stmt.on_conflict_do_nothing(index_elements=unique_cols)
                session.execute(stmt)
            # Commit once at the end for better performance
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def load_from_database(self, table_name: str) -> pd.DataFrame:
        safe_name = self.sanitize_identifier(table_name)
        with self.session() as session:
            conn = session.connection()
            if not self.relation_exists(conn, safe_name):
                logger.warning("Table %s does not exist", table_name)
                return pd.DataFrame()
            try:
                table_obj = self.get_table_class(safe_name).__table__
            except ValueError:
                table_obj = Table(safe_name, MetaData(), autoload_with=self.engine)
            rows = session.execute(select(table_obj)).mappings().all()
        return pd.DataFrame([dict(row) for row in rows])

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        table_cls = self.get_table_class(table_name)
        self.upsert_dataframe(df, table_cls)

    # -----------------------------------------------------------------------------
    def insert_dataframe(
        self, df: pd.DataFrame, table_name: str, ignore_duplicates: bool = True
    ) -> None:
        """
        Insert DataFrame rows in batches with separate commits per batch.
        This is an append-only operation (no delete).

        Args:
            df: DataFrame to insert
            table_name: Target table name
            ignore_duplicates: If True, skip rows that violate unique constraints
        """
        total_rows = len(df)
        if total_rows == 0:
            return

        self.ensure_table_schema(table_name)
        table_cls = self.get_table_class(table_name)
        table = table_cls.__table__

        records = df.to_dict(orient="records")
        batch_size = self._effective_batch_size(records[0] if records else None)
        session = self.session()
        try:
            for start in range(0, total_rows, batch_size):
                end = min(start + batch_size, total_rows)
                batch = records[start:end]
                if not batch:
                    continue
                if ignore_duplicates:
                    stmt = insert(table).values(batch).on_conflict_do_nothing()
                else:
                    stmt = insert(table).values(batch)
                session.execute(stmt)
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -----------------------------------------------------------------------------
    def get_distinct_values(self, table_name: str, column: str) -> list[str]:
        """Get distinct values from a column in the specified table."""
        safe_name = self.sanitize_identifier(table_name)
        safe_column = self.sanitize_identifier(column)
        with self.session() as session:
            conn = session.connection()
            if not self.relation_exists(conn, safe_name):
                return []
            try:
                table_obj = self.get_table_class(safe_name).__table__
            except ValueError:
                table_obj = Table(safe_name, MetaData(), autoload_with=self.engine)
            if safe_column not in table_obj.c:
                return []
            result = session.execute(select(table_obj.c[safe_column]).distinct()).scalars()
            return [value for value in result if value is not None]
