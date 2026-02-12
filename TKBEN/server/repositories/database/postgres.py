from __future__ import annotations

import urllib.parse
import re
from typing import Any

import pandas as pd
import sqlalchemy
from sqlalchemy import UniqueConstraint, inspect
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from TKBEN.server.configurations import DatabaseSettings
from TKBEN.server.repositories.database.utils import normalize_postgres_engine
from TKBEN.server.repositories.database.migration import run_schema_migration
from TKBEN.server.repositories.schemas.models import Base
from TKBEN.server.common.utils.logger import logger


###############################################################################
class PostgresRepository:
    IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(self, settings: DatabaseSettings) -> None:
        if not settings.host:
            raise ValueError("Database host must be provided for external database.")
        if not settings.database_name:
            raise ValueError(
                "Database name must be provided for external database."
            )
        if not settings.username:
            raise ValueError(
                "Database username must be provided for external database."
            )

        port = settings.port or 5432
        engine_name = normalize_postgres_engine(settings.engine)
        password = settings.password or ""
        connect_args: dict[str, Any] = {"connect_timeout": settings.connect_timeout}
        if settings.ssl:
            connect_args["sslmode"] = "require"
            if settings.ssl_ca:
                connect_args["sslrootcert"] = settings.ssl_ca

        safe_username = urllib.parse.quote_plus(settings.username)
        safe_password = urllib.parse.quote_plus(password)
        self.db_path: str | None = None
        self.engine: Engine = sqlalchemy.create_engine(
            f"{engine_name}://{safe_username}:{safe_password}@{settings.host}:{port}/{settings.database_name}",
            echo=False,
            future=True,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
        self.session = sessionmaker(bind=self.engine, future=True)
        self.insert_batch_size = settings.insert_batch_size
        Base.metadata.create_all(self.engine, checkfirst=True)
        run_schema_migration(self.engine)

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
            existing_cols = {column["name"] for column in inspector.get_columns(table_name)}
            if existing_cols == expected_cols:
                return
            logger.warning(
                "Schema mismatch for %s. Recreating table. existing=%s expected=%s",
                table_name,
                sorted(existing_cols),
                sorted(expected_cols),
            )
            table_cls.__table__.drop(conn, checkfirst=True)
            table_cls.__table__.create(conn, checkfirst=True)

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
            for i in range(0, len(records), self.insert_batch_size):
                batch = records[i : i + self.insert_batch_size]
                if not batch:
                    continue
                stmt = insert(table).values(batch)
                update_cols = {
                    col: getattr(stmt.excluded, col)  # type: ignore[attr-defined]
                    for col in batch[0]
                    if col not in unique_cols
                }
                stmt = stmt.on_conflict_do_update(
                    index_elements=unique_cols, set_=update_cols
                )
                session.execute(stmt)
                session.commit()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def load_from_database(self, table_name: str) -> pd.DataFrame:
        safe_name = self.sanitize_identifier(table_name)
        with self.engine.connect() as conn:
            if not self.relation_exists(conn, safe_name):
                logger.warning("Table %s does not exist", table_name)
                return pd.DataFrame()
            query = sqlalchemy.text(f'SELECT * FROM "{safe_name}"')
            data = pd.read_sql_query(query, conn)
        return data

    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        with self.engine.begin() as conn:
            inspector = inspect(conn)
            if inspector.has_table(table_name):
                conn.execute(sqlalchemy.text(f'DELETE FROM "{table_name}"'))
            df.to_sql(table_name, conn, if_exists="append", index=False)

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        table_cls = self.get_table_class(table_name)
        self.upsert_dataframe(df, table_cls)

    def bulk_replace_by_key(
        self, df: pd.DataFrame, table_name: str, key_column: str, key_value: str
    ) -> None:
        """
        Fast bulk insert: delete all rows matching key_value, then bulk insert new data.
        Uses batched inserts with separate commits to avoid memory issues on large datasets.
        """
        safe_name = self.sanitize_identifier(table_name)
        safe_key = self.sanitize_identifier(key_column)
        # Delete existing rows first (separate transaction)
        with self.engine.begin() as conn:
            conn.execute(
                sqlalchemy.text(f'DELETE FROM "{safe_name}" WHERE "{safe_key}" = :key'),
                {"key": key_value},
            )

        # Insert in batches with separate commits for each batch
        total_rows = len(df)
        if total_rows == 0:
            return

        batch_size = self.insert_batch_size
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            batch_df = df.iloc[start:end]
            with self.engine.begin() as conn:
                batch_df.to_sql(safe_name, conn, if_exists="append", index=False)
            logger.info(
                "Inserted batch %d-%d of %d rows into %s",
                start + 1, end, total_rows, table_name
            )

    # -------------------------------------------------------------------------
    def delete_by_key(
        self, table_name: str, key_column: str, key_value: str
    ) -> None:
        """Delete all rows matching the specified key value."""
        safe_name = self.sanitize_identifier(table_name)
        safe_key = self.sanitize_identifier(key_column)
        with self.engine.begin() as conn:
            result = conn.execute(
                sqlalchemy.text(f'DELETE FROM "{safe_name}" WHERE "{safe_key}" = :key'),
                {"key": key_value},
            )
        deleted_rows = int(result.rowcount or 0)
        if deleted_rows > 0:
            logger.info(
                "Deleted %d rows for %s=%s from %s",
                deleted_rows,
                key_column,
                key_value,
                table_name,
            )

    # -------------------------------------------------------------------------
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

        batch_size = self.insert_batch_size
        records = df.to_dict(orient="records")

        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            batch = records[start:end]

            with self.engine.begin() as conn:
                if ignore_duplicates:
                    # Use ON CONFLICT DO NOTHING to skip duplicates
                    stmt = insert(table).values(batch).on_conflict_do_nothing()
                    conn.execute(stmt)
                else:
                    # Standard insert (will fail on duplicates)
                    batch_df = df.iloc[start:end]
                    batch_df.to_sql(table_name, conn, if_exists="append", index=False)

    # -----------------------------------------------------------------------------
    def get_distinct_values(self, table_name: str, column: str) -> list[str]:
        """Get distinct values from a column in the specified table."""
        safe_name = self.sanitize_identifier(table_name)
        safe_column = self.sanitize_identifier(column)
        with self.engine.connect() as conn:
            if not self.relation_exists(conn, safe_name):
                return []
            result = conn.execute(
                sqlalchemy.text(f'SELECT DISTINCT "{safe_column}" FROM "{safe_name}"')
            )
            return [row[0] for row in result.fetchall() if row[0] is not None]




