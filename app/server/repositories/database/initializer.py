from __future__ import annotations

from pathlib import Path
import urllib.parse

import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.elements import TextClause

from server.common.path import DATABASE_PATH
from server.common.utils.logger import logger
from server.configurations import DatabaseSettings, get_server_settings
from server.repositories.database.postgres import PostgresRepository
from server.repositories.database.seeding import seed_metric_types
from server.repositories.database.sqlite import SQLiteRepository
from server.repositories.database.utils import normalize_sqlite_path
from server.repositories.schemas.models import Base
from server.services.metrics.catalog import DATASET_METRIC_CATALOG

SUPPORTED_POSTGRES_ENGINE = "postgresql+psycopg"

###############################################################################
def build_postgres_connect_args(settings: DatabaseSettings) -> dict[str, str | int]:
    connect_args: dict[str, str | int] = {"connect_timeout": settings.connect_timeout}
    if settings.ssl:
        connect_args["sslmode"] = "require"
        if settings.ssl_ca:
            connect_args["sslrootcert"] = settings.ssl_ca
    return connect_args

###############################################################################
def build_postgres_url(settings: DatabaseSettings, database_name: str) -> str:
    port = settings.port or 5432
    engine_name = _resolve_postgres_engine(settings.engine)
    safe_username = urllib.parse.quote_plus(settings.username or "")
    safe_password = urllib.parse.quote_plus(settings.password or "")
    return (
        f"{engine_name}://{safe_username}:{safe_password}"
        f"@{settings.host}:{port}/{database_name}"
    )

###############################################################################
def clone_settings_with_database(
    settings: DatabaseSettings, database_name: str
) -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=False,
        engine=settings.engine,
        host=settings.host,
        port=settings.port,
        database_name=database_name,
        username=settings.username,
        password=settings.password,
        ssl=settings.ssl,
        ssl_ca=settings.ssl_ca,
        connect_timeout=settings.connect_timeout,
        insert_batch_size=settings.insert_batch_size,
    )

###############################################################################
def build_postgres_create_database_sql(
    database_name: str,
) -> TextClause:
    safe_database = database_name.replace('"', '""')
    return sqlalchemy.text(
        f"CREATE DATABASE \"{safe_database}\" WITH ENCODING 'UTF8' TEMPLATE template0"
    )

###############################################################################
def initialize_sqlite_database(settings: DatabaseSettings) -> None:
    database_path = Path(normalize_sqlite_path(DATABASE_PATH))
    if database_path.is_file():
        logger.info(
            "SQLite database already exists at %s; skipping initialization.",
            database_path,
        )
        return

    repository = SQLiteRepository(settings)
    Base.metadata.create_all(repository.engine)
    seed_metric_types(repository.engine, DATASET_METRIC_CATALOG)
    logger.info("Initialized SQLite database at %s", repository.db_path)

###############################################################################
def connect_postgres_database(settings: DatabaseSettings) -> None:
    repository = PostgresRepository(settings)
    with repository.engine.connect() as connection:
        connection.execute(sqlalchemy.text("SELECT 1"))
    logger.info("Connected to PostgreSQL database %s.", settings.database_name)

###############################################################################
def ensure_postgres_database(settings: DatabaseSettings) -> str:
    if not settings.host:
        raise ValueError("Database host is required for PostgreSQL initialization.")
    if not settings.username:
        raise ValueError("Database username is required for PostgreSQL initialization.")
    if not settings.database_name:
        raise ValueError("Database name is required for PostgreSQL initialization.")

    target_database = settings.database_name
    connect_args = build_postgres_connect_args(settings)

    admin_url = build_postgres_url(settings, "postgres")
    admin_engine = sqlalchemy.create_engine(
        admin_url,
        echo=False,
        future=True,
        connect_args=connect_args,
        isolation_level="AUTOCOMMIT",
        pool_pre_ping=True,
    )

    with admin_engine.connect() as conn:
        exists = conn.execute(
            sqlalchemy.text("SELECT 1 FROM pg_database WHERE datname=:name"),
            {"name": target_database},
        ).scalar()
        if exists:
            logger.info("PostgreSQL database %s already exists", target_database)
        else:
            conn.execute(build_postgres_create_database_sql(target_database))
            logger.info("Created PostgreSQL database %s", target_database)

    normalized_settings = clone_settings_with_database(settings, target_database)
    repository = PostgresRepository(normalized_settings)
    Base.metadata.create_all(repository.engine)
    seed_metric_types(repository.engine, DATASET_METRIC_CATALOG)
    logger.info("Ensured PostgreSQL tables exist in %s", target_database)

    return target_database

###############################################################################
def run_database_initialization(*, startup: bool = False) -> None:
    settings = get_server_settings().database
    if settings.embedded_database:
        initialize_sqlite_database(settings)
        return

    _resolve_postgres_engine(settings.engine)
    if startup:
        connect_postgres_database(settings)
    else:
        ensure_postgres_database(settings)

###############################################################################
def _resolve_postgres_engine(engine: str | None) -> str:
    normalized = (engine or "").strip().lower()
    if normalized == SUPPORTED_POSTGRES_ENGINE:
        return SUPPORTED_POSTGRES_ENGINE
    raise ValueError(f"Unsupported database engine: {engine}")

###############################################################################
def initialize_database(*, startup: bool = False) -> None:
    try:
        run_database_initialization(startup=startup)
    except (SQLAlchemyError, ValueError) as exc:
        logger.error("Database initialization failed: %s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:
        logger.exception("Unexpected error during database initialization.")
        raise SystemExit(1) from exc
