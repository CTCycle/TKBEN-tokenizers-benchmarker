from __future__ import annotations

from pathlib import Path
import urllib.parse

import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.elements import TextClause

from server.common.path import DATABASE_PATH
from server.common.utils.logger import logger
from server.configurations import DatabaseSettings, get_server_settings
from server.repositories.database.migrations import (
    DatabaseMigrationError,
    run_locked_migrations,
)
from server.repositories.database.postgres import PostgresRepository
from server.repositories.database.sqlite import SQLiteRepository
from server.repositories.database.utils import normalize_sqlite_path

SUPPORTED_POSTGRES_ENGINE = "postgresql+psycopg"
POSTGRES_CREATION_LOCK_NAME = "tkben:database:create"

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
def build_postgres_create_database_sql(database_name: str) -> TextClause:
    safe_database = database_name.replace('"', '""')
    return sqlalchemy.text(
        f'CREATE DATABASE "{safe_database}" WITH ENCODING \'UTF8\' TEMPLATE template0'
    )

###############################################################################
def initialize_sqlite_database(settings: DatabaseSettings) -> None:
    database_path = Path(normalize_sqlite_path(DATABASE_PATH))
    logger.info("Checking SQLite database %s.", database_path)
    repository = SQLiteRepository(
        settings,
        enforce_foreign_keys=False,
        begin_immediate=True,
    )
    try:
        run_locked_migrations(
            repository.engine,
            settings,
            str(database_path),
            postgres=False,
        )
    finally:
        repository.engine.dispose()
    logger.info("SQLite database %s is synchronized.", database_path)

###############################################################################
def connect_postgres_database(settings: DatabaseSettings) -> None:
    """Verify the configured PostgreSQL target is reachable."""
    repository = PostgresRepository(settings)
    try:
        with repository.engine.connect() as connection:
            connection.execute(sqlalchemy.text("SELECT 1"))
    finally:
        repository.engine.dispose()
    logger.info("Connected to PostgreSQL database %s.", settings.database_name)

###############################################################################
def _is_missing_postgres_database(error: BaseException) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        sqlstate = getattr(current, "sqlstate", None) or getattr(
            current, "pgcode", None
        )
        if sqlstate == "3D000":
            return True
        message = str(current).lower()
        if "database" in message and "does not exist" in message:
            return True
        original = getattr(current, "orig", None)
        current = original or current.__cause__ or current.__context__
    return False

###############################################################################
def _create_missing_postgres_database(settings: DatabaseSettings) -> None:
    admin_url = build_postgres_url(settings, "postgres")
    admin_engine = sqlalchemy.create_engine(
        admin_url,
        echo=False,
        future=True,
        connect_args=build_postgres_connect_args(settings),
        isolation_level="AUTOCOMMIT",
        pool_pre_ping=True,
    )
    lock_name = f"{POSTGRES_CREATION_LOCK_NAME}:{settings.database_name}"
    try:
        with admin_engine.connect() as connection:
            logger.info(
                "Waiting for PostgreSQL database creation lock for %s.",
                settings.database_name,
            )
            connection.execute(
                sqlalchemy.text("SELECT pg_advisory_lock(hashtext(:lock_name))"),
                {"lock_name": lock_name},
            )
            try:
                exists = connection.execute(
                    sqlalchemy.text("SELECT 1 FROM pg_database WHERE datname=:name"),
                    {"name": settings.database_name},
                ).scalar()
                if exists:
                    logger.info(
                        "PostgreSQL database %s was created by another instance.",
                        settings.database_name,
                    )
                else:
                    connection.execute(
                        build_postgres_create_database_sql(
                            settings.database_name or ""
                        )
                    )
                    logger.info("Created PostgreSQL database %s.", settings.database_name)
            finally:
                connection.execute(
                    sqlalchemy.text("SELECT pg_advisory_unlock(hashtext(:lock_name))"),
                    {"lock_name": lock_name},
                )
    except SQLAlchemyError as exc:
        raise DatabaseMigrationError(
            "Unable to create the PostgreSQL database; verify the configured "
            "role has CREATEDB permission."
        ) from exc
    finally:
        admin_engine.dispose()

###############################################################################
def ensure_postgres_database(settings: DatabaseSettings) -> str:
    """Ensure the configured PostgreSQL target exists, creating it if absent."""
    if not settings.host:
        raise ValueError("Database host is required for PostgreSQL initialization.")
    if not settings.username:
        raise ValueError("Database username is required for PostgreSQL initialization.")
    if not settings.database_name:
        raise ValueError("Database name is required for PostgreSQL initialization.")

    target_repository = PostgresRepository(settings)
    try:
        try:
            with target_repository.engine.connect() as connection:
                connection.execute(sqlalchemy.text("SELECT 1"))
            logger.info("PostgreSQL database %s already exists.", settings.database_name)
            return settings.database_name
        except SQLAlchemyError as error:
            if not _is_missing_postgres_database(error):
                raise
            logger.info(
                "PostgreSQL database %s does not exist; requesting creation.",
                settings.database_name,
            )
    finally:
        target_repository.engine.dispose()

    _create_missing_postgres_database(settings)
    connect_postgres_database(settings)
    return settings.database_name

###############################################################################
def run_database_initialization(*, startup: bool = False) -> None:
    del startup  # Startup and explicit initialization share the same workflow.
    settings = get_server_settings().database
    if settings.embedded_database:
        initialize_sqlite_database(settings)
        return

    _resolve_postgres_engine(settings.engine)
    ensure_postgres_database(settings)
    repository = PostgresRepository(settings)
    try:
        run_locked_migrations(
            repository.engine,
            settings,
            settings.database_name or "postgresql",
            postgres=True,
        )
    finally:
        repository.engine.dispose()

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
    except DatabaseMigrationError as exc:
        logger.error("Database migration failed: %s", exc)
        raise
    except (SQLAlchemyError, ValueError) as exc:
        logger.error("Database initialization failed: %s", exc)
        raise DatabaseMigrationError("Database initialization failed.") from exc
    except RuntimeError as exc:
        logger.error("Database initialization failed: %s", exc)
        raise DatabaseMigrationError(str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during database initialization.")
        raise DatabaseMigrationError("Database initialization failed unexpectedly.") from exc
