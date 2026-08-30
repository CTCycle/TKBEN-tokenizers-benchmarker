from __future__ import annotations

import time
from typing import Any

from alembic import command, script
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Connection, Engine, inspect, text

from server.common.path import SERVER_DIR
from server.common.utils.logger import logger
from server.configurations import DatabaseSettings

ALEMBIC_CONFIG_PATH = SERVER_DIR / "pyproject.toml"
ALEMBIC_VERSION_TABLE = "alembic_version"
POSTGRES_LOCK_NAME = "tkben:alembic:migrations"


###############################################################################
class DatabaseMigrationError(RuntimeError):
    """Raised when the application cannot make its database current."""


###############################################################################
def build_alembic_config() -> Config:
    config = Config(toml_file=str(ALEMBIC_CONFIG_PATH))
    config.set_main_option("script_location", str(SERVER_DIR / "migrations"))
    config.set_main_option("version_table", ALEMBIC_VERSION_TABLE)
    config.set_main_option("version_table_pk", "true")
    return config


###############################################################################
def _migration_directory(config: Config) -> script.ScriptDirectory:
    directory = script.ScriptDirectory.from_config(config)
    heads = tuple(directory.get_heads())
    if len(heads) != 1:
        raise DatabaseMigrationError(
            f"Alembic migration graph must have exactly one head; found {heads!r}."
        )
    return directory


###############################################################################
def _current_heads(connection: Connection) -> tuple[str, ...]:
    migration_context = MigrationContext.configure(
        connection,
        opts={"version_table": ALEMBIC_VERSION_TABLE},
    )
    return tuple(migration_context.get_current_heads())


###############################################################################
def _domain_tables(connection: Connection) -> set[str]:
    return set(inspect(connection).get_table_names()) - {ALEMBIC_VERSION_TABLE}


###############################################################################
def _synchronize_schema(
    config: Config,
    directory: script.ScriptDirectory,
    connection: Connection,
) -> None:
    expected_heads = tuple(directory.get_heads())
    current = _current_heads(connection)
    logger.info(
        "Detected Alembic revision(s) %s; target head is %s.",
        current or ("base",),
        expected_heads,
    )
    if len(current) > 1:
        raise DatabaseMigrationError(
            f"Database has multiple Alembic revisions recorded: {current!r}."
        )

    known_revisions = {revision.revision for revision in directory.walk_revisions()}
    unknown = set(current) - known_revisions
    if unknown:
        raise DatabaseMigrationError(
            f"Database references unknown Alembic revisions: {sorted(unknown)!r}."
        )

    if not current and _domain_tables(connection):
        raise DatabaseMigrationError(
            "Database contains a non-empty unversioned schema; refusing to "
            "adopt or alter it. Restore a versioned backup or recreate the database."
        )

    if current != expected_heads:
        logger.info("Applying Alembic migrations to head.")
        command.upgrade(config, "head")

    final_heads = _current_heads(connection)
    if final_heads != expected_heads:
        raise DatabaseMigrationError(
            f"Database migration finished at {final_heads!r}; expected "
            f"{expected_heads!r}."
        )
    logger.info("Alembic migration verification succeeded at %s.", expected_heads)


###############################################################################
def _acquire_postgres_lock(
    connection: Connection,
    database_label: str,
    timeout_seconds: int,
) -> None:
    lock_name = f"{POSTGRES_LOCK_NAME}:{database_label}"
    deadline = time.monotonic() + max(1, timeout_seconds)
    logger.info("Waiting for PostgreSQL migration lock for %s.", database_label)
    while True:
        acquired = connection.execute(
            text("SELECT pg_try_advisory_xact_lock(hashtext(:lock_name))"),
            {"lock_name": lock_name},
        ).scalar()
        if acquired:
            logger.info("Acquired PostgreSQL migration lock for %s.", database_label)
            return
        if time.monotonic() >= deadline:
            raise DatabaseMigrationError(
                f"Timed out waiting for the PostgreSQL migration lock for "
                f"{database_label}."
            )
        time.sleep(0.1)


###############################################################################
def _foreign_key_violations(connection: Connection) -> list[tuple[Any, ...]]:
    return [
        tuple(row) for row in connection.execute(text("PRAGMA foreign_key_check")).all()
    ]


###############################################################################
def run_locked_migrations(
    engine: Engine,
    settings: DatabaseSettings,
    database_label: str,
    *,
    postgres: bool,
) -> None:
    config = build_alembic_config()
    directory = _migration_directory(config)
    backend_name = "PostgreSQL" if postgres else "SQLite"
    logger.info(
        "Starting %s migration check for %s.",
        backend_name,
        database_label,
    )
    try:
        with engine.connect() as connection:
            with connection.begin():
                if postgres:
                    _acquire_postgres_lock(
                        connection,
                        database_label,
                        settings.connect_timeout,
                    )
                config.attributes["connection"] = connection
                _synchronize_schema(config, directory, connection)
                if not postgres:
                    violations = _foreign_key_violations(connection)
                    if violations:
                        raise DatabaseMigrationError(
                            "SQLite foreign-key validation failed after migration: "
                            f"{violations[:5]!r}"
                        )
    except DatabaseMigrationError:
        raise
    except Exception as exc:
        raise DatabaseMigrationError(
            f"Unable to migrate database {database_label}."
        ) from exc
    finally:
        config.attributes.pop("connection", None)
