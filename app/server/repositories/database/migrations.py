from __future__ import annotations

from copy import deepcopy
import re
import time
from typing import Any, Literal

from alembic import command, script
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import CheckConstraint, Connection, Engine, UniqueConstraint, inspect, text

from server.common.path import SERVER_DIR
from server.common.utils.logger import logger
from server.configurations import DatabaseSettings
from server.repositories.database.seeding import seed_metric_types
from server.repositories.schemas.models import Base
from server.common.metric_catalog import DATASET_METRIC_CATALOG

ALEMBIC_CONFIG_PATH = SERVER_DIR / "pyproject.toml"
ALEMBIC_VERSION_TABLE = "alembic_version"
LEGACY_REVISION = "0001_pre_alembic_schema"
HEAD_REVISION = "0002_current_schema"
POSTGRES_LOCK_NAME = "tkben:alembic:migrations"

SchemaState = Literal["empty", "canonical", "legacy", "unknown"]

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
    if heads != (HEAD_REVISION,):
        raise DatabaseMigrationError(
            "Alembic migration graph must have exactly one expected head; "
            f"found {heads!r}."
        )
    return directory

###############################################################################
def _normalize_sql(value: str | None) -> str:
    normalized = (value or "").lower()
    # PostgreSQL reflects literal defaults with an explicit type cast (for
    # example ``'2'::integer``); the signature compares the literal value,
    # not that dialect spelling.
    normalized = re.sub(r"::[a-z0-9_\s\[\].]+", "", normalized)
    return re.sub(r"[^a-z0-9_]", "", normalized)

###############################################################################
def _normalize_default(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(
        r"::[a-z0-9_\s\[\].]+",
        "",
        value.lower().strip(),
    )
    while normalized.startswith("(") and normalized.endswith(")"):
        normalized = normalized[1:-1].strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] == "'":
        normalized = normalized[1:-1]
    return re.sub(r"\s+", "", normalized)

###############################################################################
def _normalize_type(value: Any, connection: Connection) -> str:
    if hasattr(value, "compile"):
        rendered = str(value.compile(dialect=connection.dialect))
    else:
        rendered = str(value)
    return re.sub(r"\s+", "", rendered).upper()

###############################################################################
def _column_default(column: Any) -> str | None:
    if column.server_default is None:
        return None
    return _normalize_default(str(column.server_default.arg))

###############################################################################
def _normalize_index_predicate(value: Any, table_name: str) -> str:
    normalized = _normalize_sql(str(value)).replace(_normalize_sql(table_name), "")
    return normalized.replace("istrue", "is1")

###############################################################################
def _index_predicate(index: Any, dialect_name: str) -> str | None:
    where = (index.dialect_options.get(dialect_name) or {}).get("where")
    return (
        _normalize_index_predicate(where, index.table.name)
        if where is not None
        else None
    )

###############################################################################
def _metadata_signature(connection: Connection) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for table_name, table in Base.metadata.tables.items():
        unique_constraints = sorted(
            tuple(column.name for column in constraint.columns)
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        )
        foreign_keys = sorted(
            (
                tuple(element.parent.name for element in constraint.elements),
                (
                    constraint.elements[0].column.table.name,
                    tuple(element.column.name for element in constraint.elements),
                ),
                (constraint.elements[0].ondelete,),
            )
            for constraint in table.foreign_key_constraints
        )
        indexes = sorted(
            (
                index.name,
                tuple(column.name for column in index.columns),
                bool(index.unique),
                _index_predicate(index, connection.dialect.name),
            )
            for index in table.indexes
        )
        checks = sorted(
            (constraint.name, _normalize_sql(str(constraint.sqltext)))
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint)
        )
        tables[table_name] = {
            "columns": tuple(
                (
                    column.name,
                    _normalize_type(column.type, connection),
                    bool(column.nullable),
                    _column_default(column),
                )
                for column in table.columns
            ),
            "primary_key": tuple(column.name for column in table.primary_key.columns),
            "unique_constraints": tuple(unique_constraints),
            "foreign_keys": tuple(foreign_keys),
            "indexes": tuple(indexes),
            "checks": tuple(checks),
        }
    return {
        "tables": tables,
        "report_default": _column_default(
            Base.metadata.tables["analysis_session"].columns["report_version"]
        ),
    }

###############################################################################
def _reflected_signature(connection: Connection) -> dict[str, Any]:
    inspector = inspect(connection)
    table_names = sorted(
        name for name in inspector.get_table_names() if name != ALEMBIC_VERSION_TABLE
    )
    tables: dict[str, Any] = {}
    for table_name in table_names:
        unique_constraints = sorted(
            tuple(item.get("column_names") or ())
            for item in inspector.get_unique_constraints(table_name)
        )
        foreign_keys = sorted(
            (
                tuple(item.get("constrained_columns") or ()),
                (
                    item.get("referred_table", ""),
                    tuple(item.get("referred_columns") or ()),
                ),
                ((item.get("options") or {}).get("ondelete"),),
            )
            for item in inspector.get_foreign_keys(table_name)
        )
        indexes = sorted(
            (
                item.get("name"),
                tuple(item.get("column_names") or ()),
                bool(item.get("unique")),
                (
                    _normalize_index_predicate(
                        (item.get("dialect_options") or {}).get(
                            f"{connection.dialect.name}_where"
                        ),
                        table_name,
                    )
                    if (item.get("dialect_options") or {}).get(
                        f"{connection.dialect.name}_where"
                    )
                    is not None
                    else None
                ),
            )
            for item in inspector.get_indexes(table_name)
        )
        checks = sorted(
            (item.get("name"), _normalize_sql(item.get("sqltext")))
            for item in inspector.get_check_constraints(table_name)
        )
        tables[table_name] = {
            "columns": tuple(
                (
                    item["name"],
                    _normalize_type(item["type"], connection),
                    bool(item.get("nullable", True)),
                    (
                        _normalize_default(item["default"])
                        if item.get("default") is not None
                        else None
                    ),
                )
                for item in inspector.get_columns(table_name)
            ),
            "primary_key": tuple(
                inspector.get_pk_constraint(table_name).get("constrained_columns")
                or ()
            ),
            "unique_constraints": tuple(unique_constraints),
            "foreign_keys": tuple(foreign_keys),
            "indexes": tuple(indexes),
            "checks": tuple(checks),
        }
    report_columns = {
        item["name"]: item
        for item in inspector.get_columns("analysis_session")
    } if "analysis_session" in tables else {}
    return {
        "tables": tables,
        "report_default": _normalize_default(
            report_columns.get("report_version", {}).get("default")
        ),
    }

###############################################################################
def _legacy_metric_check(expression: str) -> bool:
    return _normalize_sql(expression) == _normalize_sql(
        "(numeric_value IS NOT NULL) + (text_value IS NOT NULL) + "
        "(json_value IS NOT NULL) = 1"
    )

###############################################################################
def _canonical_metric_check(expression: str, expected: str) -> bool:
    return _normalize_sql(expression) == _normalize_sql(expected)

###############################################################################
def _same_structure_except_known_legacy_differences(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    if actual.get("report_default") != "1":
        return False
    if set(actual["tables"]) != set(expected["tables"]):
        return False
    actual_tables = deepcopy(actual["tables"])
    expected_tables = deepcopy(expected["tables"])
    actual_metric_checks = dict(actual_tables["metric_value"]["checks"])
    expected_metric_checks = dict(expected_tables["metric_value"]["checks"])
    actual_metric_expression = actual_metric_checks.get("ck_metric_exactly_one_value")
    expected_metric_expression = expected_metric_checks.get("ck_metric_exactly_one_value")
    if not actual_metric_expression or not expected_metric_expression:
        return False
    if not (
        _legacy_metric_check(actual_metric_expression)
        or _canonical_metric_check(
            actual_metric_expression,
            expected_metric_expression,
        )
    ):
        return False
    actual_metric_checks.pop("ck_metric_exactly_one_value", None)
    expected_metric_checks.pop("ck_metric_exactly_one_value", None)
    actual_tables["metric_value"]["checks"] = tuple(sorted(actual_metric_checks.items()))
    expected_tables["metric_value"]["checks"] = tuple(sorted(expected_metric_checks.items()))
    actual_analysis_columns = list(actual_tables["analysis_session"]["columns"])
    expected_analysis_columns = expected_tables["analysis_session"]["columns"]
    report_column_index = next(
        index
        for index, column in enumerate(actual_analysis_columns)
        if column[0] == "report_version"
    )
    expected_report_column = next(
        column
        for column in expected_analysis_columns
        if column[0] == "report_version"
    )
    actual_report_column = actual_analysis_columns[report_column_index]
    actual_analysis_columns[report_column_index] = (
        *actual_report_column[:3],
        expected_report_column[3],
    )
    actual_tables["analysis_session"]["columns"] = tuple(actual_analysis_columns)
    return actual_tables == expected_tables

###############################################################################
def classify_unversioned_schema(connection: Connection) -> SchemaState:
    reflected = _reflected_signature(connection)
    if not reflected["tables"]:
        return "empty"
    expected = _metadata_signature(connection)
    if reflected == expected:
        return "canonical"
    if _same_structure_except_known_legacy_differences(reflected, expected):
        return "legacy"
    return "unknown"

###############################################################################
def _current_heads(connection: Connection) -> tuple[str, ...]:
    migration_context = MigrationContext.configure(
        connection,
        opts={"version_table": ALEMBIC_VERSION_TABLE},
    )
    return tuple(migration_context.get_current_heads())

###############################################################################
def _pending_revisions(
    directory: script.ScriptDirectory,
    current: tuple[str, ...],
) -> list[script.Script]:
    current_revision = current[0] if current else None
    pending: list[script.Script] = []
    for revision in directory.walk_revisions():
        if revision.revision == current_revision:
            break
        pending.append(revision)
    return list(reversed(pending))

###############################################################################
def _stamp_or_upgrade_unversioned(
    config: Config,
    directory: script.ScriptDirectory,
    connection: Connection,
    current: tuple[str, ...],
) -> tuple[str, ...]:
    if current:
        return current
    state = classify_unversioned_schema(connection)
    if state == "unknown":
        raise DatabaseMigrationError(
            "Database contains an unversioned schema that does not match a "
            "supported TKBEN schema; refusing to stamp or alter it."
        )
    if state == "canonical":
        logger.info(
            "Adopting canonical unversioned database schema at revision %s.",
            HEAD_REVISION,
        )
        command.stamp(config, HEAD_REVISION)
        return (HEAD_REVISION,)
    if state == "legacy":
        logger.info(
            "Adopting supported legacy unversioned database schema at revision %s.",
            LEGACY_REVISION,
        )
        command.stamp(config, LEGACY_REVISION)
        return (LEGACY_REVISION,)
    logger.info("Database is empty; applying the complete Alembic history.")
    return ()

###############################################################################
def _synchronize_schema(
    config: Config,
    directory: script.ScriptDirectory,
    connection: Connection,
) -> None:
    current = _current_heads(connection)
    logger.info(
        "Detected Alembic revision(s) %s; target head is %s.",
        current or ("base",),
        HEAD_REVISION,
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
    current = _stamp_or_upgrade_unversioned(config, directory, connection, current)
    if current == (HEAD_REVISION,):
        logger.info("Database schema is already at Alembic head %s.", HEAD_REVISION)
    else:
        pending = _pending_revisions(directory, current)
        if pending:
            logger.info(
                "Applying Alembic revisions: %s",
                ", ".join(
                    f"{revision.revision} ({revision.doc or 'no description'})"
                    for revision in pending
                ),
            )
        command.upgrade(config, HEAD_REVISION)
    final_heads = _current_heads(connection)
    if final_heads != (HEAD_REVISION,):
        raise DatabaseMigrationError(
            f"Database migration finished at {final_heads!r}; expected "
            f"{HEAD_REVISION!r}."
        )
    logger.info("Alembic migration verification succeeded at %s.", HEAD_REVISION)

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
        tuple(row)
        for row in connection.execute(text("PRAGMA foreign_key_check")).all()
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
                seed_metric_types(
                    connection,
                    DATASET_METRIC_CATALOG,
                    commit=False,
                )
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
