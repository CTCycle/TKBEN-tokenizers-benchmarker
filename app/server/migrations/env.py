from __future__ import annotations

from typing import Any, Literal

from alembic import context
from sqlalchemy import create_engine

from server.common.path import DATABASE_PATH
from server.configurations import get_server_settings
from server.repositories.database.backend import get_database
from server.repositories.database.postgres import PostgresRepository
from server.repositories.database.utils import normalize_sqlite_path
from server.repositories.schemas.models import Base
from server.repositories.schemas.types import JSONArray, JSONObject, UTCDateTime

config = context.config
target_metadata = Base.metadata


###############################################################################
def _database_url() -> str:
    settings = get_server_settings().database
    if settings.embedded_database:
        return f"sqlite:///{normalize_sqlite_path(DATABASE_PATH)}"
    repository = PostgresRepository(settings)
    try:
        return str(repository.engine.url)
    finally:
        repository.engine.dispose()


###############################################################################
def _render_item(
    type_: str,
    obj: Any,
    autogen_context: Any,
) -> str | Literal[False]:
    del autogen_context
    if type_ != "type":
        return False
    if isinstance(obj, UTCDateTime):
        return "sa.DateTime(timezone=True)"
    if isinstance(obj, (JSONObject, JSONArray)):
        return "sa.JSON()"
    return False


###############################################################################
def _include_object(
    object_: Any,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to: Any,
) -> bool:
    del object_, reflected, compare_to
    return not (type_ == "table" and name == "alembic_version")


###############################################################################
def _configure(connection: Any) -> None:
    version_table_pk = config.get_main_option("version_table_pk") or "true"
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=connection.dialect.name == "sqlite",
        render_item=_render_item,
        include_object=_include_object,
        version_table=config.get_main_option("version_table") or "alembic_version",
        version_table_pk=version_table_pk.lower() == "true",
    )


###############################################################################
def _run_migrations(connection: Any) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


###############################################################################
def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url") or _database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_item=_render_item,
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


###############################################################################
def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run_migrations(connection)
        return

    settings = get_server_settings().database
    if settings.embedded_database:
        # Direct Alembic CLI commands use a migration-only SQLite engine so
        # batch rebuilds can run without the application FK event listener.
        migration_engine = create_engine(
            _database_url(),
            future=True,
            connect_args={
                "autocommit": False,
                "timeout": settings.connect_timeout,
            },
        )
        try:
            with migration_engine.connect() as owned_connection:
                _run_migrations(owned_connection)
        finally:
            migration_engine.dispose()
        return

    # PostgreSQL CLI commands use the normal application engine. Lifecycle
    # initialization supplies a locked connection for startup/install runs.
    database = get_database()
    with database.engine.connect() as owned_connection:
        _run_migrations(owned_connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
