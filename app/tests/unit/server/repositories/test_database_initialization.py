from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.orm import Session

from server.configurations import DatabaseSettings
from server.repositories.database import initializer
from server.repositories.database import sqlite as sqlite_repository
from server.repositories.database.backend import build_sqlite_backend
from server.repositories.database.migrations import DatabaseMigrationError
from server.repositories.schemas.models import Base, MetricType
from server.services.metrics.catalog import DATASET_METRIC_CATALOG

###############################################################################
def _sqlite_settings() -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=True,
        engine=None,
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=1,
        insert_batch_size=100,
    )

###############################################################################
def _postgres_settings(*, host: str | None = "127.0.0.1") -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=False,
        engine="postgresql+psycopg",
        host=host,
        port=5432,
        database_name="tkben_test",
        username="postgres",
        password="secret",
        ssl=False,
        ssl_ca=None,
        connect_timeout=1,
        insert_batch_size=100,
    )

###############################################################################
def _patch_sqlite_path(
    monkeypatch: pytest.MonkeyPatch,
    database_path: Path,
) -> None:
    monkeypatch.setattr(initializer, "DATABASE_PATH", database_path)
    monkeypatch.setattr(sqlite_repository, "DATABASE_PATH", database_path)

###############################################################################
def test_missing_sqlite_database_is_created_and_seeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "database.db"
    settings = _sqlite_settings()
    _patch_sqlite_path(monkeypatch, database_path)
    monkeypatch.setattr(
        initializer,
        "get_server_settings",
        lambda: SimpleNamespace(database=settings),
    )

    initializer.run_database_initialization()

    assert database_path.is_file()
    engine = create_engine(f"sqlite:///{database_path}", future=True)
    try:
        assert set(inspect(engine).get_table_names()) == {
            *Base.metadata.tables,
            "alembic_version",
        }
        with engine.connect() as connection:
            assert (
                connection.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalar_one()
                == "0002_current_schema"
            )
        with Session(engine) as session:
            seeded_count = session.scalar(select(func.count(MetricType.id)))
        expected_count = sum(
            len(category.get("metrics", []))
            for category in DATASET_METRIC_CATALOG
        )
        assert seeded_count == expected_count
    finally:
        engine.dispose()

###############################################################################
def test_unknown_existing_sqlite_database_is_rejected_without_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "database.db"
    database_path.write_bytes(b"existing database bytes")
    before = hashlib.sha256(database_path.read_bytes()).digest()
    settings = _sqlite_settings()
    _patch_sqlite_path(monkeypatch, database_path)
    monkeypatch.setattr(
        initializer,
        "get_server_settings",
        lambda: SimpleNamespace(database=settings),
    )

    with pytest.raises(DatabaseMigrationError):
        initializer.run_database_initialization()

    assert hashlib.sha256(database_path.read_bytes()).digest() == before

###############################################################################
def test_sqlite_backend_does_not_validate_existing_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "database.db"
    seed_engine = create_engine(f"sqlite:///{database_path}", future=True)
    # This fixture intentionally bypasses Alembic to verify that constructing
    # a repository does not mutate an existing database.
    Base.metadata.create_all(seed_engine)
    seed_engine.dispose()
    before = hashlib.sha256(database_path.read_bytes()).digest()
    _patch_sqlite_path(monkeypatch, database_path)

    backend = build_sqlite_backend(_sqlite_settings())
    with backend.engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    backend.engine.dispose()

    assert hashlib.sha256(database_path.read_bytes()).digest() == before

###############################################################################
def test_postgres_startup_runs_the_same_migration_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _postgres_settings()
    calls: list[str] = []
    monkeypatch.setattr(
        initializer,
        "get_server_settings",
        lambda: SimpleNamespace(database=settings),
    )
    monkeypatch.setattr(
        initializer,
        "connect_postgres_database",
        lambda received: pytest.fail(
            f"connection-only check was used for {received.database_name}"
        ),
    )
    monkeypatch.setattr(
        initializer,
        "ensure_postgres_database",
        lambda received: calls.append(f"ensure:{received.database_name}"),
    )

    class FakeEngine:
        def dispose(self) -> None:
            calls.append("dispose")

    monkeypatch.setattr(
        initializer,
        "PostgresRepository",
        lambda received: SimpleNamespace(engine=FakeEngine()),
    )
    monkeypatch.setattr(
        initializer,
        "run_locked_migrations",
        lambda engine, received, label, *, postgres: calls.append(
            f"migrate:{label}:{postgres}"
        ),
    )

    initializer.run_database_initialization(startup=True)

    assert calls == ["ensure:tkben_test", "migrate:tkben_test:True", "dispose"]

###############################################################################
def test_postgres_connection_check_executes_select_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statements: list[str] = []

    ###############################################################################
    class FakeConnection:

        # -------------------------------------------------------------------------
        def __enter__(self):
            return self

        # -------------------------------------------------------------------------
        def __exit__(self, exc_type, exc_value, traceback):
            return False

        # -------------------------------------------------------------------------
        def execute(self, statement):
            statements.append(str(statement))

    ###############################################################################
    class FakeEngine:

        # -------------------------------------------------------------------------
        def connect(self):
            return FakeConnection()

        def dispose(self):
            return None

    monkeypatch.setattr(
        initializer,
        "PostgresRepository",
        lambda settings: SimpleNamespace(engine=FakeEngine()),
    )

    initializer.connect_postgres_database(_postgres_settings())

    assert statements == ["SELECT 1"]

###############################################################################
def test_postgres_initialization_failure_is_returned_as_process_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _postgres_settings(host=None)
    monkeypatch.setattr(
        initializer,
        "get_server_settings",
        lambda: SimpleNamespace(database=settings),
    )

    with pytest.raises(DatabaseMigrationError):
        initializer.initialize_database()
