from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from alembic import command
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session
import pytest

from server.configurations import DatabaseSettings
from server.repositories.database import initializer
from server.repositories.database import migrations
from server.repositories.database import sqlite as sqlite_repository
from server.repositories.database.migrations import DatabaseMigrationError
from server.repositories.database.sqlite import SQLiteRepository

###############################################################################
def _settings() -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=True,
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=5,
        insert_batch_size=1000,
    )

###############################################################################
def _configure_database(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
) -> DatabaseSettings:
    settings = _settings()
    monkeypatch.setattr(initializer, "DATABASE_PATH", path)
    monkeypatch.setattr(sqlite_repository, "DATABASE_PATH", path)
    monkeypatch.setattr(
        initializer,
        "get_server_settings",
        lambda: SimpleNamespace(database=settings),
    )
    return settings

###############################################################################
def _head() -> str:
    return migrations._migration_directory(migrations.build_alembic_config()).get_heads()[0]

###############################################################################
def _revision(path: Path) -> str | None:
    engine = create_engine(f"sqlite:///{path}", future=True)
    try:
        tables = inspect(engine).get_table_names()
        if "alembic_version" not in tables:
            return None
        with engine.connect() as connection:
            return connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one_or_none()
    finally:
        engine.dispose()

###############################################################################
def test_repeated_initialization_is_current_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "database.db"
    _configure_database(monkeypatch, path)

    initializer.run_database_initialization()
    first_size = path.stat().st_size
    initializer.run_database_initialization()

    assert _revision(path) == _head()
    assert path.stat().st_size == first_size

###############################################################################
def test_versioned_pre_cleanup_revision_upgrades_and_preserves_metric_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "database.db"
    settings = _configure_database(monkeypatch, path)

    repository = SQLiteRepository(settings, enforce_foreign_keys=False, begin_immediate=True)
    try:
        with repository.engine.connect() as connection:
            with connection.begin():
                config = migrations.build_alembic_config()
                config.attributes["connection"] = connection
                command.upgrade(config, "0002_current_schema")
            with connection.begin():
                now = datetime.now(timezone.utc)
                dataset_id = connection.execute(
                    text(
                        "INSERT INTO dataset (name, status, document_count, created_at, updated_at, ready_at) "
                        "VALUES ('preserved', 'ready', 1, :now, :now, :now) RETURNING id"
                    ),
                    {"now": now},
                ).scalar_one()
                document_id = connection.execute(
                    text(
                        "INSERT INTO dataset_document (dataset_id, ordinal, text) "
                        "VALUES (:dataset_id, 0, 'document') RETURNING id"
                    ),
                    {"dataset_id": dataset_id},
                ).scalar_one()
                session_id = connection.execute(
                    text(
                        "INSERT INTO analysis_session "
                        "(dataset_id, status, report_version, created_at, completed_at, parameters, selected_metric_keys) "
                        "VALUES (:dataset_id, 'completed', 2, :now, :now, '{}', '[]') RETURNING id"
                    ),
                    {"dataset_id": dataset_id, "now": now},
                ).scalar_one()
                metric_type_id = connection.execute(
                    text(
                        "INSERT INTO metric_type "
                        "(key, category, label) VALUES ('metric', 'test', 'Metric') RETURNING id"
                    )
                ).scalar_one()
                connection.execute(
                    text(
                        "INSERT INTO metric_value "
                        "(session_id, dataset_id, metric_type_id, document_id, numeric_value, created_at) "
                        "VALUES (:session_id, :dataset_id, :metric_type_id, :document_id, 1.5, :now)"
                    ),
                    {
                        "session_id": session_id,
                        "dataset_id": dataset_id,
                        "metric_type_id": metric_type_id,
                        "document_id": document_id,
                        "now": now,
                    },
                )
    finally:
        repository.engine.dispose()

    initializer.run_database_initialization()

    engine = create_engine(f"sqlite:///{path}", future=True)
    try:
        with Session(engine) as session:
            assert session.execute(text("SELECT count(*) FROM dataset WHERE name='preserved'")).scalar_one() == 1
            assert session.execute(text("SELECT metric_key FROM metric_value")).scalar_one() == "metric"
        tables = inspect(engine).get_table_names()
        assert "metric_type" not in tables
        tokenizer_columns = {column["name"] for column in inspect(engine).get_columns("tokenizer")}
        assert "source" in tokenizer_columns
    finally:
        engine.dispose()
    assert _revision(path) == _head()

###############################################################################
def test_canonical_cleanup_purges_incompatible_derived_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "database.db"
    settings = _configure_database(monkeypatch, path)
    repository = SQLiteRepository(
        settings,
        enforce_foreign_keys=False,
        begin_immediate=True,
    )
    try:
        with repository.engine.connect() as connection:
            with connection.begin():
                config = migrations.build_alembic_config()
                config.attributes["connection"] = connection
                command.upgrade(config, "0002_current_schema")
            with connection.begin():
                now = datetime.now(timezone.utc)
                dataset_id = connection.execute(
                    text(
                        "INSERT INTO dataset "
                        "(name, status, document_count, created_at, updated_at, ready_at) "
                        "VALUES ('purge-me', 'ready', 0, :now, :now, :now) RETURNING id"
                    ),
                    {"now": now},
                ).scalar_one()
                connection.execute(
                    text(
                        "INSERT INTO analysis_session "
                        "(dataset_id, status, report_version, created_at, completed_at, parameters, selected_metric_keys) "
                        "VALUES (:dataset_id, 'completed', 1, :now, :now, '{}', '[]')"
                    ),
                    {"dataset_id": dataset_id, "now": now},
                )
                connection.execute(
                    text(
                        "INSERT INTO benchmark_report "
                        "(dataset_id, report_version, schema_version, methodology_version, created_at, status, "
                        "documents_processed, tokenizers_count, tokenizers_processed, selected_metric_keys, payload) "
                        "VALUES (:dataset_id, 4, 2, 'old', :now, 'completed', 0, 0, '[]', '[]', '{}')"
                    ),
                    {"dataset_id": dataset_id, "now": now},
                )
    finally:
        repository.engine.dispose()

    initializer.run_database_initialization()

    engine = create_engine(f"sqlite:///{path}", future=True)
    try:
        with engine.connect() as connection:
            assert connection.execute(text("SELECT count(*) FROM analysis_session")).scalar_one() == 0
            assert connection.execute(text("SELECT count(*) FROM benchmark_report")).scalar_one() == 0
    finally:
        engine.dispose()
    assert _revision(path) == _head()

###############################################################################
def test_nonempty_unversioned_database_is_rejected_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "database.db"
    _configure_database(monkeypatch, path)
    initializer.run_database_initialization()

    engine = create_engine(f"sqlite:///{path}", future=True)
    try:
        with engine.begin() as connection:
            connection.execute(text("DROP TABLE alembic_version"))
    finally:
        engine.dispose()
    before = path.read_bytes()

    with pytest.raises(DatabaseMigrationError, match="non-empty unversioned schema"):
        initializer.run_database_initialization()

    assert path.read_bytes() == before
    assert _revision(path) is None

###############################################################################
def test_unknown_unversioned_schema_is_rejected_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "database.db"
    _configure_database(monkeypatch, path)
    engine = create_engine(f"sqlite:///{path}", future=True)
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)"))
    finally:
        engine.dispose()
    before = path.read_bytes()

    with pytest.raises(DatabaseMigrationError, match="non-empty unversioned schema"):
        initializer.run_database_initialization()

    assert path.read_bytes() == before
    assert _revision(path) is None

###############################################################################
def test_database_ahead_of_repository_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "database.db"
    _configure_database(monkeypatch, path)
    initializer.run_database_initialization()
    engine = create_engine(f"sqlite:///{path}", future=True)
    try:
        with engine.begin() as connection:
            connection.execute(
                text("UPDATE alembic_version SET version_num='9999_future'")
            )
    finally:
        engine.dispose()

    with pytest.raises(DatabaseMigrationError, match="unknown Alembic revisions"):
        initializer.run_database_initialization()

    assert _revision(path) == "9999_future"

###############################################################################
def test_concurrent_initializers_serialize_on_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "database.db"
    _configure_database(monkeypatch, path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(initializer.run_database_initialization)
            for _ in range(2)
        ]
        for future in futures:
            future.result()

    assert _revision(path) == _head()
