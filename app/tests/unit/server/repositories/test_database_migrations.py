# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine, inspect, text
import pytest

from server.configurations import DatabaseSettings
from server.repositories.database import TKBENDatabase, initializer
from server.repositories.database import migrations
from server.repositories.database.migrations import DatabaseMigrationError
from server.repositories.runtime_settings import DatabaseRuntimeSettingsPersistence

###############################################################################
def _settings(path: Path) -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=True,
        sqlite_path=path,
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
    settings = _settings(path)
    monkeypatch.setattr(
        initializer,
        "get_server_settings",
        lambda: SimpleNamespace(database=settings),
    )
    return settings

###############################################################################
def _head() -> str:
    return migrations._migration_directory(
        migrations.build_alembic_config()
    ).get_heads()[0]

###############################################################################
def _revision(path: Path) -> str | None:
    engine = create_engine(f"sqlite:///{path}", future=True)
    try:
        tables = inspect(engine).get_table_names()
        if "alembic_version" not in tables:
            return None
        with engine.connect() as connection:
            return connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one_or_none()
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
    assert _head() == "0001_current_schema"

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
def test_published_v460_schema_is_rebased_without_losing_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "database.db"
    settings = _configure_database(monkeypatch, path)
    initializer.run_database_initialization()

    database = TKBENDatabase(settings)
    persistence = DatabaseRuntimeSettingsPersistence(database)
    expected = {
        "schema_version": 1,
        "revision": 4,
        "dashboard_revision": 5,
        "overrides": {"datasets": {"histogram_bins": 31}},
        "dashboard_preferences": {
            "widgets": ["summary"],
            "baseline_report_ids": {"summary": "report-1"},
        },
    }
    persistence.save(expected)
    try:
        with database.backend.engine.begin() as connection:
            connection.execute(
                text("UPDATE alembic_version SET version_num=:revision"),
                {"revision": "0007_dashboard_ui_preferences"},
            )
    finally:
        database.backend.engine.dispose()

    initializer.run_database_initialization()

    database = TKBENDatabase(settings)
    try:
        with database.backend.engine.connect() as connection:
            assert connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one() == "0001_current_schema"
        assert DatabaseRuntimeSettingsPersistence(database).load() == expected
    finally:
        database.backend.engine.dispose()

###############################################################################
def test_published_revision_with_schema_drift_is_not_rebased(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "database.db"
    _configure_database(monkeypatch, path)
    initializer.run_database_initialization()

    engine = create_engine(f"sqlite:///{path}", future=True)
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE unexpected_schema (id INTEGER)"))
            connection.execute(
                text("UPDATE alembic_version SET version_num=:revision"),
                {"revision": "0007_dashboard_ui_preferences"},
            )
    finally:
        engine.dispose()

    with pytest.raises(DatabaseMigrationError, match="does not match the current schema"):
        initializer.run_database_initialization()

    assert _revision(path) == "0007_dashboard_ui_preferences"
    engine = create_engine(f"sqlite:///{path}", future=True)
    try:
        assert "unexpected_schema" in inspect(engine).get_table_names()
    finally:
        engine.dispose()

###############################################################################
def test_concurrent_initializers_serialize_on_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "database.db"
    _configure_database(monkeypatch, path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(initializer.run_database_initialization) for _ in range(2)
        ]
        for future in futures:
            future.result()

    assert _revision(path) == _head()
