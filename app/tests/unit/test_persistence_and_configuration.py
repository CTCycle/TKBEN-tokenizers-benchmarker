# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import math
from datetime import datetime
from datetime import timezone
import pytest
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy import select
from sqlalchemy.orm import Session
from server.repositories.database.backend import get_database
from server.repositories.datasets import DatasetRepository
from server.repositories.database.sqlite import SQLiteRepository
from server.repositories.schemas.models import AnalysisSession
from server.repositories.schemas.models import Base
from server.repositories.schemas.models import BenchmarkReport
from server.repositories.schemas.models import Dataset
from server.repositories.schemas.models import DatasetDocument
from server.repositories.schemas.models import Tokenizer
from server.repositories.schemas.models import TokenizerReport
from server.repositories.schemas.models import TokenizerVocabulary
from server.services.benchmarks import BenchmarkService
from server.repositories.tokenizers import TokenizerRepository
import hashlib
from pathlib import Path
from types import SimpleNamespace
from sqlalchemy import inspect
from sqlalchemy import text
from server.configurations import DatabaseSettings
from server.repositories.database import initializer
from server.repositories.database import migrations
from server.repositories.database.backend import build_sqlite_backend
from server.repositories.database.migrations import DatabaseMigrationError
from concurrent.futures import ThreadPoolExecutor
from server.repositories.database import TKBENDatabase
from server.repositories.runtime_settings import DatabaseRuntimeSettingsPersistence
from sqlalchemy.exc import IntegrityError
from server.repositories.schemas.models import MetricValue
from server.repositories.queries.data import DataRepositoryQueries
from server.configurations.runtime import RuntimeSettingsStore
from server.configurations.settings import DatabaseSettings as DatabaseSettings__runtime_settings_persistence
from server.configurations.settings import build_server_settings
from server.contracts.settings import DashboardUiPreferences
from server.repositories.database.initializer import initialize_database
from server.repositories.database.backend import TKBENDatabase as TKBENDatabase__runtime_settings_persistence
import os
from collections.abc import Iterator
import json
from server.configurations.database_configuration import DATABASE_CONFIGURATION_FILENAME
from server.configurations.database_configuration import DatabaseConfigurationStore
from server.configurations.database_configuration import StoredDatabaseConfiguration
from server.contracts.database import DatabaseConfigurationRequest
from typing import get_args
from pydantic import BaseModel
from pydantic import ValidationError
from server.configurations import runtime as runtime_module
from server.configurations.runtime import RuntimeSettingsConflictError
from server.configurations.runtime import RuntimeSettingsPersistenceError
from server.configurations.runtime import RuntimeSettingsValidationError
from server.configurations.settings import BenchmarkSettings
from server.configurations.settings import DatabaseSettings as DatabaseSettings__runtime_settings
from server.configurations.settings import DatasetSettings
from server.configurations.settings import JobsSettings
from server.configurations.settings import NetworkSettings
from server.configurations.settings import PathSettings
from server.configurations.settings import SecuritySettings
from server.configurations.settings import ServerSettings
from server.configurations.settings import TokenizerSettings
from server.contracts.settings import RuntimeSettingsPatchRequest
from server.common.path import ROOT_DIR
from server.common.utils import encryption
from server.common.utils.encryption import (
    KeyMaterialNotInitializedError,
    get_hf_key_cipher,
    initialize_key_material,
    is_key_material_initialized,
)
from server.configurations import environment as bootstrap
from server.configurations.startup import get_server_settings
from server.configurations.startup import is_key_reveal_enabled
from server.configurations.startup import reset_settings_cache_for_tests

###############################################################################


###############################################################################
class FakeQueries:

    # -------------------------------------------------------------------------
    def __init__(self, engine: sqlalchemy.Engine) -> None:
        self.engine = engine

###############################################################################
def test_dataset_repository_ensure_dataset_id_is_idempotent() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    repository = DatasetRepository(queries=FakeQueries(engine))

    first_id = repository.ensure_dataset_id("wikitext/wikitext-2-v1")
    second_id = repository.ensure_dataset_id("wikitext/wikitext-2-v1")

    assert first_id == second_id
    with Session(bind=engine) as session:
        count = session.execute(select(sqlalchemy.func.count(Dataset.id))).scalar_one()
    assert int(count) == 1

###############################################################################
def test_tokenizer_repository_insert_if_missing_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)
    repository = TokenizerRepository()

    repository.insert_if_missing("bert-base-uncased")
    repository.insert_if_missing("bert-base-uncased")

    with Session(bind=engine) as session:
        rows = session.execute(select(Tokenizer)).scalars().all()
    assert len(rows) == 1
    assert rows[0].name == "bert-base-uncased"
    assert rows[0].source == "huggingface"

###############################################################################
def test_dataset_delete_cascades_documents_sessions_and_benchmark_reports() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    event.listen(engine, "connect", SQLiteRepository.enable_foreign_keys)
    Base.metadata.create_all(engine, checkfirst=True)
    repository = DatasetRepository(queries=FakeQueries(engine))
    now = datetime.now(timezone.utc)

    with Session(bind=engine) as session:
        dataset = Dataset(
            name="custom/cascade",
            status="ready",
            document_count=1,
            created_at=now,
            updated_at=now,
            ready_at=now,
        )
        session.add(dataset)
        session.flush()
        session.add(DatasetDocument(dataset_id=dataset.id, ordinal=0, text="hello"))
        session.add(
            AnalysisSession(
                dataset_id=dataset.id,
                status="completed",
                report_version=2,
                created_at=now,
                completed_at=now,
                parameters={},
                selected_metric_keys=[],
            )
        )
        session.add(
            BenchmarkReport(
                dataset_id=dataset.id,
                report_version=1,
                schema_version=1,
                methodology_version="test",
                created_at=now,
                status="completed",
                documents_processed=1,
                tokenizers_count=0,
                tokenizers_processed=[],
                selected_metric_keys=[],
                payload={},
            )
        )
        session.commit()

    repository.delete_dataset("custom/cascade")

    with Session(bind=engine) as session:
        assert session.execute(select(Dataset)).scalars().all() == []
        assert session.execute(select(DatasetDocument)).scalars().all() == []
        assert session.execute(select(AnalysisSession)).scalars().all() == []
        assert session.execute(select(BenchmarkReport)).scalars().all() == []

###############################################################################
def test_tokenizer_delete_cascades_reports_and_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    event.listen(engine, "connect", SQLiteRepository.enable_foreign_keys)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)
    repository = TokenizerRepository()
    now = datetime.now(timezone.utc)

    with Session(bind=engine) as session:
        tokenizer = Tokenizer(name="custom/cascade", created_at=now)
        session.add(tokenizer)
        session.flush()
        session.add(
            TokenizerReport(
                tokenizer_id=tokenizer.id,
                report_version=1,
                created_at=now,
                metadata_json={},
                token_length_histogram={},
            )
        )
        session.add(
            TokenizerVocabulary(
                tokenizer_id=tokenizer.id,
                token_id=0,
                token="hello",
            )
        )
        session.commit()

    assert repository.delete_tokenizer("custom/cascade") is True
    assert repository.delete_tokenizer("custom/cascade") is False
    with Session(bind=engine) as session:
        assert session.execute(select(Tokenizer)).scalars().all() == []
        assert session.execute(select(TokenizerReport)).scalars().all() == []
        assert session.execute(select(TokenizerVocabulary)).scalars().all() == []

###############################################################################
def test_benchmark_repository_requires_existing_tokenizer_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)
    service = BenchmarkService()

    with pytest.raises(ValueError, match="do not exist"):
        service.repository.get_tokenizer_ids(["tok/a", "tok/b", "tok/a"])

    with Session(bind=engine) as session:
        now = datetime.now(timezone.utc)
        session.add_all(
            [
                Tokenizer(name="tok/a", created_at=now),
                Tokenizer(name="tok/b", created_at=now),
            ]
        )
        session.commit()

    mapping = service.repository.get_tokenizer_ids(["tok/a", "tok/b", "tok/a"])

    assert set(mapping.keys()) == {"tok/a", "tok/b"}
    assert mapping["tok/a"] != mapping["tok/b"]
    with Session(bind=engine) as session:
        count = session.execute(
            select(sqlalchemy.func.count(Tokenizer.id))
        ).scalar_one()
    assert int(count) == 2

###############################################################################
def test_session_report_preserves_native_json_metrics_when_numeric_is_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = DatasetRepository.__new__(DatasetRepository)

    monkeypatch.setattr(
        repository,
        "_load_metric_rows_for_session",
        lambda session_id: [
            {
                "document_id": None,
                "key": "words.zipf_curve",
                "numeric_value": math.nan,
                "text_value": None,
                "json_value": [{"rank": 1, "frequency": 9}],
            },
            {
                "document_id": None,
                "key": "words.word_cloud",
                "numeric_value": math.nan,
                "text_value": None,
                "json_value": [{"word": "hello", "count": 9, "weight": 100}],
            },
            {
                "document_id": None,
                "key": "words.most_common",
                "numeric_value": math.nan,
                "text_value": None,
                "json_value": [{"word": "hello", "count": 9}],
            },
            {
                "document_id": None,
                "key": "corpus.document_count",
                "numeric_value": 3.0,
                "text_value": None,
                "json_value": None,
            },
        ],
    )
    monkeypatch.setattr(
        repository, "_load_histogram_rows_for_session", lambda session_id: {}
    )

    session_row = {
        "id": 123,
        "report_version": 2,
        "created_at": "2026-02-16T00:00:00Z",
        "dataset_name": "custom/tmp_zipf_cloud",
        "session_name": None,
        "selected_metric_keys": [],
        "parameters": {},
    }

    report = repository._build_session_report_response(session_row)

    assert report["aggregate_statistics"]["words.zipf_curve"] == [
        {"rank": 1, "frequency": 9}
    ]
    assert report["aggregate_statistics"]["words.word_cloud"] == [
        {"word": "hello", "count": 9, "weight": 100}
    ]
    assert report["aggregate_statistics"]["words.most_common"] == [
        {"word": "hello", "count": 9}
    ]
    assert report["word_cloud_terms"] == [{"word": "hello", "count": 9, "weight": 100}]

###############################################################################
def test_session_report_rejects_json_encoded_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = DatasetRepository.__new__(DatasetRepository)
    monkeypatch.setattr(
        repository, "_load_metric_rows_for_session", lambda session_id: []
    )
    monkeypatch.setattr(
        repository, "_load_histogram_rows_for_session", lambda session_id: {}
    )

    with pytest.raises(ValueError, match="native JSON array"):
        repository._build_session_report_response(
            {
                "id": 123,
                "report_version": 2,
                "created_at": "2026-02-16T00:00:00Z",
                "dataset_name": "custom/encoded",
                "session_name": None,
                "selected_metric_keys": "[]",
                "parameters": {},
            }
        )

###############################################################################


###############################################################################
def _sqlite_settings(database_path: Path) -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=True,
        sqlite_path=database_path,
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
        sqlite_path=Path("database.db"),
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
def test_missing_sqlite_database_is_created_from_alembic_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "database.db"
    settings = _sqlite_settings(database_path)
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
                == migrations._migration_directory(
                    migrations.build_alembic_config()
                ).get_heads()[0]
            )
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
    settings = _sqlite_settings(database_path)
    monkeypatch.setattr(
        initializer,
        "get_server_settings",
        lambda: SimpleNamespace(database=settings),
    )

    with pytest.raises(DatabaseMigrationError):
        initializer.run_database_initialization()

    assert hashlib.sha256(database_path.read_bytes()).digest() == before

###############################################################################
def test_sqlite_backend_does_not_validate_existing_database(tmp_path: Path) -> None:
    database_path = tmp_path / "database.db"
    seed_engine = create_engine(f"sqlite:///{database_path}", future=True)
    Base.metadata.create_all(seed_engine)
    seed_engine.dispose()
    before = hashlib.sha256(database_path.read_bytes()).digest()

    backend = build_sqlite_backend(_sqlite_settings(database_path))
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

    ###############################################################################
    class FakeEngine:

        # -------------------------------------------------------------------------
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

        # -------------------------------------------------------------------------
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

###############################################################################


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
    assert _head() == "0002_managed_job_created_at_index"

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
            connection.execute(text("DROP INDEX ix_managed_job_created_at"))
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
            ).scalar_one() == "0002_managed_job_created_at_index"
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

###############################################################################


###############################################################################
@pytest.fixture()
def sqlite_session():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    event.listen(
        engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session

###############################################################################
def test_canonical_tables_and_removed_validation_report(
    sqlite_session: Session,
) -> None:
    tables = set(inspect(sqlite_session.bind).get_table_names())
    assert "dataset_validation_report" not in tables
    assert {
        "dataset",
        "dataset_document",
        "analysis_session",
        "metric_value",
        "histogram_artifact",
        "tokenizer",
        "tokenizer_vocabulary",
        "tokenizer_report",
        "benchmark_report",
        "hf_access_keys",
        "managed_job",
        "application_runtime_settings",
    } == tables

###############################################################################
def test_metric_values_require_one_value_and_dataset_safe_document(
    sqlite_session: Session,
) -> None:
    now = datetime.now(timezone.utc)
    first = Dataset(
        name="one",
        status="ready",
        document_count=1,
        created_at=now,
        updated_at=now,
        ready_at=now,
    )
    second = Dataset(
        name="two",
        status="ready",
        document_count=1,
        created_at=now,
        updated_at=now,
        ready_at=now,
    )
    sqlite_session.add_all([first, second])
    sqlite_session.flush()
    document = DatasetDocument(dataset_id=first.id, ordinal=0, text="text")
    session = AnalysisSession(
        dataset_id=first.id,
        status="completed",
        report_version=2,
        created_at=now,
        completed_at=now,
        parameters={},
        selected_metric_keys=[],
    )
    sqlite_session.add_all([document, session])
    sqlite_session.flush()
    with pytest.raises(IntegrityError):
        sqlite_session.add(
            MetricValue(
                session_id=session.id,
                dataset_id=second.id,
                metric_key="metric",
                document_id=document.id,
                numeric_value=1.0,
                created_at=now,
            )
        )
        sqlite_session.flush()
    sqlite_session.rollback()

###############################################################################
def test_dataset_catalog_filters_ready_rows_by_source_search_and_count(
    sqlite_session: Session,
) -> None:
    now = datetime.now(timezone.utc)
    sqlite_session.add_all(
        [
            Dataset(
                name="public/corpus",
                status="ready",
                document_count=12,
                created_at=now,
                updated_at=now,
                ready_at=now,
            ),
            Dataset(
                name="custom/small",
                status="ready",
                document_count=3,
                created_at=now,
                updated_at=now,
                ready_at=now,
            ),
            Dataset(
                name="custom/large",
                status="ready",
                document_count=20,
                created_at=now,
                updated_at=now,
                ready_at=now,
            ),
            Dataset(
                name="custom/loading",
                status="loading",
                document_count=1,
                created_at=now,
                updated_at=now,
            ),
        ]
    )
    sqlite_session.commit()

    database = SimpleNamespace(
        backend=SimpleNamespace(engine=sqlite_session.bind),
    )
    repository = DatasetRepository(DataRepositoryQueries(database))

    assert repository.list_dataset_previews(
        source="custom",
        document_count_operator="at_most",
        document_count=5,
    ) == [{"dataset_name": "custom/small", "document_count": 3}]
    assert repository.list_dataset_previews(
        search="PUBLIC",
        source="public",
    ) == [{"dataset_name": "public/corpus", "document_count": 12}]

###############################################################################
def test_dataset_report_by_id_excludes_incomplete_sessions(
    sqlite_session: Session,
) -> None:
    now = datetime.now(timezone.utc)
    dataset = Dataset(
        name="custom/report-status",
        status="ready",
        document_count=0,
        created_at=now,
        updated_at=now,
        ready_at=now,
    )
    sqlite_session.add(dataset)
    sqlite_session.flush()
    running = AnalysisSession(
        dataset_id=dataset.id,
        status="running",
        report_version=2,
        created_at=now,
        completed_at=None,
        parameters={},
        selected_metric_keys=[],
    )
    completed = AnalysisSession(
        dataset_id=dataset.id,
        status="completed",
        report_version=2,
        created_at=now,
        completed_at=now,
        parameters={},
        selected_metric_keys=[],
    )
    sqlite_session.add_all([running, completed])
    sqlite_session.commit()

    database = SimpleNamespace(
        backend=SimpleNamespace(engine=sqlite_session.bind),
    )
    repository = DatasetRepository(DataRepositoryQueries(database))

    assert repository.load_analysis_report_by_session_id(running.id) is None
    completed_report = repository.load_analysis_report_by_session_id(completed.id)
    assert completed_report is not None
    assert completed_report["report_id"] == completed.id

###############################################################################

# Copyright © 2026 Thomas Virdis


###############################################################################
def _database_settings(path: Path) -> DatabaseSettings__runtime_settings_persistence:
    return DatabaseSettings__runtime_settings_persistence(
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
        insert_batch_size=100,
    )


###############################################################################
def _initialized_store(path: Path) -> RuntimeSettingsStore:
    database_settings = _database_settings(path)
    settings = build_server_settings(database_settings=database_settings)
    initialize_database(settings=settings)
    database = TKBENDatabase__runtime_settings_persistence(database_settings)
    return RuntimeSettingsStore(
        settings,
        DatabaseRuntimeSettingsPersistence(database),
    )


###############################################################################
def test_runtime_settings_survive_restart_in_their_own_database(tmp_path: Path) -> None:
    first_path = tmp_path / "first.db"
    second_path = tmp_path / "second.db"
    first = _initialized_store(first_path)
    _initialized_store(second_path)

    updated = first.apply_patch(
        {"datasets": {"histogram_bins": 31}},
        expected_revision=0,
    )
    preferences = DashboardUiPreferences(
        ordered_widget_ids=["speed"],
        hidden_widget_ids=[],
        visualization_by_widget_id={"speed": "bar"},
        baseline_by_report_id={"7": "alpha"},
    )
    first.set_dashboard_preferences(preferences, expected_revision=0)
    restarted = _initialized_store(first_path)
    independent = _initialized_store(second_path)

    assert updated.revision == 1
    assert restarted.get_state().settings.datasets.histogram_bins == 31
    assert restarted.get_state().revision == 1
    assert restarted.get_dashboard_preferences_state().preferences == preferences
    assert restarted.get_dashboard_preferences_state().revision == 1
    assert independent.get_state().settings.datasets.histogram_bins == 20
    assert independent.get_state().revision == 0
    assert independent.get_dashboard_preferences_state().preferences == DashboardUiPreferences()
    assert independent.get_dashboard_preferences_state().revision == 0

###############################################################################


_RUNTIME_ENV_NAMES = (
    "TKBEN_PATHS_CONFIG",
    "FASTAPI_HOST",
    "FASTAPI_PORT",
    "UI_HOST",
    "UI_PORT",
    "VITE_API_BASE_URL",
    "ALLOW_KEY_REVEAL",
)
_BASE_ENVIRONMENT = {name: os.environ.get(name) for name in _RUNTIME_ENV_NAMES}

###############################################################################
@pytest.fixture(autouse=True)
def restore_runtime_environment() -> Iterator[None]:
    """Keep dotenv-backed settings tests isolated from process environment leaks."""
    for name in _RUNTIME_ENV_NAMES:
        os.environ.pop(name, None)

    yield

    for name in _RUNTIME_ENV_NAMES:
        original = _BASE_ENVIRONMENT[name]
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original

###############################################################################

# Copyright © 2026 Thomas Virdis


###############################################################################
def _store(tmp_path: Path) -> DatabaseConfigurationStore:
    material_path = tmp_path / "data" / "hf-key-material.json"
    initialize_key_material(material_path)
    return DatabaseConfigurationStore(tmp_path / "data", material_path)


###############################################################################
def test_default_configuration_is_file_based_and_independent(tmp_path: Path) -> None:
    store = _store(tmp_path)

    configuration = store.load()

    assert store.path.name == DATABASE_CONFIGURATION_FILENAME
    assert configuration.backend == "sqlite"
    assert store.to_database_settings(configuration).sqlite_path == (
        tmp_path / "data" / "database.db"
    ).resolve()
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert payload["backend"] == "sqlite"
    assert "FASTAPI_PORT" not in store.path.read_text(encoding="utf-8")


###############################################################################
def test_invalid_primary_file_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{broken", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Database configuration is invalid"):
        store.load()

    assert store.path.read_text(encoding="utf-8") == "{broken"


###############################################################################
def test_atomic_write_failure_preserves_last_valid_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    original = store.load()
    original_bytes = store.path.read_bytes()
    candidate = store.build_candidate(backend="sqlite", sqlite_path="alternate.db")

    def fail_replace(_source: str, _destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("server.configurations.database_configuration.os.replace", fail_replace)
    with pytest.raises(RuntimeError, match="Unable to persist database configuration"):
        store.save(candidate)

    assert store.load() == original
    assert store.path.read_bytes() == original_bytes
    assert not list(store.path.parent.glob(".database-config.json.*.tmp"))


###############################################################################
def test_candidate_does_not_reuse_password_for_a_different_target(tmp_path: Path) -> None:
    store = _store(tmp_path)
    initial = store.build_candidate(
        backend="postgresql",
        host="db1.example.test",
        port=5432,
        database_name="one",
        username="alice",
        password="secret-one",
    )
    store.save(initial)

    other = store.build_candidate(
        backend="postgresql",
        host="db2.example.test",
        port=5432,
        database_name="two",
        username="bob",
    )
    same_target = store.build_candidate(
        backend="postgresql",
        host="db1.example.test",
        port=5432,
        database_name="one",
        username="alice",
    )

    assert other.encrypted_password is None
    assert same_target.encrypted_password == initial.encrypted_password


###############################################################################
def test_sqlite_configuration_rejects_blank_paths_and_postgresql_fields() -> None:
    with pytest.raises(ValueError, match="SQLite database path is required"):
        DatabaseConfigurationRequest(backend="sqlite", sqlite_path="  ")

    with pytest.raises(ValueError, match="PostgreSQL fields are only valid"):
        DatabaseConfigurationRequest(
            backend="sqlite",
            sqlite_path="database.db",
            port=5432,
        )

    with pytest.raises(ValueError, match="PostgreSQL fields are not allowed"):
        StoredDatabaseConfiguration(
            schema_version=1,
            backend="sqlite",
            sqlite_path="database.db",
            ssl=True,
        )

###############################################################################


###############################################################################
class _FilePersistenceForStoreUnitTests:
    """Small test adapter; production persistence is the active SQL database."""

    # -------------------------------------------------------------------------
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fail_writes = False

    # -------------------------------------------------------------------------
    def load(self) -> dict[str, object] | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    # -------------------------------------------------------------------------
    def save(self, payload: dict[str, object]) -> None:
        if self.fail_writes:
            raise OSError("test persistence failure")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


###############################################################################
class _PersistedRuntimeSettingsStore(RuntimeSettingsStore):

    # -------------------------------------------------------------------------
    def __init__(self, defaults: ServerSettings, path: Path) -> None:
        self.persistence = _FilePersistenceForStoreUnitTests(path)
        self.path = path
        super().__init__(defaults, self.persistence)


###############################################################################
def _store__runtime_settings(defaults: ServerSettings, path: Path) -> _PersistedRuntimeSettingsStore:
    return _PersistedRuntimeSettingsStore(defaults, path)

###############################################################################
def _non_finite_runtime_float_cases():
    cases = []
    for group, group_field in RuntimeSettingsPatchRequest.model_fields.items():
        if group == "expected_revision":
            continue
        patch_model = next(
            option
            for option in get_args(group_field.annotation)
            if isinstance(option, type) and issubclass(option, BaseModel)
        )
        settings_model = ServerSettings.model_fields[group].annotation
        for name in patch_model.model_fields:
            default = settings_model.model_fields[name].default
            if isinstance(default, float):
                for value in (float("nan"), float("inf"), -float("inf")):
                    cases.append((group, name, value))
    return cases

###############################################################################
def _server_settings(tmp_path: Path) -> ServerSettings:
    data = tmp_path / "data"
    return ServerSettings(
        database=DatabaseSettings__runtime_settings(
            embedded_database=True,
            sqlite_path=data / "database.db",
            host=None,
            port=None,
            database_name=None,
            username=None,
            password=None,
            ssl=False,
            ssl_ca=None,
            connect_timeout=30,
            insert_batch_size=1000,
        ),
        paths=PathSettings(
            data=data,
            sources=data / "sources",
            datasets=data / "sources/datasets",
            tokenizers=data / "sources/tokenizers",
            logs=data / "logs",
            templates=data / "templates",
        ),
        network=NetworkSettings(
            fastapi_host="127.0.0.1",
            fastapi_port=5000,
            ui_host="127.0.0.1",
            ui_port=8000,
            api_base_url="/api",
        ),
        security=SecuritySettings(
            allow_key_reveal=False,
            hf_keys_encryption_material_file=data / "hf-key-material.json",
        ),
        datasets=DatasetSettings(),
        tokenizers=TokenizerSettings(),
        benchmarks=BenchmarkSettings(),
        jobs=JobsSettings(),
    )

###############################################################################
def test_default_only_startup_and_missing_runtime_file(tmp_path: Path) -> None:
    defaults = _server_settings(tmp_path)
    path = tmp_path / "runtime.json"

    store = _store__runtime_settings(defaults, path)
    state = store.get_state()

    assert state.settings == defaults
    assert state.defaults == defaults
    assert state.revision == 0
    assert state.overridden_keys == ()
    assert state.warning is None
    assert not path.exists()

###############################################################################
def test_valid_partial_overrides_merge_and_persist_sparsely(tmp_path: Path) -> None:
    defaults = _server_settings(tmp_path)
    path = tmp_path / "runtime.json"
    store = _store__runtime_settings(defaults, path)

    first = store.apply_patch(
        {"datasets": {"histogram_bins": 30}},
        expected_revision=0,
    )
    second = store.apply_patch(
        {"jobs": {"polling_interval": 2.0}},
        expected_revision=1,
    )

    assert first.settings.datasets.histogram_bins == 30
    assert second.settings.datasets.histogram_bins == 30
    assert second.settings.jobs.polling_interval == 2.0
    assert second.overridden_keys == (
        "datasets.histogram_bins",
        "jobs.polling_interval",
    )
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "dashboard_preferences": {
            "baseline_by_report_id": {},
            "tokenizer_colors": {},
            "hidden_widget_ids": [],
            "ordered_widget_ids": [],
            "visualization_by_widget_id": {},
        },
        "dashboard_revision": 0,
        "overrides": {
            "datasets": {"histogram_bins": 30},
            "jobs": {"polling_interval": 2.0},
        },
        "revision": 2,
        "schema_version": 1,
    }
    assert "network" not in path.read_text(encoding="utf-8")

###############################################################################
def test_dashboard_preferences_have_an_independent_revision_and_survive_reload(
    tmp_path: Path,
) -> None:
    defaults = _server_settings(tmp_path)
    path = tmp_path / "runtime.json"
    store = _store__runtime_settings(defaults, path)
    store.apply_patch({"datasets": {"histogram_bins": 30}}, expected_revision=0)
    preferences = DashboardUiPreferences(
        ordered_widget_ids=["speed", "size"],
        hidden_widget_ids=["size"],
        visualization_by_widget_id={"speed": "horizontal_bar"},
        baseline_by_report_id={"42": "tokenizer-a"},
    )

    saved = store.set_dashboard_preferences(preferences, expected_revision=0)
    restarted = _store__runtime_settings(defaults, path)

    assert saved.revision == 1
    assert store.get_state().revision == 1
    assert restarted.get_state().settings.datasets.histogram_bins == 30
    assert restarted.get_dashboard_preferences_state().revision == 1
    assert restarted.get_dashboard_preferences_state().preferences == preferences

###############################################################################
@pytest.mark.parametrize(
    "patch",
    [
        {"datasets": {"histogram_bins": 4}},
        {"datasets": {"histogram_bins": "30"}},
        {"settings": {"histogram_bins": 30}},
        {"network": {"fastapi_port": 9999}},
        {"datasets": {"unknown": 30}},
    ],
)
def test_invalid_unknown_and_environment_fields_are_rejected(
    tmp_path: Path,
    patch: dict[str, dict[str, object]],
) -> None:
    store = _store__runtime_settings(_server_settings(tmp_path), tmp_path / "runtime.json")

    with pytest.raises(RuntimeSettingsValidationError):
        store.apply_patch(patch, expected_revision=0)

    assert store.get_state().revision == 0
    assert not store.path.exists()

###############################################################################
def test_tokenizer_cross_field_validation_remains_authoritative(tmp_path: Path) -> None:
    store = _store__runtime_settings(_server_settings(tmp_path), tmp_path / "runtime.json")

    with pytest.raises(RuntimeSettingsValidationError):
        store.apply_patch(
            {"tokenizers": {"max_discovery_limit": 10}},
            expected_revision=0,
        )

    with pytest.raises(RuntimeSettingsValidationError):
        store.apply_patch(
            {"tokenizers": {"max_discovery_candidates": 20}},
            expected_revision=0,
        )

    assert store.get_state().settings.tokenizers.max_discovery_limit == 250

###############################################################################
@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": 2,
            "revision": 4,
            "overrides": {},
            "dashboard_revision": 0,
            "dashboard_preferences": {},
        },
        {
            "schema_version": 1,
            "revision": 4,
            "overrides": {"network": {}},
            "dashboard_revision": 0,
            "dashboard_preferences": {},
        },
        {
            "schema_version": 1,
            "revision": 4,
            "overrides": {"jobs": {"polling_interval": 0.1}},
            "dashboard_revision": 0,
            "dashboard_preferences": {},
        },
    ],
)
def test_invalid_schema_or_override_file_falls_back_with_warning(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    original = path.read_bytes()

    store = _store__runtime_settings(_server_settings(tmp_path), path)
    state = store.get_state()

    assert state.settings.datasets.histogram_bins == 20
    assert state.revision == 0
    assert state.overridden_keys == ()
    assert state.warning == runtime_module.RUNTIME_SETTINGS_WARNING
    assert path.read_bytes() == original

###############################################################################
def test_malformed_runtime_json_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    path.write_text("{not-json", encoding="utf-8")

    store = _store__runtime_settings(_server_settings(tmp_path), path)

    assert store.get_state().warning is not None
    assert path.read_text(encoding="utf-8") == "{not-json"

###############################################################################
def test_successful_save_replaces_corrupt_stored_document(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    path.write_text("{not-json", encoding="utf-8")

    store = _store__runtime_settings(_server_settings(tmp_path), path)
    state = store.apply_patch(
        {"datasets": {"histogram_bins": 20}},
        expected_revision=0,
    )

    assert state.warning is None
    assert state.revision == 1
    assert state.overridden_keys == ()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "dashboard_preferences": {
            "baseline_by_report_id": {},
            "tokenizer_colors": {},
            "hidden_widget_ids": [],
            "ordered_widget_ids": [],
            "visualization_by_widget_id": {},
        },
        "dashboard_revision": 0,
        "schema_version": 1,
        "revision": 1,
        "overrides": {},
    }

###############################################################################
def test_revision_conflict_does_not_write(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    store = _store__runtime_settings(_server_settings(tmp_path), path)
    store.apply_patch({"datasets": {"histogram_bins": 30}}, expected_revision=0)
    original = path.read_bytes()

    with pytest.raises(RuntimeSettingsConflictError) as error:
        store.apply_patch(
            {"datasets": {"histogram_bins": 31}},
            expected_revision=0,
        )

    assert error.value.actual_revision == 1
    assert path.read_bytes() == original
    assert store.get_state().settings.datasets.histogram_bins == 30

###############################################################################
def test_reset_one_value_and_equal_default_remove_the_override(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    store = _store__runtime_settings(_server_settings(tmp_path), path)
    store.apply_patch(
        {
            "datasets": {"histogram_bins": 30, "streaming_batch_size": 20_000},
            "jobs": {"polling_interval": 2.0},
        },
        expected_revision=0,
    )

    reset_one = store.reset(
        expected_revision=1,
        keys=["datasets.histogram_bins"],
    )
    assert reset_one.settings.datasets.histogram_bins == 20
    assert reset_one.settings.datasets.streaming_batch_size == 20_000
    assert reset_one.overridden_keys == (
        "datasets.streaming_batch_size",
        "jobs.polling_interval",
    )

    reset_to_default = store.apply_patch(
        {"datasets": {"streaming_batch_size": 10_000}},
        expected_revision=2,
    )
    assert reset_to_default.settings.datasets.streaming_batch_size == 10_000
    assert reset_to_default.overridden_keys == ("jobs.polling_interval",)

###############################################################################
def test_reset_all_persists_an_empty_database_document(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    store = _store__runtime_settings(_server_settings(tmp_path), path)
    store.apply_patch({"datasets": {"histogram_bins": 30}}, expected_revision=0)

    state = store.reset(expected_revision=1)

    assert state.settings.datasets.histogram_bins == 20
    assert state.overridden_keys == ()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "dashboard_preferences": {
            "baseline_by_report_id": {},
            "tokenizer_colors": {},
            "hidden_widget_ids": [],
            "ordered_widget_ids": [],
            "visualization_by_widget_id": {},
        },
        "dashboard_revision": 0,
        "schema_version": 1,
        "revision": 2,
        "overrides": {},
    }

###############################################################################
def test_reset_all_keeps_revision_of_an_unchanged_empty_document(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    payload = {
        "schema_version": 1,
        "revision": 3,
        "overrides": {},
        "dashboard_revision": 0,
        "dashboard_preferences": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    store = _store__runtime_settings(_server_settings(tmp_path), path)

    state = store.reset(expected_revision=3)

    assert state.revision == 3
    assert state.overridden_keys == ()
    assert json.loads(path.read_text(encoding="utf-8")) == payload

###############################################################################
def test_persistence_failure_rolls_back_state_and_stored_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.json"
    store = _store__runtime_settings(_server_settings(tmp_path), path)
    store.apply_patch({"datasets": {"histogram_bins": 30}}, expected_revision=0)
    original = path.read_bytes()

    store.persistence.fail_writes = True
    with pytest.raises(RuntimeSettingsPersistenceError):
        store.apply_patch(
            {"datasets": {"histogram_bins": 31}},
            expected_revision=1,
        )

    assert path.read_bytes() == original
    assert store.get_state().settings.datasets.histogram_bins == 30
    assert store.get_state().revision == 1

###############################################################################
def test_process_restart_and_reload_load_persisted_overrides(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    defaults = _server_settings(tmp_path)
    first = _store__runtime_settings(defaults, path)
    first.apply_patch(
        {"datasets": {"histogram_bins": defaults.datasets.histogram_bins + 1}},
        expected_revision=0,
    )

    sparse_file = json.loads(path.read_text(encoding="utf-8"))
    assert sparse_file["overrides"] == {
        "datasets": {"histogram_bins": defaults.datasets.histogram_bins + 1}
    }

    restarted = _store__runtime_settings(defaults, path)
    restored = restarted.get_state()
    assert (
        restored.settings.datasets.histogram_bins
        == defaults.datasets.histogram_bins + 1
    )
    assert (
        restored.settings.datasets.model_copy(
            update={"histogram_bins": defaults.datasets.histogram_bins}
        )
        == defaults.datasets
    )
    assert restored.settings.tokenizers == defaults.tokenizers
    assert restored.settings.benchmarks == defaults.benchmarks
    assert restored.settings.jobs == defaults.jobs
    assert restarted.get_state().revision == 1
    assert restored.warning is None

###############################################################################
@pytest.mark.parametrize(
    ("group", "name", "value"),
    _non_finite_runtime_float_cases(),
)
def test_non_finite_float_update_preserves_persisted_snapshot(
    tmp_path: Path,
    group: str,
    name: str,
    value: float,
) -> None:
    defaults = _server_settings(tmp_path)
    path = tmp_path / "runtime.json"
    store = _store__runtime_settings(defaults, path)
    store.apply_patch(
        {
            "benchmarks": {
                "default_batch_size": defaults.benchmarks.default_batch_size + 1
            }
        },
        expected_revision=0,
    )
    before = store.get_state()
    persisted_before = path.read_bytes()
    patch = {group: {name: value}}
    patch.setdefault("datasets", {})["histogram_bins"] = (
        defaults.datasets.histogram_bins + 1
    )

    with pytest.raises(RuntimeSettingsValidationError):
        store.apply_patch(patch, expected_revision=before.revision)

    assert store.get_state() == before
    assert path.read_bytes() == persisted_before

###############################################################################
def test_runtime_override_models_are_frozen(tmp_path: Path) -> None:
    store = _store__runtime_settings(_server_settings(tmp_path), tmp_path / "runtime.json")
    state = store.apply_patch(
        {"datasets": {"histogram_bins": 30}},
        expected_revision=0,
    )

    with pytest.raises(ValidationError):
        state.settings.datasets.histogram_bins = 20

###############################################################################
def test_benchmark_defaults_persist_sparsely_and_reset_independently(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.json"
    store = _store__runtime_settings(_server_settings(tmp_path), path)

    saved = store.apply_patch(
        {
            "benchmarks": {
                "default_max_documents": 2500,
                "default_batch_size": 32,
                "default_parallelism": 4,
            }
        },
        expected_revision=0,
    )

    assert saved.settings.benchmarks.default_max_documents == 2500
    assert saved.settings.benchmarks.default_batch_size == 32
    assert saved.settings.benchmarks.default_parallelism == 4
    assert saved.overridden_keys == (
        "benchmarks.default_max_documents",
        "benchmarks.default_batch_size",
        "benchmarks.default_parallelism",
    )
    assert json.loads(path.read_text(encoding="utf-8"))["overrides"] == {
        "benchmarks": {
            "default_batch_size": 32,
            "default_max_documents": 2500,
            "default_parallelism": 4,
        }
    }

    reset = store.reset(
        expected_revision=1,
        keys=["benchmarks.default_batch_size"],
    )
    assert reset.settings.benchmarks.default_batch_size == 16
    assert reset.settings.benchmarks.default_max_documents == 2500
    assert reset.settings.benchmarks.default_parallelism == 4

###############################################################################


###############################################################################
@pytest.fixture(autouse=True)
def reset_configuration_state() -> Iterator[None]:
    previous_cors_origins = os.environ.pop("CORS_ORIGINS", None)
    reset_settings_cache_for_tests()
    bootstrap.reset_environment_bootstrap_for_tests()
    yield
    reset_settings_cache_for_tests()
    bootstrap.reset_environment_bootstrap_for_tests()
    os.environ.pop("CORS_ORIGINS", None)
    if previous_cors_origins is not None:
        os.environ["CORS_ORIGINS"] = previous_cors_origins

###############################################################################
def _write_env(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

###############################################################################
def _configure_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data_dir: str | Path,
    log_dir: str | Path | None = None,
) -> None:
    config_path = tmp_path / "paths-config.json"
    payload = {"data_dir": str(data_dir)}
    if log_dir is not None:
        payload["log_dir"] = str(log_dir)
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("TKBEN_PATHS_CONFIG", str(config_path))

###############################################################################
def _configure_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_lines: list[str],
    *,
    data_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
) -> None:
    _configure_paths(tmp_path, monkeypatch, data_dir or tmp_path / "runtime-data", log_dir)
    env_path = tmp_path / ".env"
    _write_env(env_path, env_lines)
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)
    reset_settings_cache_for_tests()
    bootstrap.reset_environment_bootstrap_for_tests()

###############################################################################
def test_bootstrap_preserves_existing_process_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    _write_env(
        env_path,
        ["FASTAPI_HOST=from_dotenv", "UI_HOST=from-dotenv-ui"],
    )
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)
    monkeypatch.setenv("FASTAPI_HOST", "from_process")
    monkeypatch.setenv("UI_HOST", "from-process-ui")

    bootstrap.ensure_environment_loaded()

    assert os.getenv("FASTAPI_HOST") == "from_process"
    assert os.getenv("UI_HOST") == "from-process-ui"

###############################################################################
def test_missing_environment_is_created_from_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    template_path = tmp_path / ".env.example"
    template_bytes = b"FASTAPI_HOST=from_template\n"
    template_path.write_bytes(template_bytes)
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)
    monkeypatch.setattr(bootstrap, "ENV_EXAMPLE_FILE_PATH", template_path)

    assert bootstrap.ensure_environment_loaded() == env_path
    assert env_path.read_bytes() == template_bytes
    assert os.getenv("FASTAPI_HOST") == "from_template"

###############################################################################
def test_environment_template_exposes_canonical_runtime_inputs() -> None:
    example = (ROOT_DIR / "settings/.env.example").read_text(encoding="utf-8")

    for removed in (
        "TKBEN_DATA_DIR",
        "TKBEN_LOG_DIR",
        "HF_KEYS_ENCRYPTION_MATERIAL_FILE",
        "KERAS_BACKEND",
        "MPLBACKEND",
    ):
        assert removed not in example
    assert "UI_HOST=127.0.0.1" in example
    assert "CORS_ORIGINS=\n" in example
    assert "DATABASE_" not in example
    assert "ALLOW_KEY_REVEAL=false" in example

###############################################################################
def test_default_runtime_paths_use_repository_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path, [])
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)
    monkeypatch.setenv("TKBEN_PATHS_CONFIG", str(tmp_path / "missing-paths.json"))
    reset_settings_cache_for_tests()
    bootstrap.reset_environment_bootstrap_for_tests()

    settings = get_server_settings()
    default_root = (ROOT_DIR / "data").resolve()

    assert settings.paths.data == default_root
    assert settings.paths.logs == (default_root / "logs").resolve()
    assert settings.database.sqlite_path == (default_root / "database.db").resolve()
    assert settings.network.cors_origins is None
    assert settings.security.hf_keys_encryption_material_file == (
        default_root / "hf-key-material.json"
    ).resolve()

###############################################################################
def test_bootstrap_is_idempotent_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path, ["FASTAPI_HOST=first"])
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)

    bootstrap.ensure_environment_loaded()
    _write_env(env_path, ["FASTAPI_HOST=second"])
    bootstrap.ensure_environment_loaded()

    assert os.getenv("FASTAPI_HOST") == "first"

###############################################################################
def test_settings_initialization_refreshes_after_explicit_bootstrap_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path, ["FASTAPI_PORT=5101"])
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)

    first = get_server_settings()

    _write_env(env_path, ["FASTAPI_PORT=5102"])
    reset_settings_cache_for_tests()
    bootstrap.reset_environment_bootstrap_for_tests()

    refreshed = get_server_settings()

    assert first.network.fastapi_port == 5101
    assert refreshed.network.fastapi_port == 5102

###############################################################################
def test_typed_defaults_resolve_without_structured_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "runtime-data"
    _configure_environment(tmp_path, monkeypatch, [], data_dir=data_root)

    settings = get_server_settings()

    assert settings.tokenizers.default_discovery_limit == 50
    assert settings.tokenizers.max_discovery_limit == 250
    assert settings.tokenizers.max_discovery_candidates == 750
    assert settings.tokenizers.metadata_candidate_multiplier == 3
    assert settings.tokenizers.max_upload_bytes == 10 * 1024 * 1024
    assert settings.datasets.histogram_bins == 20
    assert settings.datasets.streaming_batch_size == 10_000
    assert settings.datasets.max_upload_bytes == 25 * 1024 * 1024
    assert settings.datasets.download_timeout_seconds == 180.0
    assert settings.datasets.download_retry_attempts == 3
    assert settings.datasets.download_retry_backoff_seconds == 2.0
    assert settings.benchmarks.streaming_batch_size == 1000
    assert settings.jobs.polling_interval == 1.0

###############################################################################
def test_runtime_environment_resolves_into_one_settings_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "runtime-data"
    log_root = tmp_path / "runtime-logs"
    _configure_environment(
        tmp_path,
        monkeypatch,
        [
            "FASTAPI_HOST=0.0.0.0",
            "FASTAPI_PORT=5100",
            "UI_HOST=localhost",
            "UI_PORT=8100",
            "VITE_API_BASE_URL=/api",
            "CORS_ORIGINS= https://Example.com, http://localhost:8000,https://example.com ",
            "ALLOW_KEY_REVEAL=true",
        ],
        data_dir=data_root,
        log_dir=log_root,
    )

    settings = get_server_settings()

    assert settings.paths.data == data_root.resolve()
    assert settings.paths.datasets == (data_root / "sources/datasets").resolve()
    assert settings.paths.tokenizers == (data_root / "sources/tokenizers").resolve()
    assert settings.paths.logs == log_root.resolve()
    assert settings.database.embedded_database is True
    assert settings.database.sqlite_path == (data_root / "database.db").resolve()
    assert settings.network.fastapi_host == "0.0.0.0"
    assert settings.network.fastapi_port == 5100
    assert settings.network.ui_host == "localhost"
    assert settings.network.ui_port == 8100
    assert settings.network.cors_origins == (
        "https://example.com",
        "http://localhost:8000",
    )
    assert settings.security.allow_key_reveal is True
    assert settings.security.hf_keys_encryption_material_file == (
        data_root / "hf-key-material.json"
    ).resolve()
    saved = (data_root / "database-config.json").read_text(encoding="utf-8")
    assert "secret" not in saved
    assert "DATABASE_" not in (tmp_path / ".env").read_text(encoding="utf-8")

###############################################################################
def test_blank_cors_origins_leave_origins_derived_from_ui_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_environment(
        tmp_path,
        monkeypatch,
        ["CORS_ORIGINS=", "UI_HOST=localhost", "UI_PORT=8100"],
    )

    settings = get_server_settings()

    assert settings.network.cors_origins is None
    assert settings.network.ui_host == "localhost"
    assert settings.network.ui_port == 8100

###############################################################################
@pytest.mark.parametrize(
    "value",
    [
        "*",
        "ftp://example.com",
        "https://example.com/api",
        "http://user@example.com",
        "http://example.com:70000",
        "http://example.com,,https://other.example",
    ],
)
def test_invalid_cors_origins_fail_during_settings_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    _configure_environment(tmp_path, monkeypatch, [f"CORS_ORIGINS={value}"])

    with pytest.raises(RuntimeError, match="CORS_ORIGINS"):
        get_server_settings()

###############################################################################
def test_relative_runtime_paths_are_repository_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_environment(
        tmp_path,
        monkeypatch,
        [],
        data_dir="custom/runtime-data",
        log_dir="custom/runtime-logs",
    )

    bootstrap.ensure_environment_loaded()
    settings = build_server_settings()

    assert settings.paths.data == (ROOT_DIR / "custom/runtime-data").resolve()
    assert settings.paths.logs == (ROOT_DIR / "custom/runtime-logs").resolve()

###############################################################################
def test_boolean_aliases_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_environment(
        tmp_path,
        monkeypatch,
        ["ALLOW_KEY_REVEAL=on"],
    )

    with pytest.raises(RuntimeError, match="ALLOW_KEY_REVEAL"):
        get_server_settings()

###############################################################################
def test_settings_models_are_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_environment(tmp_path, monkeypatch, [])
    settings = get_server_settings()

    with pytest.raises(ValidationError):
        settings.datasets.histogram_bins = 99

###############################################################################
def test_runtime_setting_models_keep_cross_field_and_polling_validation() -> None:
    with pytest.raises(ValidationError):
        TokenizerSettings(default_discovery_limit=251)
    with pytest.raises(ValidationError):
        TokenizerSettings(default_discovery_limit=20, max_discovery_limit=10)
    with pytest.raises(ValidationError):
        TokenizerSettings(max_discovery_limit=300, max_discovery_candidates=300)
    with pytest.raises(ValidationError):
        JobsSettings(polling_interval=0.1)

###############################################################################
def test_key_reveal_policy_uses_canonical_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_environment(
        tmp_path,
        monkeypatch,
        ["ALLOW_KEY_REVEAL=true"],
    )

    assert is_key_reveal_enabled() is True

###############################################################################
def test_hf_key_cipher_requires_explicit_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material_path = tmp_path / "hf-key-material.json"
    monkeypatch.setattr(encryption, "DATA_PATH", tmp_path)

    assert is_key_material_initialized() is False
    with pytest.raises(KeyMaterialNotInitializedError):
        get_hf_key_cipher()
    assert not material_path.exists()

    initialize_key_material()
    cipher = get_hf_key_cipher()

    assert is_key_material_initialized() is True
    assert cipher.decrypt(cipher.encrypt("hf_test")) == "hf_test"
    with pytest.raises(FileExistsError):
        initialize_key_material()
