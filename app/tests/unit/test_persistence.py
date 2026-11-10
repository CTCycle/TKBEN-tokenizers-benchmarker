# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import math
import threading
from datetime import datetime
from datetime import timezone
import pytest
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy import select
from sqlalchemy.orm import Session
from server.repositories.database.backend import get_database
from server.repositories.datasets import (
    DatasetImportInProgressError,
    DatasetRepository,
)
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


###############################################################################
class FakeQueries:

    # -------------------------------------------------------------------------
    def __init__(self, engine: sqlalchemy.Engine) -> None:
        self.engine = engine


###############################################################################
def _claim_dataset_after_barrier(
    arguments: tuple[DatasetRepository, threading.Barrier],
) -> int | None:
    repository, ready = arguments
    ready.wait(timeout=2.0)
    try:
        return repository.begin_dataset_import("custom/concurrent")
    except DatasetImportInProgressError:
        return None


###############################################################################
def test_dataset_import_claim_rejects_active_and_replaces_ready_dataset() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    repository = DatasetRepository(queries=FakeQueries(engine))

    first_id = repository.ensure_dataset_id("wikitext/wikitext-2-v1")
    with pytest.raises(DatasetImportInProgressError, match="already being imported"):
        repository.ensure_dataset_id("wikitext/wikitext-2-v1")
    repository.finalize_dataset_import(first_id, 0)
    second_id = repository.ensure_dataset_id("wikitext/wikitext-2-v1")

    with Session(bind=engine) as session:
        rows = session.execute(select(Dataset.id, Dataset.status)).all()
    assert rows == [(second_id, "loading")]


###############################################################################
def test_concurrent_dataset_import_claims_have_one_owner(tmp_path: Path) -> None:
    database_path = tmp_path / "dataset-imports.sqlite3"
    engine = create_engine(f"sqlite+pysqlite:///{database_path}", future=True)
    event.listen(engine, "connect", SQLiteRepository.enable_foreign_keys)
    event.listen(engine, "begin", SQLiteRepository.begin_immediate)
    Base.metadata.create_all(engine, checkfirst=True)
    repository = DatasetRepository(queries=FakeQueries(engine))
    ready = threading.Barrier(2)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            arguments = [(repository, ready), (repository, ready)]
            claims = list(executor.map(_claim_dataset_after_barrier, arguments))

        assert sum(claim is not None for claim in claims) == 1
        assert sum(claim is None for claim in claims) == 1
    finally:
        engine.dispose()


###############################################################################
def test_incomplete_dataset_import_recovery_cascades_partial_rows() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    event.listen(engine, "connect", SQLiteRepository.enable_foreign_keys)
    Base.metadata.create_all(engine, checkfirst=True)
    now = datetime.now(timezone.utc)
    with Session(bind=engine) as session:
        loading = Dataset(
            name="custom/interrupted",
            status="loading",
            created_at=now,
            updated_at=now,
        )
        ready = Dataset(
            name="custom/ready",
            status="ready",
            document_count=1,
            created_at=now,
            updated_at=now,
            ready_at=now,
        )
        session.add_all([loading, ready])
        session.flush()
        session.add_all(
            [
                DatasetDocument(dataset_id=loading.id, ordinal=0, text="partial"),
                DatasetDocument(dataset_id=ready.id, ordinal=0, text="ready"),
            ]
        )
        session.commit()

    repository = DatasetRepository(queries=FakeQueries(engine))
    try:
        repository.delete_incomplete_dataset_imports()
        with Session(bind=engine) as session:
            assert session.execute(select(Dataset.name)).all() == [("custom/ready",)]
            assert session.execute(
                select(DatasetDocument.text).order_by(DatasetDocument.ordinal)
            ).all() == [("ready",)]
    finally:
        engine.dispose()


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
def test_sqlite_backend_serializes_concurrent_read_then_write(tmp_path: Path) -> None:
    database_path = tmp_path / "database.db"
    seed_engine = create_engine(f"sqlite:///{database_path}", future=True)
    Base.metadata.create_all(seed_engine)
    seed_engine.dispose()
    backend = build_sqlite_backend(_sqlite_settings(database_path))

    def read_then_write(name: str) -> None:
        with Session(backend.engine) as session:
            session.execute(select(Tokenizer.id)).all()
            session.add(Tokenizer(name=name, created_at=datetime.now(timezone.utc)))
            session.commit()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(read_then_write, f"tok-{i}") for i in range(16)]
        for future in futures:
            future.result()

    with Session(backend.engine) as session:
        assert len(session.execute(select(Tokenizer.id)).all()) == 16
    backend.engine.dispose()


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
            assert (
                connection.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalar_one()
                == "0002_managed_job_created_at_index"
            )
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

    with pytest.raises(
        DatabaseMigrationError, match="does not match the current schema"
    ):
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
