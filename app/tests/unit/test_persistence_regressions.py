from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest
import sqlalchemy
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from server.repositories.database.backend import get_database
from server.repositories.datasets import DatasetRepository
from server.repositories.database.sqlite import SQLiteRepository
from server.repositories.schemas.models import (
    AnalysisSession,
    Base,
    BenchmarkReport,
    Dataset,
    DatasetDocument,
    Tokenizer,
    TokenizerReport,
    TokenizerVocabulary,
)
from server.services.benchmarks import BenchmarkService
from server.repositories.tokenizers import TokenizerRepository

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
        session.add(AnalysisSession(
            dataset_id=dataset.id,
            status="completed",
            report_version=2,
            created_at=now,
            completed_at=now,
            parameters={},
            selected_metric_keys=[],
        ))
        session.add(BenchmarkReport(
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
        ))
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
        session.add(TokenizerReport(
            tokenizer_id=tokenizer.id,
            report_version=1,
            created_at=now,
            metadata_json={},
            token_length_histogram={},
        ))
        session.add(TokenizerVocabulary(
            tokenizer_id=tokenizer.id,
            token_id=0,
            token="hello",
        ))
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
        session.add_all([
            Tokenizer(name="tok/a", created_at=now),
            Tokenizer(name="tok/b", created_at=now),
        ])
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
    monkeypatch.setattr(repository, "_load_metric_rows_for_session", lambda session_id: [])
    monkeypatch.setattr(repository, "_load_histogram_rows_for_session", lambda session_id: {})

    with pytest.raises(ValueError, match="native JSON array"):
        repository._build_session_report_response({
            "id": 123,
            "report_version": 2,
            "created_at": "2026-02-16T00:00:00Z",
            "dataset_name": "custom/encoded",
            "session_name": None,
            "selected_metric_keys": "[]",
            "parameters": {},
        })
