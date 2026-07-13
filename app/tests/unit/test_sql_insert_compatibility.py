from __future__ import annotations

import math

import pytest
import sqlalchemy
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from server.repositories.database.backend import get_database
from server.repositories.serialization.data import DatasetSerializer
from server.repositories.schemas.models import Base, Dataset, Tokenizer
from server.services.benchmarks import BenchmarkService
from server.services.tokenizers import TokenizersService

###############################################################################
class FakeQueries:

    # -------------------------------------------------------------------------
    def __init__(self, engine: sqlalchemy.Engine) -> None:
        self.engine = engine

###############################################################################
def test_dataset_serializer_ensure_dataset_id_is_idempotent() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    serializer = DatasetSerializer(queries=FakeQueries(engine))

    first_id = serializer.ensure_dataset_id("wikitext/wikitext-2-v1")
    second_id = serializer.ensure_dataset_id("wikitext/wikitext-2-v1")

    assert first_id == second_id
    with Session(bind=engine) as session:
        count = session.execute(select(sqlalchemy.func.count(Dataset.id))).scalar_one()
    assert int(count) == 1

###############################################################################
def test_tokenizers_service_insert_if_missing_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)
    service = TokenizersService()

    service.insert_tokenizer_if_missing("bert-base-uncased")
    service.insert_tokenizer_if_missing("bert-base-uncased")

    with Session(bind=engine) as session:
        rows = session.execute(select(Tokenizer)).scalars().all()
    assert len(rows) == 1
    assert rows[0].name == "bert-base-uncased"

###############################################################################
def test_benchmark_service_ensure_tokenizer_ids_returns_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)
    service = BenchmarkService()

    mapping = service.ensure_tokenizer_ids(["tok/a", "tok/b", "tok/a"])

    assert set(mapping.keys()) == {"tok/a", "tok/b"}
    assert mapping["tok/a"] != mapping["tok/b"]
    with Session(bind=engine) as session:
        count = session.execute(
            select(sqlalchemy.func.count(Tokenizer.id))
        ).scalar_one()
    assert int(count) == 2

###############################################################################
###############################################################################
def test_session_report_rehydrates_json_metrics_when_numeric_is_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serializer = DatasetSerializer.__new__(DatasetSerializer)

    monkeypatch.setattr(
        serializer,
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
        serializer, "_load_histogram_rows_for_session", lambda session_id: {}
    )

    session_row = {
        "id": 123,
        "report_version": 2,
        "created_at": "2026-02-16T00:00:00Z",
        "dataset_name": "custom/tmp_zipf_cloud",
        "session_name": None,
        "selected_metric_keys": "[]",
        "parameters": "{}",
    }

    report = serializer._build_session_report_response(session_row)

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
