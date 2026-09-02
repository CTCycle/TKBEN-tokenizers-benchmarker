from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from server.common.tokenizer_metrics import compute_vocabulary_shape_metrics
from server.repositories.queries.data import DataRepositoryQueries
from server.repositories.schemas.models import Base, Tokenizer
from server.repositories.tokenizer_reports import TokenizerReportRepository


###############################################################################
def test_compute_vocabulary_shape_metrics_vectorized_summary() -> None:
    metrics = compute_vocabulary_shape_metrics(
        [
            {"token": "a"},
            {"token": "bb"},
            {"token": "ccc"},
            {"token": "dddd"},
        ]
    )

    assert metrics["token_length_std"] == pytest.approx(1.1180339887)
    assert metrics["token_length_p90"] == pytest.approx(3.7)
    assert metrics["token_length_cv"] == pytest.approx(0.4472135955)
    assert metrics["single_character_token_percentage"] == pytest.approx(25.0)


###############################################################################
def test_repository_persists_vocabulary_shape_metrics_in_report_json() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    tokenizer_name = "test/vectorized-metrics"
    with Session(engine) as session:
        session.add(
            Tokenizer(
                name=tokenizer_name,
                source="custom",
                created_at=datetime.now(timezone.utc),
            )
        )
        session.commit()

    repository = TokenizerReportRepository(
        DataRepositoryQueries(SimpleNamespace(backend=SimpleNamespace(engine=engine)))
    )
    report_id = repository.replace_report_and_vocabulary(
        tokenizer_name,
        {
            "report_version": 1,
            "global_stats": {
                "vocabulary_size": 4,
                "vocabulary_stats": {"mean_token_length": 2.5},
            },
            "token_length_histogram": {
                "bins": ["1", "2", "3", "4"],
                "counts": [1, 1, 1, 1],
                "bin_edges": [1.0, 2.0, 3.0, 4.0, 5.0],
                "min_length": 1,
                "max_length": 4,
                "mean_length": 2.5,
                "median_length": 2.5,
            },
        },
        [
            {"token_id": 0, "token": "a", "decoded_token": "a"},
            {"token_id": 1, "token": "bb", "decoded_token": "bb"},
            {"token_id": 2, "token": "ccc", "decoded_token": "ccc"},
            {"token_id": 3, "token": "dddd", "decoded_token": "dddd"},
        ],
    )

    loaded = repository.load_tokenizer_report_by_id(report_id)

    assert loaded is not None
    vocabulary_stats = loaded["global_stats"]["vocabulary_stats"]
    histogram = loaded["token_length_histogram"]
    assert vocabulary_stats["token_length_std"] == pytest.approx(1.1180339887)
    assert vocabulary_stats["token_length_p90"] == pytest.approx(3.7)
    assert vocabulary_stats["token_length_cv"] == pytest.approx(0.4472135955)
    assert vocabulary_stats["single_character_token_percentage"] == pytest.approx(25.0)
    assert histogram["token_length_p90"] == pytest.approx(3.7)
    assert histogram["single_character_token_percentage"] == pytest.approx(25.0)
