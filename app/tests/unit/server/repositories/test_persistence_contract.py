from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from server.repositories.schemas.models import (
    AnalysisSession,
    Base,
    Dataset,
    DatasetDocument,
    MetricValue,
)
from server.repositories.queries.data import DataRepositoryQueries
from server.repositories.datasets import DatasetRepository

###############################################################################
@pytest.fixture()
def sqlite_session():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    event.listen(engine, "connect", lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session

###############################################################################
def test_canonical_tables_and_removed_validation_report(sqlite_session: Session) -> None:
    tables = set(inspect(sqlite_session.bind).get_table_names())
    assert "dataset_validation_report" not in tables
    assert {
        "dataset", "dataset_document", "analysis_session",
        "metric_value", "histogram_artifact", "tokenizer", "tokenizer_vocabulary",
        "tokenizer_report", "benchmark_report", "hf_access_keys",
    } == tables

###############################################################################
def test_metric_values_require_one_value_and_dataset_safe_document(sqlite_session: Session) -> None:
    now = datetime.now(timezone.utc)
    first = Dataset(name="one", status="ready", document_count=1, created_at=now, updated_at=now, ready_at=now)
    second = Dataset(name="two", status="ready", document_count=1, created_at=now, updated_at=now, ready_at=now)
    sqlite_session.add_all([first, second])
    sqlite_session.flush()
    document = DatasetDocument(dataset_id=first.id, ordinal=0, text="text")
    session = AnalysisSession(dataset_id=first.id, status="completed", report_version=2, created_at=now, completed_at=now, parameters={}, selected_metric_keys=[])
    sqlite_session.add_all([document, session])
    sqlite_session.flush()
    with pytest.raises(IntegrityError):
        sqlite_session.add(MetricValue(session_id=session.id, dataset_id=second.id, metric_key="metric", document_id=document.id, numeric_value=1.0, created_at=now))
        sqlite_session.flush()
    sqlite_session.rollback()

###############################################################################
def test_dataset_catalog_filters_ready_rows_by_source_search_and_count(
    sqlite_session: Session,
) -> None:
    now = datetime.now(timezone.utc)
    sqlite_session.add_all([
        Dataset(name="public/corpus", status="ready", document_count=12, created_at=now, updated_at=now, ready_at=now),
        Dataset(name="custom/small", status="ready", document_count=3, created_at=now, updated_at=now, ready_at=now),
        Dataset(name="custom/large", status="ready", document_count=20, created_at=now, updated_at=now, ready_at=now),
        Dataset(name="custom/loading", status="loading", document_count=1, created_at=now, updated_at=now),
    ])
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
