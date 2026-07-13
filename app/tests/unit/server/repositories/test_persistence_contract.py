from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from server.repositories.schemas.models import (
    AnalysisSession,
    Base,
    Dataset,
    DatasetDocument,
    MetricType,
    MetricValue,
)


@pytest.fixture()
def sqlite_session():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    event.listen(engine, "connect", lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_canonical_tables_and_removed_validation_report(sqlite_session: Session) -> None:
    tables = set(inspect(sqlite_session.bind).get_table_names())
    assert "dataset_validation_report" not in tables
    assert {
        "dataset", "dataset_document", "analysis_session", "metric_type",
        "metric_value", "histogram_artifact", "tokenizer", "tokenizer_vocabulary",
        "tokenizer_report", "benchmark_report", "hf_access_keys",
    } == tables


def test_metric_values_require_one_value_and_dataset_safe_document(sqlite_session: Session) -> None:
    now = datetime.now(timezone.utc)
    first = Dataset(name="one", status="ready", document_count=1, created_at=now, updated_at=now, ready_at=now)
    second = Dataset(name="two", status="ready", document_count=1, created_at=now, updated_at=now, ready_at=now)
    sqlite_session.add_all([first, second])
    sqlite_session.flush()
    document = DatasetDocument(dataset_id=first.id, ordinal=0, text="text")
    session = AnalysisSession(dataset_id=first.id, status="completed", report_version=1, created_at=now, completed_at=now, parameters={}, selected_metric_keys=[])
    metric = MetricType(key="metric", category="test", label="Metric", scope="per_document", value_kind="number")
    sqlite_session.add_all([document, session, metric])
    sqlite_session.flush()
    with pytest.raises(IntegrityError):
        sqlite_session.add(MetricValue(session_id=session.id, dataset_id=second.id, metric_type_id=metric.id, document_id=document.id, numeric_value=1.0, created_at=now))
        sqlite_session.flush()
    sqlite_session.rollback()
