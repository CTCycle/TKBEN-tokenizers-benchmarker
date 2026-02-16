from __future__ import annotations

import math

import pandas as pd
import pytest
import sqlalchemy

from TKBEN.server.repositories.database.backend import database
from TKBEN.server.repositories.database.postgres import PostgresRepository
from TKBEN.server.repositories.database.sqlite import SQLiteRepository
from TKBEN.server.repositories.serialization.data import DatasetSerializer
from TKBEN.server.services.benchmarks import BenchmarkService
from TKBEN.server.services.tokenizers import TokenizersService


###############################################################################
class FakeResult:
    def __init__(self, row: tuple | None = None) -> None:
        self.row = row

    # -------------------------------------------------------------------------
    def first(self) -> tuple | None:
        return self.row


###############################################################################
class SQLCaptureConnection:
    def __init__(self, state: dict[str, object]) -> None:
        self.state = state

    # -------------------------------------------------------------------------
    def __enter__(self) -> "SQLCaptureConnection":
        return self

    # -------------------------------------------------------------------------
    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    # -------------------------------------------------------------------------
    def execute(self, statement, params=None) -> FakeResult:  # noqa: ANN001
        sql = str(statement)
        self.state["queries"].append(sql)  # type: ignore[index]
        self.state["params"].append(dict(params or {}))  # type: ignore[index]

        if 'SELECT "id" FROM "dataset"' in sql:
            return FakeResult(row=(17,))
        if 'SELECT "id", "name" FROM "tokenizer"' in sql:
            name = str((params or {}).get("name", ""))
            name_to_id = {"tok/a": 101, "tok/b": 202}
            return FakeResult(row=(name_to_id.get(name, 0), name))
        return FakeResult()


###############################################################################
class SQLCaptureEngine:
    def __init__(self, state: dict[str, object]) -> None:
        self.state = state

    # -------------------------------------------------------------------------
    def begin(self) -> SQLCaptureConnection:
        return SQLCaptureConnection(self.state)

    # -------------------------------------------------------------------------
    def connect(self) -> SQLCaptureConnection:
        return SQLCaptureConnection(self.state)


###############################################################################
class FakeQueries:
    def __init__(self, engine: SQLCaptureEngine) -> None:
        self.engine = engine


###############################################################################
class InsertCaptureConnection:
    def __init__(self) -> None:
        self.statements: list[object] = []

    # -------------------------------------------------------------------------
    def __enter__(self) -> "InsertCaptureConnection":
        return self

    # -------------------------------------------------------------------------
    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    # -------------------------------------------------------------------------
    def execute(self, statement, params=None) -> FakeResult:  # noqa: ANN001
        del params
        self.statements.append(statement)
        return FakeResult()


###############################################################################
class InsertCaptureEngine:
    def __init__(self, connection: InsertCaptureConnection) -> None:
        self.connection = connection

    # -------------------------------------------------------------------------
    def begin(self) -> InsertCaptureConnection:
        return self.connection


###############################################################################
def build_metric_value_table() -> sqlalchemy.Table:
    metadata = sqlalchemy.MetaData()
    return sqlalchemy.Table(
        "metric_value",
        metadata,
        sqlalchemy.Column("session_id", sqlalchemy.Integer, nullable=False),
        sqlalchemy.Column("metric_type_id", sqlalchemy.Integer, nullable=False),
        sqlalchemy.Column("document_id", sqlalchemy.Integer, nullable=True),
        sqlalchemy.Column("numeric_value", sqlalchemy.Float, nullable=True),
        sqlalchemy.Column("text_value", sqlalchemy.String, nullable=True),
        sqlalchemy.Column("json_value", sqlalchemy.JSON, nullable=True),
    )


###############################################################################
def test_dataset_serializer_ensure_dataset_id_uses_conflict_safe_insert() -> None:
    state: dict[str, object] = {"queries": [], "params": []}
    serializer = DatasetSerializer(queries=FakeQueries(SQLCaptureEngine(state)))

    dataset_id = serializer.ensure_dataset_id("wikitext/wikitext-2-v1")

    assert dataset_id == 17
    insert_query = next(
        query for query in state["queries"]  # type: ignore[index]
        if 'INSERT INTO "dataset" ("name")' in query
    )
    assert 'ON CONFLICT ("name") DO NOTHING' in insert_query
    assert "WHERE NOT EXISTS" not in insert_query


###############################################################################
def test_tokenizers_service_insert_uses_conflict_safe_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, object] = {"queries": [], "params": []}
    service = TokenizersService()
    monkeypatch.setattr(database.backend, "engine", SQLCaptureEngine(state))

    service.insert_tokenizer_if_missing("bert-base-uncased")

    insert_query = next(
        query for query in state["queries"]  # type: ignore[index]
        if 'INSERT INTO "tokenizer" ("name")' in query
    )
    assert 'ON CONFLICT ("name") DO NOTHING' in insert_query
    assert "WHERE NOT EXISTS" not in insert_query


###############################################################################
def test_benchmark_service_ensure_tokenizer_ids_uses_conflict_safe_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, object] = {"queries": [], "params": []}
    service = BenchmarkService()
    monkeypatch.setattr(database.backend, "engine", SQLCaptureEngine(state))

    mapping = service.ensure_tokenizer_ids(["tok/a", "tok/b"])

    assert mapping == {"tok/a": 101, "tok/b": 202}
    insert_query = next(
        query for query in state["queries"]  # type: ignore[index]
        if 'INSERT INTO "tokenizer" ("name")' in query
    )
    assert 'ON CONFLICT ("name") DO NOTHING' in insert_query
    assert "WHERE NOT EXISTS" not in insert_query


###############################################################################
@pytest.mark.parametrize(
    "repository_class",
    [PostgresRepository, SQLiteRepository],
)
def test_insert_dataframe_without_ignore_duplicates_uses_sqlalchemy_insert(
    monkeypatch: pytest.MonkeyPatch,
    repository_class: type[PostgresRepository] | type[SQLiteRepository],
) -> None:
    connection = InsertCaptureConnection()
    repository = repository_class.__new__(repository_class)
    repository.engine = InsertCaptureEngine(connection)  # type: ignore[attr-defined]
    repository.insert_batch_size = 100  # type: ignore[attr-defined]

    table = build_metric_value_table()
    table_class = type("FakeMetricValueTableClass", (), {"__table__": table})

    monkeypatch.setattr(repository, "ensure_table_schema", lambda table_name: None)
    monkeypatch.setattr(repository, "get_table_class", lambda table_name: table_class)

    def fail_to_sql(self, *args, **kwargs) -> None:  # noqa: ANN001
        del self, args, kwargs
        raise AssertionError("DataFrame.to_sql should not be used for inserts.")

    monkeypatch.setattr(pd.DataFrame, "to_sql", fail_to_sql)

    frame = pd.DataFrame(
        [
            {
                "session_id": 1,
                "metric_type_id": 11,
                "document_id": 101,
                "numeric_value": 3.14,
                "text_value": None,
                "json_value": None,
            },
            {
                "session_id": 1,
                "metric_type_id": 12,
                "document_id": None,
                "numeric_value": None,
                "text_value": None,
                "json_value": [{"word": "alpha", "count": 2}],
            },
        ]
    )

    repository.insert_dataframe(frame, "metric_value", ignore_duplicates=False)  # type: ignore[attr-defined]

    assert len(connection.statements) == 1
    sql = str(connection.statements[0])
    assert "INSERT INTO metric_value" in sql
    assert "json_value" in sql


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
    monkeypatch.setattr(serializer, "_load_histogram_rows_for_session", lambda session_id: {})

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

    assert report["aggregate_statistics"]["words.zipf_curve"] == [{"rank": 1, "frequency": 9}]
    assert report["aggregate_statistics"]["words.word_cloud"] == [
        {"word": "hello", "count": 9, "weight": 100}
    ]
    assert report["aggregate_statistics"]["words.most_common"] == [{"word": "hello", "count": 9}]
    assert report["word_cloud_terms"] == [{"word": "hello", "count": 9, "weight": 100}]
