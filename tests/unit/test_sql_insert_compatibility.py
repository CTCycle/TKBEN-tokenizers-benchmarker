from __future__ import annotations

import pytest

from TKBEN.server.repositories.database.backend import database
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
