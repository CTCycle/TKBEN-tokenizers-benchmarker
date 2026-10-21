# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.configurations.database_configuration import (
    DATABASE_CONFIGURATION_FILENAME,
    DatabaseConfigurationStore,
    StoredDatabaseConfiguration,
)
from server.contracts.database import DatabaseConfigurationRequest


###############################################################################
def _store(tmp_path: Path) -> DatabaseConfigurationStore:
    return DatabaseConfigurationStore(
        tmp_path / "data",
        tmp_path / "data" / "hf-key-material.json",
    )


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
