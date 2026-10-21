# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from server.configurations.database_configuration import (
    DATABASE_CONFIGURATION_FILENAME,
    DatabaseConfigurationStore,
    StoredDatabaseConfiguration,
)
from server.contracts.database import DatabaseConfigurationRequest


###############################################################################
def _store(tmp_path: Path, env_text: str = "") -> DatabaseConfigurationStore:
    env_path = tmp_path / "settings" / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(env_text, encoding="utf-8")
    return DatabaseConfigurationStore(
        tmp_path / "data",
        tmp_path / "data" / "hf-key-material.json",
        env_path,
    )


###############################################################################
def test_default_configuration_is_file_based_and_independent(tmp_path: Path) -> None:
    store = _store(tmp_path, "FASTAPI_PORT=5000\n")

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
def test_legacy_postgresql_configuration_is_encrypted_and_removed_from_env(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path,
        "DATABASE_EMBEDDED=false\n"
        "DATABASE_HOST=db.example.test\n"
        "DATABASE_PORT=5544\n"
        "DATABASE_NAME=benchmark\n"
        "DATABASE_USERNAME=bench_user\n"
        "DATABASE_PASSWORD=correct-horse\n"
        "DATABASE_SSL=true\n"
        "DATABASE_CONNECT_TIMEOUT=17\n"
        "DATABASE_INSERT_BATCH_SIZE=321\n"
        "FASTAPI_PORT=5000\n",
    )

    configuration = store.load()
    settings = store.to_database_settings(configuration)

    assert settings.embedded_database is False
    assert settings.host == "db.example.test"
    assert settings.port == 5544
    assert settings.database_name == "benchmark"
    assert settings.username == "bench_user"
    assert settings.password == "correct-horse"
    assert settings.ssl is True
    assert settings.connect_timeout == 17
    assert settings.insert_batch_size == 321
    config_text = store.path.read_text(encoding="utf-8")
    assert "correct-horse" not in config_text
    assert configuration.encrypted_password
    assert "DATABASE_" not in store._env_file.read_text(encoding="utf-8")
    assert "FASTAPI_PORT=5000" in store._env_file.read_text(encoding="utf-8")


###############################################################################
def test_legacy_process_environment_is_migrated_once_and_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    monkeypatch.setenv("DATABASE_EMBEDDED", "false")
    monkeypatch.setenv("DATABASE_HOST", "db.example.test")
    monkeypatch.setenv("DATABASE_NAME", "benchmark")
    monkeypatch.setenv("DATABASE_USERNAME", "bench_user")

    configuration = store.load()

    assert configuration.backend == "postgresql"
    assert configuration.host == "db.example.test"
    assert configuration.database_name == "benchmark"
    assert configuration.username == "bench_user"
    assert all(name not in os.environ for name in tuple(os.environ) if name.startswith("DATABASE_"))


###############################################################################
def test_invalid_primary_file_fails_closed_without_env_fallback(tmp_path: Path) -> None:
    store = _store(tmp_path, "DATABASE_EMBEDDED=true\n")
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{broken", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Database configuration is invalid"):
        store.load()

    assert store.path.read_text(encoding="utf-8") == "{broken"
    assert "DATABASE_EMBEDDED=true" in store._env_file.read_text(encoding="utf-8")


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
