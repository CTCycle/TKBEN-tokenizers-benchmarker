from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest
from server.common.path import ROOT_DIR
from server.common.utils.encryption import get_hf_key_cipher
from server.configurations import environment as bootstrap
from server.configurations import is_key_reveal_enabled
from server.configurations.startup import (
    get_configuration_manager,
    get_server_settings,
    reload_settings_for_tests,
)
from server.repositories.database.initializer import _resolve_postgres_engine

###############################################################################
@pytest.fixture(autouse=True)
def reset_configuration_state() -> None:
    reload_settings_for_tests()
    bootstrap.reset_environment_bootstrap_for_tests()
    yield
    reload_settings_for_tests()
    bootstrap.reset_environment_bootstrap_for_tests()

###############################################################################
def _write_env(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

###############################################################################
def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")

###############################################################################
def _minimal_config_json() -> dict[str, object]:
    return {
        "database": {"embedded_database": True},
        "datasets": {},
        "tokenizers": {},
        "benchmarks": {},
        "jobs": {"polling_interval": 1.0},
    }

###############################################################################
def test_bootstrap_environment_overrides_existing_process_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path, ["FASTAPI_HOST=from_dotenv"])

    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)
    monkeypatch.setenv("FASTAPI_HOST", "from_process")

    bootstrap.ensure_environment_loaded()

    assert os.getenv("FASTAPI_HOST") == "from_dotenv"

###############################################################################
def test_missing_environment_is_created_from_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    template_path = tmp_path / ".env.example"
    template_bytes = b"FASTAPI_HOST=from_template\n"
    template_path.write_bytes(template_bytes)

    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)
    monkeypatch.setattr(bootstrap, "ENV_EXAMPLE_FILE_PATH", template_path)

    assert bootstrap.ensure_environment_loaded() == env_path
    assert env_path.read_bytes() == template_bytes
    assert os.getenv("FASTAPI_HOST") == "from_template"

###############################################################################
def test_environment_template_exposes_resource_directory() -> None:
    example = (ROOT_DIR / "settings/.env.example").read_text(encoding="utf-8")

    assert "TKBEN_DATA_DIR=app/resources" in example

###############################################################################
def test_existing_environment_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    template_path = tmp_path / ".env.example"
    existing_bytes = b"FASTAPI_HOST=existing\n"
    env_path.write_bytes(existing_bytes)
    template_path.write_bytes(b"FASTAPI_HOST=template\n")

    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)
    monkeypatch.setattr(bootstrap, "ENV_EXAMPLE_FILE_PATH", template_path)

    bootstrap.ensure_environment_loaded()

    assert env_path.read_bytes() == existing_bytes
    assert os.getenv("FASTAPI_HOST") == "existing"

###############################################################################
def test_bootstrap_is_idempotent_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path, ["FASTAPI_HOST=first"])

    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)

    bootstrap.ensure_environment_loaded()
    _write_env(env_path, ["FASTAPI_HOST=second"])
    bootstrap.ensure_environment_loaded()

    assert os.getenv("FASTAPI_HOST") == "first"

###############################################################################
def test_server_package_import_bootstraps_env_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path, ["FASTAPI_HOST=192.168.1.1"])

    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)
    monkeypatch.setenv("FASTAPI_HOST", "127.0.0.1")

    import server as server_package

    importlib.reload(server_package)

    assert os.getenv("FASTAPI_HOST") == "192.168.1.1"

###############################################################################
def test_missing_configuration_file_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path, ["FASTAPI_HOST=127.0.0.1"])
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)

    with pytest.raises(RuntimeError, match="Configuration file not found"):
        _ = get_server_settings(config_path=tmp_path / "missing.json")

###############################################################################
def test_invalid_configuration_file_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text("{invalid-json", encoding="utf-8")

    env_path = tmp_path / ".env"
    _write_env(env_path, ["FASTAPI_HOST=127.0.0.1"])
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)

    with pytest.raises(RuntimeError, match="Unable to load configuration"):
        _ = get_server_settings(config_path=config_path)

###############################################################################
def test_json_owned_db_embedded_ignores_environment_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "configurations.json"
    _write_json(config_path, _minimal_config_json())

    env_path = tmp_path / ".env"
    _write_env(
        env_path,
        [
            "DB_EMBEDDED=false",
            "DB_ENGINE=postgresql+psycopg",
            "DB_HOST=remote-db",
            "DB_NAME=remote_db",
            "DB_USER=remote_user",
        ],
    )
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)

    settings = get_server_settings(config_path=config_path)

    assert settings.database.embedded_database is True
    assert settings.database.engine is None
    assert settings.database.host is None

###############################################################################
def test_absent_json_database_block_uses_environment_database_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "configurations.json"
    _write_json(
        config_path,
        {
            "datasets": {},
            "tokenizers": {},
            "benchmarks": {},
            "jobs": {},
        },
    )

    env_path = tmp_path / ".env"
    _write_env(
        env_path,
        [
            "DATABASE_EMBEDDED=false",
            "DATABASE_ENGINE=postgresql+psycopg",
            "DATABASE_HOST=127.0.0.1",
            "DATABASE_PORT=5432",
            "DATABASE_NAME=tkben_test",
            "DATABASE_USERNAME=postgres",
        ],
    )
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)

    settings = get_server_settings(config_path=config_path)

    assert settings.database.embedded_database is False
    assert settings.database.engine == "postgresql+psycopg"
    assert settings.database.database_name == "tkben_test"

###############################################################################
def test_structured_database_block_overrides_environment_database_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "configurations.json"
    _write_json(
        config_path,
        {
            "database": {"embedded_database": True},
            "datasets": {},
            "tokenizers": {},
            "benchmarks": {},
            "jobs": {},
        },
    )

    env_path = tmp_path / ".env"
    _write_env(
        env_path,
        [
            "DATABASE_EMBEDDED=false",
            "DATABASE_ENGINE=postgresql+psycopg",
            "DATABASE_HOST=127.0.0.1",
            "DATABASE_NAME=tkben_test",
            "DATABASE_USERNAME=postgres",
        ],
    )
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)

    settings = get_server_settings(config_path=config_path)

    assert settings.database.embedded_database is True
    assert settings.database.engine is None
    assert settings.database.database_name is None

###############################################################################
def test_external_database_requires_host_name_and_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "configurations.json"
    _write_json(
        config_path,
        {
            "database": {"embedded_database": False, "engine": "postgresql+psycopg"},
            "datasets": {},
            "tokenizers": {},
            "benchmarks": {},
            "jobs": {},
        },
    )

    env_path = tmp_path / ".env"
    _write_env(env_path, ["FASTAPI_HOST=127.0.0.1"])
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)

    with pytest.raises(
        RuntimeError,
        match="database.host, database.database_name, database.username",
    ):
        _ = get_server_settings(config_path=config_path)

###############################################################################
def test_get_server_settings_path_scoped_loading_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "configurations.json"
    _write_json(
        config_path,
        {
            "database": {
                "embedded_database": False,
                "engine": "postgresql+psycopg",
                "host": "127.0.0.1",
                "port": 5432,
                "database_name": "app",
                "username": "postgres",
                "password": "secret",
                "ssl": False,
                "ssl_ca": None,
                "connect_timeout": 30,
                "insert_batch_size": 1000,
            },
            "datasets": {"histogram_bins": 30},
            "tokenizers": {"default_scan_limit": 150},
            "benchmarks": {"streaming_batch_size": 2000},
            "jobs": {"polling_interval": 2.5},
        },
    )

    env_path = tmp_path / ".env"
    _write_env(env_path, ["FASTAPI_HOST=0.0.0.0"])
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)

    settings_a = get_server_settings(config_path=config_path)
    settings_b = get_server_settings(config_path=config_path)

    assert settings_a == settings_b
    assert settings_a.database.embedded_database is False
    assert settings_a.database.engine == "postgresql+psycopg"
    assert settings_a.datasets.histogram_bins == 30
    assert settings_a.tokenizers.default_scan_limit == 150
    assert settings_a.benchmarks.streaming_batch_size == 2000
    assert settings_a.jobs.polling_interval == 2.5

###############################################################################
def test_configuration_manager_reload_reflects_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "configurations.json"
    _write_json(
        config_path,
        {
            **_minimal_config_json(),
            "datasets": {"histogram_bins": 20},
        },
    )

    env_path = tmp_path / ".env"
    _write_env(env_path, ["FASTAPI_HOST=127.0.0.1"])
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)

    manager = get_configuration_manager(config_path=config_path)
    assert manager.server_settings.datasets.histogram_bins == 20

    _write_json(
        config_path,
        {
            **_minimal_config_json(),
            "datasets": {"histogram_bins": 45},
        },
    )

    manager.reload()
    assert manager.server_settings.datasets.histogram_bins == 45
    assert manager.get_value("datasets", "histogram_bins") == 45

###############################################################################
def test_configuration_payload_omits_fitting_block(tmp_path: Path) -> None:
    config_path = tmp_path / "configurations.json"
    _write_json(config_path, _minimal_config_json())

    manager = get_configuration_manager(config_path=config_path)

    assert manager.get_block("fitting") == {}
    assert not hasattr(manager.server_settings, "fitting")

###############################################################################
@pytest.mark.parametrize(
    "engine",
    ["postgres", "postgresql", "postgresql+psycopg2"],
)
def test_external_database_rejects_legacy_engine_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
) -> None:
    config_path = tmp_path / "configurations.json"
    _write_json(
        config_path,
        {
            "database": {
                "embedded_database": False,
                "engine": engine,
                "host": "127.0.0.1",
                "port": 5432,
                "database_name": "app",
                "username": "postgres",
                "password": "secret",
            },
            "datasets": {},
            "tokenizers": {},
            "benchmarks": {},
            "jobs": {},
        },
    )
    env_path = tmp_path / ".env"
    _write_env(env_path, ["FASTAPI_HOST=127.0.0.1"])
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)

    settings = get_server_settings(config_path=config_path)
    with pytest.raises(ValueError, match="Unsupported database engine"):
        _resolve_postgres_engine(settings.database.engine)

###############################################################################
def test_allow_key_reveal_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_KEY_REVEAL", "true")
    assert is_key_reveal_enabled() is True

    monkeypatch.delenv("ALLOW_KEY_REVEAL", raising=False)
    assert is_key_reveal_enabled() is False

###############################################################################
def test_hf_key_cipher_seeds_external_material_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material_path = tmp_path / "hf-key-material.json"
    monkeypatch.setenv("HF_KEYS_ENCRYPTION_MATERIAL_FILE", str(material_path))

    cipher = get_hf_key_cipher()
    encrypted = cipher.encrypt("hf_test")
    assert cipher.decrypt(encrypted) == "hf_test"
    assert material_path.is_file()

    second_cipher = get_hf_key_cipher()
    assert second_cipher.decrypt(encrypted) == "hf_test"
