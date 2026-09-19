from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from server.common.path import ROOT_DIR
from server.common.utils.encryption import get_hf_key_cipher
from server.configurations import environment as bootstrap
from server.configurations.settings import JobsSettings, TokenizerSettings
from server.configurations.startup import (
    get_server_settings,
    is_key_reveal_enabled,
    reset_settings_cache_for_tests,
)

###############################################################################
@pytest.fixture(autouse=True)
def reset_configuration_state() -> None:
    reset_settings_cache_for_tests()
    bootstrap.reset_environment_bootstrap_for_tests()
    yield
    reset_settings_cache_for_tests()
    bootstrap.reset_environment_bootstrap_for_tests()

###############################################################################
def _write_env(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

###############################################################################
def _configure_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_lines: list[str],
) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path, env_lines)
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)
    reset_settings_cache_for_tests()
    bootstrap.reset_environment_bootstrap_for_tests()

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
def test_environment_template_exposes_canonical_runtime_inputs() -> None:
    example = (ROOT_DIR / "settings/.env.example").read_text(encoding="utf-8")

    assert "TKBEN_DATA_DIR=app/resources" in example
    assert "TKBEN_LOG_DIR=app/resources/logs" in example
    assert "UI_HOST=127.0.0.1" in example
    assert "DATABASE_EMBEDDED=true" in example
    assert "ALLOW_KEY_REVEAL=false" in example

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
def test_typed_defaults_resolve_without_structured_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "runtime-data"
    _configure_environment(
        tmp_path,
        monkeypatch,
        ["DATABASE_EMBEDDED=true", f"TKBEN_DATA_DIR={data_root}"],
    )

    settings = get_server_settings()

    assert settings.tokenizers.default_discovery_limit == 50
    assert settings.tokenizers.max_discovery_limit == 250
    assert settings.tokenizers.max_discovery_candidates == 750
    assert settings.tokenizers.metadata_candidate_multiplier == 3
    assert settings.tokenizers.max_upload_bytes == 10 * 1024 * 1024
    assert settings.datasets.histogram_bins == 20
    assert settings.datasets.streaming_batch_size == 10_000
    assert settings.datasets.max_upload_bytes == 25 * 1024 * 1024
    assert settings.datasets.download_timeout_seconds == 180.0
    assert settings.datasets.download_retry_attempts == 3
    assert settings.datasets.download_retry_backoff_seconds == 2.0
    assert settings.benchmarks.streaming_batch_size == 1000
    assert settings.jobs.polling_interval == 1.0
    assert not (data_root / "runtime-settings.json").exists()

###############################################################################
def test_runtime_environment_resolves_into_one_settings_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "runtime-data"
    log_root = tmp_path / "runtime-logs"
    material_path = tmp_path / "secrets" / "hf-material.json"
    _configure_environment(
        tmp_path,
        monkeypatch,
        [
            f"TKBEN_DATA_DIR={data_root}",
            f"TKBEN_LOG_DIR={log_root}",
            "FASTAPI_HOST=0.0.0.0",
            "FASTAPI_PORT=5100",
            "UI_HOST=localhost",
            "UI_PORT=8100",
            "VITE_API_BASE_URL=/api",
            "DATABASE_EMBEDDED=true",
            "DATABASE_CONNECT_TIMEOUT=17",
            "DATABASE_INSERT_BATCH_SIZE=321",
            "ALLOW_KEY_REVEAL=true",
            f"HF_KEYS_ENCRYPTION_MATERIAL_FILE={material_path}",
        ],
    )

    settings = get_server_settings()

    assert settings.paths.resources == data_root.resolve()
    assert settings.paths.datasets == (data_root / "sources/datasets").resolve()
    assert settings.paths.tokenizers == (data_root / "sources/tokenizers").resolve()
    assert settings.paths.logs == log_root.resolve()
    assert settings.database.sqlite_path == (data_root / "database.db").resolve()
    assert settings.database.connect_timeout == 17
    assert settings.database.insert_batch_size == 321
    assert settings.network.fastapi_host == "0.0.0.0"
    assert settings.network.fastapi_port == 5100
    assert settings.network.ui_host == "localhost"
    assert settings.network.ui_port == 8100
    assert settings.security.allow_key_reveal is True
    assert settings.security.hf_keys_encryption_material_file == material_path.resolve()

###############################################################################
def test_relative_runtime_paths_are_repository_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_environment(
        tmp_path,
        monkeypatch,
        [
            "TKBEN_DATA_DIR=custom/runtime-data",
            "TKBEN_LOG_DIR=custom/runtime-logs",
            "DATABASE_EMBEDDED=true",
        ],
    )

    settings = get_server_settings()

    assert settings.paths.resources == (ROOT_DIR / "custom/runtime-data").resolve()
    assert settings.paths.logs == (ROOT_DIR / "custom/runtime-logs").resolve()

###############################################################################
def test_external_database_settings_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_environment(
        tmp_path,
        monkeypatch,
        [
            "DATABASE_EMBEDDED=false",
            "DATABASE_HOST=remote-db",
            "DATABASE_PORT=5544",
            "DATABASE_NAME=remote_db",
            "DATABASE_USERNAME=remote_user",
            "DATABASE_PASSWORD=secret",
            "DATABASE_SSL=true",
        ],
    )

    settings = get_server_settings()

    assert settings.database.embedded_database is False
    assert settings.database.host == "remote-db"
    assert settings.database.port == 5544
    assert settings.database.database_name == "remote_db"
    assert settings.database.username == "remote_user"
    assert settings.database.ssl is True

###############################################################################
def test_external_database_requires_host_name_and_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_environment(
        tmp_path,
        monkeypatch,
        [
            "DATABASE_EMBEDDED=false",
            "DATABASE_HOST=",
            "DATABASE_NAME=",
            "DATABASE_USERNAME=",
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="database.host, database.database_name, database.username",
    ):
        get_server_settings()

###############################################################################
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DATABASE_EMBEDDED", "yes"),
        ("DATABASE_SSL", "1"),
        ("ALLOW_KEY_REVEAL", "on"),
    ],
)
def test_boolean_aliases_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    lines = ["DATABASE_EMBEDDED=true", f"{name}={value}"]
    if name == "DATABASE_SSL":
        lines = [
            "DATABASE_EMBEDDED=false",
            "DATABASE_HOST=host",
            "DATABASE_NAME=db",
            "DATABASE_USERNAME=user",
            f"{name}={value}",
        ]
    _configure_environment(tmp_path, monkeypatch, lines)

    with pytest.raises(RuntimeError, match=name):
        get_server_settings()

###############################################################################
def test_settings_models_are_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_environment(tmp_path, monkeypatch, ["DATABASE_EMBEDDED=true"])
    settings = get_server_settings()

    with pytest.raises(ValidationError):
        settings.datasets.histogram_bins = 99

###############################################################################
def test_runtime_setting_models_keep_cross_field_and_polling_validation() -> None:
    with pytest.raises(ValidationError):
        TokenizerSettings(default_discovery_limit=251)
    with pytest.raises(ValidationError):
        TokenizerSettings(default_discovery_limit=20, max_discovery_limit=10)
    with pytest.raises(ValidationError):
        TokenizerSettings(max_discovery_limit=300, max_discovery_candidates=300)
    with pytest.raises(ValidationError):
        JobsSettings(polling_interval=0.1)

###############################################################################
def test_key_reveal_policy_uses_canonical_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_environment(
        tmp_path,
        monkeypatch,
        ["DATABASE_EMBEDDED=true", "ALLOW_KEY_REVEAL=true"],
    )

    assert is_key_reveal_enabled() is True

###############################################################################
def test_hf_key_cipher_uses_configured_material_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material_path = tmp_path / "hf-key-material.json"
    _configure_environment(
        tmp_path,
        monkeypatch,
        [
            "DATABASE_EMBEDDED=true",
            f"HF_KEYS_ENCRYPTION_MATERIAL_FILE={material_path}",
        ],
    )

    cipher = get_hf_key_cipher()
    encrypted = cipher.encrypt("hf_test")

    assert cipher.decrypt(encrypted) == "hf_test"
    assert material_path.is_file()
