from __future__ import annotations

import pytest

from TKBEN.server.configurations.server import build_database_settings


###############################################################################
@pytest.mark.parametrize(
    ("env_value", "json_value", "expected_embedded"),
    [
        ("false", True, False),
        ("true", False, True),
    ],
)
def test_build_database_settings_db_embedded_env_overrides_json_default(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    json_value: bool,
    expected_embedded: bool,
) -> None:
    monkeypatch.setenv("DB_EMBEDDED", env_value)

    settings = build_database_settings({"embedded_database": json_value})

    assert settings.embedded_database is expected_embedded


###############################################################################
def test_build_database_settings_uses_external_db_env_fields_when_not_embedded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "embedded_database": True,
        "engine": "postgresql+psycopg2",
        "host": "payload-host",
        "port": 15432,
        "database_name": "payload_db",
        "username": "payload_user",
        "password": "payload_password",
        "ssl": False,
        "ssl_ca": None,
        "connect_timeout": 11,
        "insert_batch_size": 123,
    }

    monkeypatch.setenv("DB_EMBEDDED", "false")
    monkeypatch.setenv("DB_ENGINE", "postgresql+psycopg")
    monkeypatch.setenv("DB_HOST", "env-host")
    monkeypatch.setenv("DB_PORT", "6543")
    monkeypatch.setenv("DB_NAME", "env_db")
    monkeypatch.setenv("DB_USER", "env_user")
    monkeypatch.setenv("DB_PASSWORD", "env_password")
    monkeypatch.setenv("DB_SSL", "true")
    monkeypatch.setenv("DB_SSL_CA", "/tmp/ca.pem")
    monkeypatch.setenv("DB_CONNECT_TIMEOUT", "42")
    monkeypatch.setenv("DB_INSERT_BATCH_SIZE", "777")

    settings = build_database_settings(payload)

    assert settings.embedded_database is False
    assert settings.engine == "postgresql+psycopg"
    assert settings.host == "env-host"
    assert settings.port == 6543
    assert settings.database_name == "env_db"
    assert settings.username == "env_user"
    assert settings.password == "env_password"
    assert settings.ssl is True
    assert settings.ssl_ca == "/tmp/ca.pem"
    assert settings.connect_timeout == 42
    assert settings.insert_batch_size == 777
