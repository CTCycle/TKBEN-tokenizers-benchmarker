from __future__ import annotations

import json

import pytest

from TKBEN.server.configurations.server import (
    build_database_settings,
    get_server_settings,
)


###############################################################################
@pytest.mark.parametrize(
    ("json_value", "expected_embedded"),
    [
        (True, True),
        (False, False),
    ],
)
def test_build_database_settings_uses_json_embedded_switch(
    json_value: bool,
    expected_embedded: bool,
) -> None:
    settings = build_database_settings({"embedded_database": json_value})

    assert settings.embedded_database is expected_embedded


###############################################################################
def test_build_database_settings_uses_external_db_json_fields_when_not_embedded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = build_database_settings(
        {
            "embedded_database": False,
            "engine": "postgresql+psycopg",
            "host": "payload-host",
            "port": 15432,
            "database_name": "payload_db",
            "username": "payload_user",
            "password": "payload_password",
            "ssl": True,
            "ssl_ca": "/tmp/ca.pem",
            "connect_timeout": 42,
            "insert_batch_size": 777,
        }
    )

    monkeypatch.setenv("DB_EMBEDDED", "true")
    monkeypatch.setenv("DB_HOST", "ignored-host")
    monkeypatch.setenv("DB_PORT", "9999")

    assert settings.embedded_database is False
    assert settings.engine == "postgresql+psycopg"
    assert settings.host == "payload-host"
    assert settings.port == 15432
    assert settings.database_name == "payload_db"
    assert settings.username == "payload_user"
    assert settings.password == "payload_password"
    assert settings.ssl is True
    assert settings.ssl_ca == "/tmp/ca.pem"
    assert settings.connect_timeout == 42
    assert settings.insert_batch_size == 777


###############################################################################
def test_get_server_settings_reads_database_block_from_json(tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(
        json.dumps(
            {
                "database": {
                    "embedded_database": False,
                    "engine": "postgresql+psycopg",
                    "host": "127.0.0.1",
                    "port": 5432,
                    "database_name": "tkben",
                    "username": "postgres",
                    "password": "secret",
                    "ssl": False,
                    "ssl_ca": None,
                    "connect_timeout": 30,
                    "insert_batch_size": 1000,
                },
                "datasets": {},
                "tokenizers": {},
                "benchmarks": {},
                "jobs": {},
                "fitting": {},
            }
        ),
        encoding="utf-8",
    )

    settings = get_server_settings(str(config_path))

    assert settings.database.embedded_database is False
    assert settings.database.engine == "postgresql+psycopg"
    assert settings.database.host == "127.0.0.1"
    assert settings.database.port == 5432


###############################################################################
def test_build_database_settings_embedded_mode_ignores_external_fields() -> None:
    settings = build_database_settings(
        {
            "embedded_database": True,
            "engine": "postgresql+psycopg",
            "host": "127.0.0.1",
            "port": 5432,
            "database_name": "tkben",
            "username": "postgres",
            "password": "secret",
            "ssl": True,
            "ssl_ca": "/tmp/ca.pem",
            "connect_timeout": 30,
            "insert_batch_size": 1000,
        }
    )

    assert settings.embedded_database is True
    assert settings.engine is None
    assert settings.host is None
    assert settings.port is None
    assert settings.database_name is None
    assert settings.username is None
    assert settings.password is None
    assert settings.ssl is False
    assert settings.ssl_ca is None


###############################################################################
def test_build_database_settings_defaults_for_external_mode() -> None:
    settings = build_database_settings({"embedded_database": False})

    assert settings.embedded_database is False
    assert settings.engine == "postgres"
    assert settings.port == 5432
    assert settings.ssl is False
    assert settings.connect_timeout == 10
    assert settings.insert_batch_size == 1000


###############################################################################
def test_build_database_settings_custom_external_values() -> None:
    settings = build_database_settings(
        {
            "embedded_database": False,
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
    )

    assert settings.embedded_database is False
    assert settings.engine == "postgresql+psycopg2"
    assert settings.host == "payload-host"
    assert settings.port == 15432
    assert settings.database_name == "payload_db"
    assert settings.username == "payload_user"
    assert settings.password == "payload_password"
    assert settings.ssl is False
    assert settings.ssl_ca is None
    assert settings.connect_timeout == 11
    assert settings.insert_batch_size == 123
