# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from server.common.utils.encryption import get_purpose_cipher
from server.configurations.settings import DatabaseSettings

DATABASE_CONFIGURATION_FILENAME = "database-config.json"
DATABASE_CONFIGURATION_SCHEMA_VERSION = 1
DATABASE_PASSWORD_PURPOSE = "database_connection_password"
_DATABASE_ENV_LINE = re.compile(r"^\s*(?:export\s+)?DATABASE_[A-Z0-9_]+\s*=")


class StoredDatabaseConfiguration(BaseModel):
    """Validated, database-independent connection configuration."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: StrictInt = Field(ge=1)
    backend: Literal["sqlite", "postgresql"]
    sqlite_path: StrictStr | None = None
    host: StrictStr | None = None
    port: StrictInt | None = Field(default=None, ge=1, le=65535)
    database_name: StrictStr | None = None
    username: StrictStr | None = None
    encrypted_password: StrictStr | None = None
    ssl: bool = False
    ssl_ca: StrictStr | None = None
    connect_timeout: StrictInt = Field(default=30, ge=1)
    insert_batch_size: StrictInt = Field(default=1000, ge=1)

    @model_validator(mode="after")
    def validate_backend_fields(self) -> "StoredDatabaseConfiguration":
        if self.schema_version != DATABASE_CONFIGURATION_SCHEMA_VERSION:
            raise ValueError("Unsupported database configuration schema version.")
        if self.backend == "sqlite":
            if not self.sqlite_path or not self.sqlite_path.strip():
                raise ValueError("SQLite database path is required.")
            if any(
                (
                    self.host,
                    self.port,
                    self.database_name,
                    self.username,
                    self.encrypted_password,
                    self.ssl,
                    self.ssl_ca,
                )
            ):
                raise ValueError("PostgreSQL fields are not allowed for SQLite.")
        else:
            missing = [
                name
                for name, value in (
                    ("host", self.host),
                    ("database_name", self.database_name),
                    ("username", self.username),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "PostgreSQL configuration requires: " + ", ".join(missing)
                )
            if self.sqlite_path is not None:
                raise ValueError("SQLite path is not allowed for PostgreSQL.")
        return self


class DatabaseConfigurationStore:
    """Own the single JSON source of truth for database connection settings."""

    def __init__(self, data_directory: Path, material_file: Path, env_file: Path) -> None:
        self.path = data_directory / DATABASE_CONFIGURATION_FILENAME
        self._data_directory = data_directory
        self._material_file = material_file
        self._env_file = env_file
        self._active_configuration: StoredDatabaseConfiguration | None = None

    def load(self) -> StoredDatabaseConfiguration:
        if self._active_configuration is not None:
            return self._active_configuration
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                configuration = StoredDatabaseConfiguration.model_validate(payload)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"Database configuration is invalid: {self.path}. "
                    "Restore a valid configuration file before starting the server."
                ) from exc
            self._strip_legacy_environment()
            self._active_configuration = configuration
            return configuration

        legacy = self._read_legacy_environment()
        configuration = self._from_legacy_environment(legacy)
        self.save(configuration)
        self._strip_legacy_environment()
        return configuration

    def to_database_settings(
        self, configuration: StoredDatabaseConfiguration
    ) -> DatabaseSettings:
        if configuration.backend == "sqlite":
            sqlite_path = Path(configuration.sqlite_path or "database.db").expanduser()
            if not sqlite_path.is_absolute():
                sqlite_path = self._data_directory / sqlite_path
            return DatabaseSettings(
                embedded_database=True,
                sqlite_path=sqlite_path.resolve(),
                host=None,
                port=None,
                database_name=None,
                username=None,
                password=None,
                ssl=False,
                ssl_ca=None,
                connect_timeout=configuration.connect_timeout,
                insert_batch_size=configuration.insert_batch_size,
            )

        password = None
        if configuration.encrypted_password:
            try:
                password = get_purpose_cipher(
                    self._material_file, DATABASE_PASSWORD_PURPOSE
                ).decrypt(configuration.encrypted_password)
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    "Unable to decrypt the configured PostgreSQL password."
                ) from exc
        return DatabaseSettings(
            embedded_database=False,
            sqlite_path=(self._data_directory / "database.db").resolve(),
            host=configuration.host,
            port=configuration.port,
            database_name=configuration.database_name,
            username=configuration.username,
            password=password,
            ssl=configuration.ssl,
            ssl_ca=configuration.ssl_ca,
            connect_timeout=configuration.connect_timeout,
            insert_batch_size=configuration.insert_batch_size,
        )

    def build_candidate(
        self,
        *,
        backend: Literal["sqlite", "postgresql"],
        sqlite_path: str | None = None,
        host: str | None = None,
        port: int | None = None,
        database_name: str | None = None,
        username: str | None = None,
        password: str | None = None,
        clear_password: bool = False,
        ssl: bool = False,
        ssl_ca: str | None = None,
        connect_timeout: int = 30,
        insert_batch_size: int = 1000,
    ) -> StoredDatabaseConfiguration:
        current = self.load()
        encrypted_password = current.encrypted_password
        if backend == "sqlite":
            return StoredDatabaseConfiguration(
                schema_version=DATABASE_CONFIGURATION_SCHEMA_VERSION,
                backend=backend,
                sqlite_path=sqlite_path or "database.db",
                connect_timeout=connect_timeout,
                insert_batch_size=insert_batch_size,
            )

        if clear_password:
            encrypted_password = None
        if password:
            encrypted_password = get_purpose_cipher(
                self._material_file, DATABASE_PASSWORD_PURPOSE
            ).encrypt(password)
        same_target = (
            current.backend == "postgresql"
            and current.host == host
            and current.port == port
            and current.database_name == database_name
            and current.username == username
        )
        if not password and not clear_password and not same_target:
            encrypted_password = None
        return StoredDatabaseConfiguration(
            schema_version=DATABASE_CONFIGURATION_SCHEMA_VERSION,
            backend=backend,
            host=host,
            port=port or 5432,
            database_name=database_name,
            username=username,
            encrypted_password=encrypted_password,
            ssl=ssl,
            ssl_ca=ssl_ca or None,
            connect_timeout=connect_timeout,
            insert_batch_size=insert_batch_size,
        )

    def save(self, configuration: StoredDatabaseConfiguration) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor: int | None = None
        temporary_path: str | None = None
        try:
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
                descriptor = None
                destination.write(
                    json.dumps(
                        configuration.model_dump(mode="json"),
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
            self._active_configuration = configuration
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError("Unable to persist database configuration.") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass

    def _read_legacy_environment(self) -> dict[str, str]:
        values = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("DATABASE_")
        }
        if self._env_file.is_file():
            values.update(
                {
                    key: value
                    for key, value in dotenv_values(self._env_file).items()
                    if key.startswith("DATABASE_") and value is not None
                }
            )
        return values

    def _from_legacy_environment(
        self, values: dict[str, str]
    ) -> StoredDatabaseConfiguration:
        embedded = _legacy_bool(values, "DATABASE_EMBEDDED", True)
        timeout = _legacy_int(values, "DATABASE_CONNECT_TIMEOUT", 30, minimum=1)
        batch_size = _legacy_int(values, "DATABASE_INSERT_BATCH_SIZE", 1000, minimum=1)
        if embedded:
            return StoredDatabaseConfiguration(
                schema_version=DATABASE_CONFIGURATION_SCHEMA_VERSION,
                backend="sqlite",
                sqlite_path="database.db",
                connect_timeout=timeout,
                insert_batch_size=batch_size,
            )

        host = values.get("DATABASE_HOST", "").strip()
        database_name = values.get("DATABASE_NAME", "").strip()
        username = values.get("DATABASE_USERNAME", "").strip()
        missing = [
            name
            for name, value in (
                ("database.host", host),
                ("database.database_name", database_name),
                ("database.username", username),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "External database configuration requires: " + ", ".join(missing)
            )
        password = values.get("DATABASE_PASSWORD") or None
        ssl_ca = values.get("DATABASE_SSL_CA", "").strip() or None
        return StoredDatabaseConfiguration(
            schema_version=DATABASE_CONFIGURATION_SCHEMA_VERSION,
            backend="postgresql",
            host=host,
            port=_legacy_int(
                values, "DATABASE_PORT", 5432, minimum=1, maximum=65535
            ),
            database_name=database_name,
            username=username,
            encrypted_password=(
                get_purpose_cipher(self._material_file, DATABASE_PASSWORD_PURPOSE)
                .encrypt(password)
                if password
                else None
            ),
            ssl=_legacy_bool(values, "DATABASE_SSL", False),
            ssl_ca=ssl_ca,
            connect_timeout=timeout,
            insert_batch_size=batch_size,
        )

    def _strip_legacy_environment(self) -> None:
        if self._env_file.is_file():
            temporary_path: str | None = None
            try:
                lines = self._env_file.read_text(encoding="utf-8").splitlines()
                retained = [line for line in lines if not _DATABASE_ENV_LINE.match(line)]
                if retained != lines:
                    descriptor, temporary_path = tempfile.mkstemp(
                        prefix=f".{self._env_file.name}.",
                        suffix=".tmp",
                        dir=self._env_file.parent,
                    )
                    with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
                        destination.write("\n".join(retained).rstrip() + "\n")
                        destination.flush()
                        os.fsync(destination.fileno())
                    os.replace(temporary_path, self._env_file)
                    temporary_path = None
            except OSError as exc:
                raise RuntimeError(
                    "Database configuration was saved, but obsolete database values "
                    "could not be removed from settings/.env."
                ) from exc
            finally:
                if temporary_path is not None:
                    try:
                        os.unlink(temporary_path)
                    except OSError:
                        pass
        for name in tuple(os.environ):
            if name.startswith("DATABASE_"):
                os.environ.pop(name, None)


def _legacy_bool(values: dict[str, str], key: str, default: bool) -> bool:
    value = values.get(key)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise RuntimeError(f"Legacy {key} must be either 'true' or 'false'.")
    return normalized == "true"


def _legacy_int(
    values: dict[str, str],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    raw_value = values.get(key)
    try:
        value = default if raw_value is None or not raw_value.strip() else int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"Legacy {key} must be a valid integer.") from exc
    if value < minimum or (maximum is not None and value > maximum):
        raise RuntimeError(f"Legacy {key} is outside the supported range.")
    return value
