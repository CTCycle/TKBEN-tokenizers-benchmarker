from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from server.common.path import APP_DIR, ROOT_DIR


class _FrozenSettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


###############################################################################
class DatabaseSettings(_FrozenSettingsModel):
    embedded_database: bool
    sqlite_path: Path
    host: str | None
    port: int | None
    database_name: str | None
    username: str | None
    password: str | None
    ssl: bool
    ssl_ca: str | None
    connect_timeout: int
    insert_batch_size: int


###############################################################################
class PathSettings(_FrozenSettingsModel):
    resources: Path
    sources: Path
    datasets: Path
    tokenizers: Path
    logs: Path
    templates: Path


###############################################################################
class NetworkSettings(_FrozenSettingsModel):
    fastapi_host: str
    fastapi_port: int = Field(ge=1, le=65535)
    ui_host: str
    ui_port: int = Field(ge=1, le=65535)
    api_base_url: str


###############################################################################
class SecuritySettings(_FrozenSettingsModel):
    allow_key_reveal: bool
    hf_keys_encryption_material_file: Path


###############################################################################
class DatasetSettings(_FrozenSettingsModel):
    allowed_extensions: tuple[str, ...]
    column_detection_cutoff: float = Field(ge=0.0, le=1.0)
    max_upload_bytes: int = Field(ge=1)
    histogram_bins: int = Field(ge=5, le=100)
    streaming_batch_size: int = Field(ge=100)
    log_interval: int = Field(ge=1000)
    cleanup_downloaded_sources: bool
    download_timeout_seconds: float = Field(ge=1.0)
    download_retry_attempts: int = Field(ge=1, le=10)
    download_retry_backoff_seconds: float = Field(ge=0.0, le=60.0)


###############################################################################
class TokenizerSettings(_FrozenSettingsModel):
    default_discovery_limit: int = Field(ge=1, le=250)
    max_discovery_limit: int = Field(ge=1, le=250)
    max_discovery_candidates: int = Field(ge=1)
    metadata_candidate_multiplier: int = Field(ge=1, le=10)
    max_upload_bytes: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_discovery_limits(self) -> "TokenizerSettings":
        if self.default_discovery_limit > self.max_discovery_limit:
            raise ValueError(
                "tokenizers.default_discovery_limit must be <= tokenizers.max_discovery_limit"
            )
        if self.max_discovery_candidates < self.max_discovery_limit:
            raise ValueError(
                "tokenizers.max_discovery_candidates must be >= tokenizers.max_discovery_limit"
            )
        return self


###############################################################################
class BenchmarkSettings(_FrozenSettingsModel):
    streaming_batch_size: int = Field(ge=100)
    log_interval: int = Field(ge=100)


###############################################################################
class JobsSettings(_FrozenSettingsModel):
    polling_interval: float = Field(gt=0.0)
    terminal_retention_seconds: float = Field(ge=0.0)


###############################################################################
class ApplicationConfiguration(_FrozenSettingsModel):
    datasets: DatasetSettings
    tokenizers: TokenizerSettings
    benchmarks: BenchmarkSettings
    jobs: JobsSettings

    @classmethod
    def from_path(cls, path: str | Path) -> "ApplicationConfiguration":
        configuration_path = Path(path)
        if not configuration_path.exists():
            raise RuntimeError(f"Configuration file not found: {configuration_path}")
        try:
            payload = json.loads(configuration_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unable to load configuration from {configuration_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Configuration must be a JSON object.")
        return cls.model_validate(payload)


###############################################################################
class ServerSettings(_FrozenSettingsModel):
    database: DatabaseSettings
    paths: PathSettings
    network: NetworkSettings
    security: SecuritySettings
    datasets: DatasetSettings
    tokenizers: TokenizerSettings
    benchmarks: BenchmarkSettings
    jobs: JobsSettings


###############################################################################
def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


###############################################################################
def _read_env_text(name: str, default: str) -> str:
    value = _normalize_optional_text(os.getenv(name))
    return value if value is not None else default


###############################################################################
def _read_env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    normalized = raw_value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(
        f"{name} must be either 'true' or 'false', got: {raw_value}"
    )


###############################################################################
def _read_env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        value = default
    else:
        try:
            value = int(raw_value.strip())
        except ValueError as exc:
            raise RuntimeError(
                f"{name} must be a valid integer, got: {raw_value}"
            ) from exc

    if minimum is not None and value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got: {value}")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"{name} must be <= {maximum}, got: {value}")
    return value


###############################################################################
def _resolve_runtime_path(configured_path: str | None, default_path: Path) -> Path:
    configured = _normalize_optional_text(configured_path)
    candidate = Path(configured).expanduser() if configured is not None else default_path
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate.resolve()


###############################################################################
def _load_path_settings() -> PathSettings:
    resources = _resolve_runtime_path(os.getenv("TKBEN_DATA_DIR"), APP_DIR / "resources")
    sources = resources / "sources"
    return PathSettings(
        resources=resources,
        sources=sources,
        datasets=sources / "datasets",
        tokenizers=sources / "tokenizers",
        logs=_resolve_runtime_path(os.getenv("TKBEN_LOG_DIR"), resources / "logs"),
        templates=resources / "templates",
    )


###############################################################################
def _load_database_settings(paths: PathSettings) -> DatabaseSettings:
    embedded_database = _read_env_bool("DATABASE_EMBEDDED", True)
    connect_timeout = _read_env_int("DATABASE_CONNECT_TIMEOUT", 30, minimum=1)
    insert_batch_size = _read_env_int("DATABASE_INSERT_BATCH_SIZE", 1000, minimum=1)
    sqlite_path = paths.resources / "database.db"

    if embedded_database:
        return DatabaseSettings(
            embedded_database=True,
            sqlite_path=sqlite_path,
            host=None,
            port=None,
            database_name=None,
            username=None,
            password=None,
            ssl=False,
            ssl_ca=None,
            connect_timeout=connect_timeout,
            insert_batch_size=insert_batch_size,
        )

    host = _normalize_optional_text(os.getenv("DATABASE_HOST"))
    username = _normalize_optional_text(os.getenv("DATABASE_USERNAME"))
    password = _normalize_optional_text(os.getenv("DATABASE_PASSWORD"))
    database_name = _normalize_optional_text(os.getenv("DATABASE_NAME"))
    port = _read_env_int("DATABASE_PORT", 5432, minimum=1, maximum=65535)
    ssl = _read_env_bool("DATABASE_SSL", False)
    ssl_ca = _normalize_optional_text(os.getenv("DATABASE_SSL_CA"))

    missing: list[str] = []
    if not host:
        missing.append("database.host")
    if not database_name:
        missing.append("database.database_name")
    if not username:
        missing.append("database.username")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"External database configuration requires: {joined}")

    return DatabaseSettings(
        embedded_database=False,
        sqlite_path=sqlite_path,
        host=host,
        port=port,
        database_name=database_name,
        username=username,
        password=password,
        ssl=ssl,
        ssl_ca=ssl_ca,
        connect_timeout=connect_timeout,
        insert_batch_size=insert_batch_size,
    )


###############################################################################
def _load_network_settings() -> NetworkSettings:
    return NetworkSettings(
        fastapi_host=_read_env_text("FASTAPI_HOST", "127.0.0.1"),
        fastapi_port=_read_env_int("FASTAPI_PORT", 5000, minimum=1, maximum=65535),
        ui_host=_read_env_text("UI_HOST", "127.0.0.1"),
        ui_port=_read_env_int("UI_PORT", 8000, minimum=1, maximum=65535),
        api_base_url=_read_env_text("VITE_API_BASE_URL", "/api"),
    )


###############################################################################
def _load_security_settings(paths: PathSettings) -> SecuritySettings:
    return SecuritySettings(
        allow_key_reveal=_read_env_bool("ALLOW_KEY_REVEAL", False),
        hf_keys_encryption_material_file=_resolve_runtime_path(
            os.getenv("HF_KEYS_ENCRYPTION_MATERIAL_FILE"),
            paths.resources / "hf-key-material.json",
        ),
    )


###############################################################################
def build_server_settings(configuration: ApplicationConfiguration) -> ServerSettings:
    paths = _load_path_settings()
    return ServerSettings(
        database=_load_database_settings(paths),
        paths=paths,
        network=_load_network_settings(),
        security=_load_security_settings(paths),
        datasets=configuration.datasets,
        tokenizers=configuration.tokenizers,
        benchmarks=configuration.benchmarks,
        jobs=configuration.jobs,
    )
