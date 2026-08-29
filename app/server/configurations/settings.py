from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

###############################################################################
@dataclass(frozen=True)
class DatabaseSettings:
    embedded_database: bool
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
@dataclass(frozen=True)
class DatasetSettings:
    allowed_extensions: tuple[str, ...]
    column_detection_cutoff: float
    max_upload_bytes: int
    histogram_bins: int
    streaming_batch_size: int
    log_interval: int
    cleanup_downloaded_sources: bool
    download_timeout_seconds: float
    download_retry_attempts: int
    download_retry_backoff_seconds: float

###############################################################################
@dataclass(frozen=True)
class TokenizerSettings:
    default_discovery_limit: int
    max_discovery_limit: int
    max_discovery_candidates: int
    metadata_candidate_multiplier: int
    max_upload_bytes: int

###############################################################################
@dataclass(frozen=True)
class BenchmarkSettings:
    streaming_batch_size: int
    log_interval: int

###############################################################################
@dataclass(frozen=True)
class JobsSettings:
    polling_interval: float
    terminal_retention_seconds: float

###############################################################################
@dataclass(frozen=True)
class ServerSettings:
    database: DatabaseSettings
    datasets: DatasetSettings
    tokenizers: TokenizerSettings
    benchmarks: BenchmarkSettings
    jobs: JobsSettings

###############################################################################
def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return text

###############################################################################
def _read_env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean value, got: {raw_value}")

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

def _load_database_settings_from_sources() -> DatabaseSettings:
    """Load database settings from the environment only.

    Keeping this source single-purpose prevents the JSON application
    configuration from silently selecting a different database than the
    launcher and migration tooling configured in ``settings/.env``.
    """
    embedded_database = _read_env_bool("DATABASE_EMBEDDED", True)
    connect_timeout = _read_env_int("DATABASE_CONNECT_TIMEOUT", 30, minimum=1)
    insert_batch_size = _read_env_int("DATABASE_INSERT_BATCH_SIZE", 1000, minimum=1)

    if embedded_database:
        return DatabaseSettings(
            embedded_database=True,
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
class JsonDatasetSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_extensions: tuple[str, ...] = (".csv", ".xls", ".xlsx")
    column_detection_cutoff: float = Field(default=0.6, ge=0.0, le=1.0)
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
    histogram_bins: int = Field(default=20, ge=5, le=100)
    streaming_batch_size: int = Field(default=10000, ge=100)
    log_interval: int = Field(default=100000, ge=1000)
    cleanup_downloaded_sources: bool = False
    download_timeout_seconds: float = Field(default=120.0, ge=1.0)
    download_retry_attempts: int = Field(default=3, ge=1, le=10)
    download_retry_backoff_seconds: float = Field(default=1.0, ge=0.0, le=60.0)

###############################################################################
class JsonTokenizerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_discovery_limit: int = Field(default=50, ge=1, le=250)
    max_discovery_limit: int = Field(default=250, ge=1, le=250)
    max_discovery_candidates: int = Field(default=750, ge=1)
    metadata_candidate_multiplier: int = Field(default=3, ge=1, le=10)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_discovery_limits(self) -> "JsonTokenizerSettings":
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
class JsonBenchmarkSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    streaming_batch_size: int = Field(default=1000, ge=100)
    log_interval: int = Field(default=10000, ge=100)

###############################################################################
class JsonJobsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    polling_interval: float = Field(default=1.0, gt=0.0)
    terminal_retention_seconds: float = Field(default=3600.0, ge=0.0)

###############################################################################
class JsonConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: JsonDatasetSettings = Field(default_factory=JsonDatasetSettings)
    tokenizers: JsonTokenizerSettings = Field(default_factory=JsonTokenizerSettings)
    benchmarks: JsonBenchmarkSettings = Field(default_factory=JsonBenchmarkSettings)
    jobs: JsonJobsSettings = Field(default_factory=JsonJobsSettings)

    # -------------------------------------------------------------------------
    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "JsonConfiguration":
        return cls.model_validate(payload)

    # -------------------------------------------------------------------------
    @classmethod
    def from_path(cls, path: str | Path) -> "JsonConfiguration":
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
        return cls.from_payload(payload)

    # -------------------------------------------------------------------------
    def to_server_settings(self) -> ServerSettings:
        return ServerSettings(
            database=_load_database_settings_from_sources(),
            datasets=DatasetSettings(
                allowed_extensions=tuple(self.datasets.allowed_extensions),
                column_detection_cutoff=self.datasets.column_detection_cutoff,
                max_upload_bytes=self.datasets.max_upload_bytes,
                histogram_bins=self.datasets.histogram_bins,
                streaming_batch_size=self.datasets.streaming_batch_size,
                log_interval=self.datasets.log_interval,
                cleanup_downloaded_sources=self.datasets.cleanup_downloaded_sources,
                download_timeout_seconds=self.datasets.download_timeout_seconds,
                download_retry_attempts=self.datasets.download_retry_attempts,
                download_retry_backoff_seconds=self.datasets.download_retry_backoff_seconds,
            ),
            tokenizers=TokenizerSettings(
                default_discovery_limit=self.tokenizers.default_discovery_limit,
                max_discovery_limit=self.tokenizers.max_discovery_limit,
                max_discovery_candidates=self.tokenizers.max_discovery_candidates,
                metadata_candidate_multiplier=self.tokenizers.metadata_candidate_multiplier,
                max_upload_bytes=self.tokenizers.max_upload_bytes,
            ),
            benchmarks=BenchmarkSettings(
                streaming_batch_size=self.benchmarks.streaming_batch_size,
                log_interval=self.benchmarks.log_interval,
            ),
            jobs=JobsSettings(
                polling_interval=self.jobs.polling_interval,
                terminal_retention_seconds=self.jobs.terminal_retention_seconds,
            ),
        )
