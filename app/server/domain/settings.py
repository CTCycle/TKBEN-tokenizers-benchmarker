from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


###############################################################################
@dataclass(frozen=True)
class DatabaseSettings:
    embedded_database: bool
    engine: str | None
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


@dataclass(frozen=True)
class TokenizerSettings:
    default_scan_limit: int
    max_scan_limit: int
    min_scan_limit: int
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


###############################################################################
@dataclass(frozen=True)
class ServerSettings:
    database: DatabaseSettings
    datasets: DatasetSettings
    tokenizers: TokenizerSettings
    benchmarks: BenchmarkSettings
    jobs: JobsSettings


###############################################################################
class JsonDatabaseSettings(BaseModel):
    embedded_database: bool = True
    engine: str = "postgresql+psycopg"
    host: str | None = None
    port: int = Field(default=5432, ge=1, le=65535)
    database_name: str | None = None
    username: str | None = None
    password: str | None = None
    ssl: bool = False
    ssl_ca: str | None = None
    connect_timeout: int = Field(default=10, ge=1)
    insert_batch_size: int = Field(default=1000, ge=1)

    @field_validator(
        "engine",
        "host",
        "database_name",
        "username",
        "password",
        "ssl_ca",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        return text

    @model_validator(mode="after")
    def validate_external_database_requirements(self) -> "JsonDatabaseSettings":
        if self.embedded_database:
            return self

        missing: list[str] = []
        if not self.host:
            missing.append("database.host")
        if not self.database_name:
            missing.append("database.database_name")
        if not self.username:
            missing.append("database.username")
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"External database mode requires configuration keys: {joined}")
        return self


###############################################################################
class JsonDatasetSettings(BaseModel):
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
    default_scan_limit: int = Field(default=100, ge=1)
    max_scan_limit: int = Field(default=1000, ge=1)
    min_scan_limit: int = Field(default=1, ge=1)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)

    @model_validator(mode="after")
    def validate_scan_limits(self) -> "JsonTokenizerSettings":
        if self.max_scan_limit < self.min_scan_limit:
            raise ValueError("tokenizers.max_scan_limit must be >= tokenizers.min_scan_limit")
        if self.default_scan_limit < self.min_scan_limit:
            raise ValueError(
                "tokenizers.default_scan_limit must be >= tokenizers.min_scan_limit"
            )
        if self.default_scan_limit > self.max_scan_limit:
            raise ValueError(
                "tokenizers.default_scan_limit must be <= tokenizers.max_scan_limit"
            )
        return self


###############################################################################
class JsonBenchmarkSettings(BaseModel):
    streaming_batch_size: int = Field(default=1000, ge=100)
    log_interval: int = Field(default=10000, ge=100)


###############################################################################
class JsonJobsSettings(BaseModel):
    polling_interval: float = Field(default=1.0, gt=0.0)


###############################################################################
class JsonConfiguration(BaseModel):
    database: JsonDatabaseSettings = Field(default_factory=JsonDatabaseSettings)
    datasets: JsonDatasetSettings = Field(default_factory=JsonDatasetSettings)
    tokenizers: JsonTokenizerSettings = Field(default_factory=JsonTokenizerSettings)
    benchmarks: JsonBenchmarkSettings = Field(default_factory=JsonBenchmarkSettings)
    jobs: JsonJobsSettings = Field(default_factory=JsonJobsSettings)

    # -------------------------------------------------------------------------
    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "JsonConfiguration":
        return cls(
            database=payload.get("database", {}),
            datasets=payload.get("datasets", {}),
            tokenizers=payload.get("tokenizers", {}),
            benchmarks=payload.get("benchmarks", {}),
            jobs=payload.get("jobs", {}),
        )

    # -------------------------------------------------------------------------
    @classmethod
    def from_path(cls, path: str | Path) -> "JsonConfiguration":
        configuration_path = Path(path)
        if not configuration_path.exists():
            raise RuntimeError(f"Configuration file not found: {configuration_path}")
        try:
            payload = json.loads(configuration_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to load configuration from {configuration_path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Configuration must be a JSON object.")
        return cls.from_payload(payload)

    # -------------------------------------------------------------------------
    def to_server_settings(self) -> ServerSettings:
        db = self.database
        if db.embedded_database:
            database_settings = DatabaseSettings(
                embedded_database=True,
                engine=None,
                host=None,
                port=None,
                database_name=None,
                username=None,
                password=None,
                ssl=False,
                ssl_ca=None,
                connect_timeout=db.connect_timeout,
                insert_batch_size=db.insert_batch_size,
            )
        else:
            database_settings = DatabaseSettings(
                embedded_database=False,
                engine=db.engine.lower() if db.engine else "postgresql+psycopg",
                host=db.host,
                port=db.port,
                database_name=db.database_name,
                username=db.username,
                password=db.password,
                ssl=db.ssl,
                ssl_ca=db.ssl_ca,
                connect_timeout=db.connect_timeout,
                insert_batch_size=db.insert_batch_size,
            )

        return ServerSettings(
            database=database_settings,
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
                default_scan_limit=self.tokenizers.default_scan_limit,
                max_scan_limit=self.tokenizers.max_scan_limit,
                min_scan_limit=self.tokenizers.min_scan_limit,
                max_upload_bytes=self.tokenizers.max_upload_bytes,
            ),
            benchmarks=BenchmarkSettings(
                streaming_batch_size=self.benchmarks.streaming_batch_size,
                log_interval=self.benchmarks.log_interval,
            ),
            jobs=JobsSettings(
                polling_interval=self.jobs.polling_interval,
            ),
        )
