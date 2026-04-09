from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from TKBEN.server.common.constants import CONFIGURATIONS_FILE
from TKBEN.server.configurations.bootstrap import ensure_environment_loaded
from TKBEN.server.domain.settings import (
    BenchmarkSettings,
    DatabaseSettings,
    DatasetSettings,
    FittingSettings,
    JobsSettings,
    ServerSettings,
    TokenizerSettings,
)


###############################################################################
class JsonDatabaseSettings(BaseModel):
    embedded_database: bool = True
    engine: str = "postgres"
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
class JsonFittingSettings(BaseModel):
    default_max_iterations: int = Field(default=1000, ge=1)
    max_iterations_upper_bound: int = Field(default=1_000_000, ge=1)
    save_best_default: bool = True
    parameter_initial_default: float = Field(default=1.0, ge=0.0)
    parameter_min_default: float = Field(default=0.0, ge=0.0)
    parameter_max_default: float = Field(default=100.0, ge=0.0)
    preview_row_limit: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def validate_ranges(self) -> "JsonFittingSettings":
        if self.max_iterations_upper_bound < self.default_max_iterations:
            raise ValueError(
                "fitting.max_iterations_upper_bound must be >= fitting.default_max_iterations"
            )
        if self.parameter_max_default < self.parameter_min_default:
            raise ValueError(
                "fitting.parameter_max_default must be >= fitting.parameter_min_default"
            )
        return self


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
class JsonConfigurationSettingsSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        raw_path = getattr(settings_cls, "_configuration_file", CONFIGURATIONS_FILE)
        self.configuration_file = Path(raw_path)

    # -------------------------------------------------------------------------
    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    # -------------------------------------------------------------------------
    def __call__(self) -> dict[str, Any]:
        if not self.configuration_file.exists():
            raise RuntimeError(f"Configuration file not found: {self.configuration_file}")
        try:
            payload = json.loads(self.configuration_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to load configuration from {self.configuration_file}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Configuration must be a JSON object.")

        return {
            "database": payload.get("database", {}),
            "datasets": payload.get("datasets", {}),
            "fitting": payload.get("fitting", {}),
            "tokenizers": payload.get("tokenizers", {}),
            "benchmarks": payload.get("benchmarks", {}),
            "jobs": payload.get("jobs", {}),
        }


###############################################################################
class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    _configuration_file: ClassVar[str] = CONFIGURATIONS_FILE

    database: JsonDatabaseSettings = Field(default_factory=JsonDatabaseSettings)
    datasets: JsonDatasetSettings = Field(default_factory=JsonDatasetSettings)
    fitting: JsonFittingSettings = Field(default_factory=JsonFittingSettings)
    tokenizers: JsonTokenizerSettings = Field(default_factory=JsonTokenizerSettings)
    benchmarks: JsonBenchmarkSettings = Field(default_factory=JsonBenchmarkSettings)
    jobs: JsonJobsSettings = Field(default_factory=JsonJobsSettings)

    fastapi_host: str = "127.0.0.1"
    fastapi_port: int = Field(default=5000, ge=1, le=65535)
    ui_host: str = "127.0.0.1"
    ui_port: int = Field(default=8000, ge=1, le=65535)
    vite_api_base_url: str = "/api"
    reload: bool = False
    optional_dependencies: bool = False
    keras_backend: str | None = None
    mplbackend: str | None = None
    allow_key_reveal: bool = False
    hf_keys_encryption_key: str | None = None
    tkben_tauri_mode: bool = False

    @field_validator(
        "fastapi_host",
        "ui_host",
        "vite_api_base_url",
        "keras_backend",
        "mplbackend",
        "hf_keys_encryption_key",
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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        _ = dotenv_settings
        return (
            init_settings,
            env_settings,
            JsonConfigurationSettingsSource(settings_cls),
            file_secret_settings,
        )

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
                engine=db.engine.lower() if db.engine else "postgres",
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
            fitting=FittingSettings(
                default_max_iterations=self.fitting.default_max_iterations,
                max_iterations_upper_bound=self.fitting.max_iterations_upper_bound,
                save_best_default=self.fitting.save_best_default,
                parameter_initial_default=self.fitting.parameter_initial_default,
                parameter_min_default=self.fitting.parameter_min_default,
                parameter_max_default=self.fitting.parameter_max_default,
                preview_row_limit=self.fitting.preview_row_limit,
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


# -----------------------------------------------------------------------------
def _build_path_scoped_settings_class(config_path: str) -> type[AppSettings]:
    class PathScopedAppSettings(AppSettings):
        _configuration_file: ClassVar[str] = config_path

    return PathScopedAppSettings


# -----------------------------------------------------------------------------
def _load_app_settings(settings_cls: type[AppSettings]) -> AppSettings:
    ensure_environment_loaded()
    try:
        return settings_cls()
    except ValidationError as exc:
        raise RuntimeError(f"Invalid application settings: {exc}") from exc


###############################################################################
@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    return _load_app_settings(AppSettings)


# -----------------------------------------------------------------------------
def get_server_settings(config_path: str | None = None) -> ServerSettings:
    if config_path:
        scoped_class = _build_path_scoped_settings_class(config_path)
        return _load_app_settings(scoped_class).to_server_settings()
    return get_app_settings().to_server_settings()


# -----------------------------------------------------------------------------
def reload_settings_for_tests() -> AppSettings:
    get_app_settings.cache_clear()
    return get_app_settings()
