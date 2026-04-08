from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from TKBEN.server.common.utils.types import coerce_str_sequence

from TKBEN.server.configurations import (
    ensure_mapping,
    load_configurations,
)
from TKBEN.server.common.constants import CONFIGURATIONS_FILE
from TKBEN.server.common.utils.types import (
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_str_or_none,
)
from TKBEN.server.common.utils.variables import env_variables


# [SERVER SETTINGS]
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


# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class FittingSettings:
    default_max_iterations: int
    max_iterations_upper_bound: int
    save_best_default: bool
    parameter_initial_default: float
    parameter_min_default: float
    parameter_max_default: float
    preview_row_limit: int


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TokenizerSettings:
    default_scan_limit: int
    max_scan_limit: int
    min_scan_limit: int
    max_upload_bytes: int


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkSettings:
    streaming_batch_size: int
    log_interval: int


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class JobsSettings:
    polling_interval: float


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ServerSettings:
    database: DatabaseSettings
    datasets: DatasetSettings
    fitting: FittingSettings
    tokenizers: TokenizerSettings
    benchmarks: BenchmarkSettings
    jobs: JobsSettings


# [BUILDER FUNCTIONS]
###############################################################################
def build_database_settings(payload: dict[str, Any] | Any) -> DatabaseSettings:
    embedded = coerce_bool(payload.get("embedded_database"), True)
    insert_batch_value = payload.get("insert_batch_size")
    if embedded:
        return DatabaseSettings(
            embedded_database=True,
            engine=None,
            host=None,
            port=None,
            database_name=None,
            username=None,
            password=None,
            ssl=False,
            ssl_ca=None,
            connect_timeout=coerce_int(payload.get("connect_timeout"), 10, minimum=1),
            insert_batch_size=coerce_int(insert_batch_value, 1000, minimum=1),
        )

    engine_value = (
        coerce_str_or_none(payload.get("engine"))
        or "postgres"
    )
    normalized_engine = engine_value.lower() if engine_value else None

    return DatabaseSettings(
        embedded_database=False,
        engine=normalized_engine,
        host=coerce_str_or_none(payload.get("host")),
        port=coerce_int(payload.get("port"), 5432, minimum=1, maximum=65535),
        database_name=coerce_str_or_none(payload.get("database_name")),
        username=coerce_str_or_none(payload.get("username")),
        password=coerce_str_or_none(payload.get("password")),
        ssl=coerce_bool(payload.get("ssl"), False),
        ssl_ca=coerce_str_or_none(payload.get("ssl_ca")),
        connect_timeout=coerce_int(payload.get("connect_timeout"), 10, minimum=1),
        insert_batch_size=coerce_int(insert_batch_value, 1000, minimum=1),
    )


# -----------------------------------------------------------------------------
def build_dataset_settings(payload: dict[str, Any] | Any) -> DatasetSettings:
    download_timeout_value = env_variables.get("DATASET_DOWNLOAD_TIMEOUT_SECONDS")
    if download_timeout_value is None:
        download_timeout_value = payload.get("download_timeout_seconds")

    download_retry_attempts_value = env_variables.get("DATASET_DOWNLOAD_RETRY_ATTEMPTS")
    if download_retry_attempts_value is None:
        download_retry_attempts_value = payload.get("download_retry_attempts")

    download_retry_backoff_value = env_variables.get(
        "DATASET_DOWNLOAD_RETRY_BACKOFF_SECONDS"
    )
    if download_retry_backoff_value is None:
        download_retry_backoff_value = payload.get("download_retry_backoff_seconds")

    return DatasetSettings(
        allowed_extensions=coerce_str_sequence(
            payload.get("allowed_extensions"), [".csv", ".xls", ".xlsx"]
        ),
        column_detection_cutoff=coerce_float(
            payload.get("column_detection_cutoff"), 0.6, minimum=0.0, maximum=1.0
        ),
        max_upload_bytes=coerce_int(
            payload.get("max_upload_bytes"), 25 * 1024 * 1024, minimum=1
        ),
        histogram_bins=coerce_int(
            payload.get("histogram_bins"), 20, minimum=5, maximum=100
        ),
        streaming_batch_size=coerce_int(
            payload.get("streaming_batch_size"), 10000, minimum=100
        ),
        log_interval=coerce_int(payload.get("log_interval"), 100000, minimum=1000),
        cleanup_downloaded_sources=coerce_bool(
            payload.get("cleanup_downloaded_sources"), False
        ),
        download_timeout_seconds=coerce_float(
            download_timeout_value, 120.0, minimum=1.0
        ),
        download_retry_attempts=coerce_int(
            download_retry_attempts_value, 3, minimum=1, maximum=10
        ),
        download_retry_backoff_seconds=coerce_float(
            download_retry_backoff_value, 1.0, minimum=0.0, maximum=60.0
        ),
    )


# -----------------------------------------------------------------------------
def build_fitting_settings(payload: dict[str, Any] | Any) -> FittingSettings:
    default_iterations = coerce_int(
        payload.get("default_max_iterations"), 1000, minimum=1
    )
    upper_bound = coerce_int(
        payload.get("max_iterations_upper_bound"), 1_000_000, minimum=default_iterations
    )
    parameter_initial_default = coerce_float(
        payload.get("default_parameter_initial"), 1.0, minimum=0.0
    )
    parameter_min_default = coerce_float(
        payload.get("default_parameter_min"), 0.0, minimum=0.0
    )
    parameter_max_default = coerce_float(
        payload.get("default_parameter_max"), 100.0, minimum=parameter_min_default
    )
    return FittingSettings(
        default_max_iterations=default_iterations,
        max_iterations_upper_bound=upper_bound,
        save_best_default=coerce_bool(payload.get("save_best_default"), True),
        parameter_initial_default=parameter_initial_default,
        parameter_min_default=parameter_min_default,
        parameter_max_default=parameter_max_default,
        preview_row_limit=coerce_int(payload.get("preview_row_limit"), 5, minimum=1),
    )


# -----------------------------------------------------------------------------
def build_tokenizer_settings(payload: dict[str, Any] | Any) -> TokenizerSettings:
    min_limit = coerce_int(payload.get("min_scan_limit"), 1, minimum=1)
    max_limit = coerce_int(payload.get("max_scan_limit"), 1000, minimum=min_limit)
    default_limit = coerce_int(
        payload.get("default_scan_limit"), 100, minimum=min_limit, maximum=max_limit
    )
    return TokenizerSettings(
        default_scan_limit=default_limit,
        max_scan_limit=max_limit,
        min_scan_limit=min_limit,
        max_upload_bytes=coerce_int(
            payload.get("max_upload_bytes"), 10 * 1024 * 1024, minimum=1
        ),
    )


# -----------------------------------------------------------------------------
def build_benchmark_settings(payload: dict[str, Any] | Any) -> BenchmarkSettings:
    return BenchmarkSettings(
        streaming_batch_size=coerce_int(
            payload.get("streaming_batch_size"), 1000, minimum=100
        ),
        log_interval=coerce_int(payload.get("log_interval"), 10000, minimum=100),
    )


# -----------------------------------------------------------------------------
def build_jobs_settings(payload: dict[str, Any] | Any) -> JobsSettings:
    return JobsSettings(
        polling_interval=coerce_float(payload.get("polling_interval"), 1.0),
    )


# -----------------------------------------------------------------------------
def build_server_settings(payload: dict[str, Any] | Any) -> ServerSettings:
    database_payload = ensure_mapping(payload.get("database"))
    dataset_payload = ensure_mapping(payload.get("datasets"))
    fitting_payload = ensure_mapping(payload.get("fitting"))
    tokenizers_payload = ensure_mapping(payload.get("tokenizers"))
    benchmarks_payload = ensure_mapping(payload.get("benchmarks"))
    jobs_payload = ensure_mapping(payload.get("jobs"))

    return ServerSettings(
        database=build_database_settings(database_payload),
        datasets=build_dataset_settings(dataset_payload),
        fitting=build_fitting_settings(fitting_payload),
        tokenizers=build_tokenizer_settings(tokenizers_payload),
        benchmarks=build_benchmark_settings(benchmarks_payload),
        jobs=build_jobs_settings(jobs_payload),
    )


# [SERVER CONFIGURATION LOADER]
###############################################################################
def get_server_settings(config_path: str | None = None) -> ServerSettings:
    path = config_path or CONFIGURATIONS_FILE
    payload = load_configurations(path)

    return build_server_settings(payload)


server_settings = get_server_settings()
