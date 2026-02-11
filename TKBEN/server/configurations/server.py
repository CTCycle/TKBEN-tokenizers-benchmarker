from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from TKBEN.server.utils.types import coerce_str_sequence

from TKBEN.server.configurations import (  
    ensure_mapping,
    load_configurations,
)
from TKBEN.server.utils.constants import CONFIGURATIONS_FILE
from TKBEN.server.utils.types import (
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_str_or_none,
)
from TKBEN.server.utils.variables import env_variables

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
    histogram_bins: int
    streaming_batch_size: int
    log_interval: int
    cleanup_downloaded_sources: bool

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
    embedded_value = payload.get("embedded_database")
    embedded = coerce_bool(embedded_value, True)

    insert_batch_value = env_variables.get("DB_INSERT_BATCH_SIZE") or payload.get("insert_batch_size")
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
            connect_timeout=coerce_int(
                env_variables.get("DB_CONNECT_TIMEOUT") or payload.get("connect_timeout"),
                10,
                minimum=1,
            ),
            insert_batch_size=coerce_int(insert_batch_value, 1000, minimum=1),
        )

    engine_value = (
        coerce_str_or_none(env_variables.get("DB_ENGINE"))
        or coerce_str_or_none(payload.get("engine"))
        or "postgres"
    )
    normalized_engine = engine_value.lower() if engine_value else None

    host_value = env_variables.get("DB_HOST") or payload.get("host")
    port_value = env_variables.get("DB_PORT") or payload.get("port")
    name_value = env_variables.get("DB_NAME") or payload.get("database_name")
    user_value = env_variables.get("DB_USER") or payload.get("username")
    password_value = env_variables.get("DB_PASSWORD") or payload.get("password")
    ssl_value = env_variables.get("DB_SSL") or payload.get("ssl")
    ssl_ca_value = env_variables.get("DB_SSL_CA") or payload.get("ssl_ca")
    timeout_value = env_variables.get("DB_CONNECT_TIMEOUT") or payload.get("connect_timeout")

    return DatabaseSettings(
        embedded_database=False,
        engine=normalized_engine,
        host=coerce_str_or_none(host_value),
        port=coerce_int(port_value, 5432, minimum=1, maximum=65535),
        database_name=coerce_str_or_none(name_value),
        username=coerce_str_or_none(user_value),
        password=coerce_str_or_none(password_value),
        ssl=coerce_bool(ssl_value, False),
        ssl_ca=coerce_str_or_none(ssl_ca_value),
        connect_timeout=coerce_int(timeout_value, 10, minimum=1),
        insert_batch_size=coerce_int(insert_batch_value, 1000, minimum=1),
    )

# -----------------------------------------------------------------------------
def build_dataset_settings(payload: dict[str, Any] | Any) -> DatasetSettings:
    return DatasetSettings(
        allowed_extensions=coerce_str_sequence(
            payload.get("allowed_extensions"), [".csv", ".xls", ".xlsx"]
        ),
        column_detection_cutoff=coerce_float(
            payload.get("column_detection_cutoff"), 0.6, minimum=0.0, maximum=1.0
        ),
        histogram_bins=coerce_int(
            payload.get("histogram_bins"), 20, minimum=5, maximum=100
        ),
        streaming_batch_size=coerce_int(
            payload.get("streaming_batch_size"), 10000, minimum=100
        ),
        log_interval=coerce_int(
            payload.get("log_interval"), 100000, minimum=1000
        ),
        cleanup_downloaded_sources=coerce_bool(
            payload.get("cleanup_downloaded_sources"), False
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
    )

# -----------------------------------------------------------------------------
def build_benchmark_settings(payload: dict[str, Any] | Any) -> BenchmarkSettings:
    return BenchmarkSettings(
        streaming_batch_size=coerce_int(
            payload.get("streaming_batch_size"), 1000, minimum=100
        ),
        log_interval=coerce_int(
            payload.get("log_interval"), 10000, minimum=100
        ),
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
