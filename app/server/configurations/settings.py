# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from ipaddress import IPv6Address
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from server.common.path import ROOT_DIR, read_path_configuration, resolve_runtime_path


###############################################################################
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
    data: Path
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
    cors_origins: tuple[str, ...] | None = None


###############################################################################
class SecuritySettings(_FrozenSettingsModel):
    allow_key_reveal: bool
    hf_keys_encryption_material_file: Path


###############################################################################
class DatasetSettings(_FrozenSettingsModel):
    allowed_extensions: tuple[str, ...] = (".csv", ".xlsx")
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
    histogram_bins: int = Field(default=20, ge=5, le=100)
    streaming_batch_size: int = Field(default=10_000, ge=100)
    log_interval: int = Field(default=100_000, ge=1000)
    download_timeout_seconds: float = Field(default=180.0, ge=1.0)
    download_retry_attempts: int = Field(default=3, ge=1, le=10)
    download_retry_backoff_seconds: float = Field(
        default=2.0,
        ge=0.0,
        le=60.0,
    )


###############################################################################
class TokenizerSettings(_FrozenSettingsModel):
    default_discovery_limit: int = Field(default=50, ge=1, le=250)
    max_discovery_limit: int = Field(default=250, ge=1, le=250)
    max_discovery_candidates: int = Field(default=750, ge=1)
    metadata_candidate_multiplier: int = Field(default=3, ge=1, le=10)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)

    # -------------------------------------------------------------------------
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
    default_max_documents: int = Field(default=1000, ge=1, le=100_000)
    default_batch_size: int = Field(default=16, ge=1, le=4096)
    default_parallelism: int = Field(default=1, ge=1, le=128)
    streaming_batch_size: int = Field(default=1000, ge=100)
    log_interval: int = Field(default=10_000, ge=100)


###############################################################################
class JobsSettings(_FrozenSettingsModel):
    polling_interval: float = Field(default=1.0, ge=0.25)
    max_stored_jobs: int = Field(default=1000, ge=1)


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
    raise RuntimeError(f"{name} must be either 'true' or 'false', got: {raw_value}")


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
def _read_env_origins(name: str) -> tuple[str, ...] | None:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return None

    origins: list[str] = []
    for raw_origin in raw_value.split(","):
        origin = raw_origin.strip()
        if not origin:
            raise RuntimeError(
                f"{name} must be a comma-separated list of HTTP(S) origins."
            )
        normalized_origin = _normalize_cors_origin(name, origin)
        if normalized_origin not in origins:
            origins.append(normalized_origin)
    return tuple(origins)


###############################################################################
def _normalize_cors_origin(name: str, origin: str) -> str:
    try:
        parsed = urlsplit(origin)
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError(
            f"{name} must contain valid HTTP(S) origins without paths or credentials."
        ) from exc

    scheme = parsed.scheme.lower()
    hostname = parsed.hostname
    if (
        scheme not in {"http", "https"}
        or not parsed.netloc
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or "?" in origin
        or "#" in origin
        or parsed.netloc.endswith(":")
        or any(character.isspace() for character in origin)
        or port == 0
    ):
        raise RuntimeError(
            f"{name} must contain valid HTTP(S) origins without paths or credentials."
        )

    if ":" in hostname:
        try:
            normalized_host = f"[{IPv6Address(hostname).compressed}]"
        except ValueError as exc:
            raise RuntimeError(f"{name} contains an invalid IPv6 origin.") from exc
    else:
        try:
            normalized_host = hostname.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise RuntimeError(f"{name} contains an invalid hostname.") from exc
        hostname_labels = normalized_host.removesuffix(".").split(".")
        if not normalized_host or any(
            not label
            or len(label) > 63
            or re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label) is None
            for label in hostname_labels
        ):
            raise RuntimeError(f"{name} contains an invalid hostname.")

    port_suffix = ""
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        port_suffix = f":{port}"
    return f"{scheme}://{normalized_host}{port_suffix}"


###############################################################################
def _load_path_settings() -> PathSettings:
    configured_data, configured_logs = read_path_configuration()
    data = resolve_runtime_path(configured_data, ROOT_DIR / "data")
    sources = data / "sources"
    return PathSettings(
        data=data,
        sources=sources,
        datasets=sources / "datasets",
        tokenizers=sources / "tokenizers",
        logs=resolve_runtime_path(configured_logs, data / "logs"),
        templates=data / "templates",
    )


###############################################################################
def _load_database_settings(paths: PathSettings) -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=True,
        sqlite_path=paths.data / "database.db",
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=30,
        insert_batch_size=1000,
    )


###############################################################################
def _load_network_settings() -> NetworkSettings:
    return NetworkSettings(
        fastapi_host=_read_env_text("FASTAPI_HOST", "127.0.0.1"),
        fastapi_port=_read_env_int("FASTAPI_PORT", 5000, minimum=1, maximum=65535),
        ui_host=_read_env_text("UI_HOST", "127.0.0.1"),
        ui_port=_read_env_int("UI_PORT", 8000, minimum=1, maximum=65535),
        api_base_url=_read_env_text("VITE_API_BASE_URL", "/api"),
        cors_origins=_read_env_origins("CORS_ORIGINS"),
    )


###############################################################################
def _load_security_settings(paths: PathSettings) -> SecuritySettings:
    return SecuritySettings(
        allow_key_reveal=_read_env_bool("ALLOW_KEY_REVEAL", False),
        hf_keys_encryption_material_file=paths.data / "hf-key-material.json",
    )


###############################################################################
def build_server_settings(
    *, database_settings: DatabaseSettings | None = None
) -> ServerSettings:
    paths = _load_path_settings()
    return ServerSettings(
        database=database_settings or _load_database_settings(paths),
        paths=paths,
        network=_load_network_settings(),
        security=_load_security_settings(paths),
        datasets=DatasetSettings(),
        tokenizers=TokenizerSettings(),
        benchmarks=BenchmarkSettings(),
        jobs=JobsSettings(),
    )
