from __future__ import annotations

import os
from pathlib import Path

from server.common.path import (
    DATASETS_PATH,
    ENV_FILE_PATH,
    CLIENT_INDEX_FILE_PATH,
    LOGS_PATH,
    TEMPLATES_PATH,
    TOKENIZERS_PATH,
)


def ensure_runtime_directories() -> None:
    for directory in (
        LOGS_PATH,
        DATASETS_PATH,
        TOKENIZERS_PATH,
        TEMPLATES_PATH,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def validate_runtime_files() -> None:
    if not ENV_FILE_PATH.is_file():
        raise RuntimeError(f"Environment file not found: {ENV_FILE_PATH}")


def validate_tauri_client_bundle(
    *,
    tauri_mode_enabled: bool,
    client_index_file_path: os.PathLike[str] | str = CLIENT_INDEX_FILE_PATH,
) -> None:
    client_index_path = Path(client_index_file_path)
    if tauri_mode_enabled and not client_index_path.is_file():
        raise RuntimeError(
            "TKBEN_TAURI_MODE is enabled but the packaged frontend build is missing. "
            f"Expected file: {client_index_path}"
        )


def build_cors_origins() -> list[str]:
    ui_host = _normalized_host(os.getenv("UI_HOST", "127.0.0.1"))
    ui_port = _normalized_port(os.getenv("UI_PORT", "8000"))

    hosts = {ui_host}
    if ui_host == "127.0.0.1":
        hosts.add("localhost")
    elif ui_host == "localhost":
        hosts.add("127.0.0.1")

    return sorted(f"http://{host}:{ui_port}" for host in hosts)


def run_startup_validations(
    *,
    tauri_mode_enabled: bool,
    client_index_file_path: os.PathLike[str] | str = CLIENT_INDEX_FILE_PATH,
) -> None:
    validate_runtime_files()
    ensure_runtime_directories()
    validate_tauri_client_bundle(
        tauri_mode_enabled=tauri_mode_enabled,
        client_index_file_path=client_index_file_path,
    )


def _normalized_host(raw_host: str) -> str:
    host = raw_host.strip() or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def _normalized_port(raw_port: str) -> str:
    port = raw_port.strip() or "8000"
    try:
        parsed = int(port)
    except ValueError as exc:
        raise RuntimeError(f"UI_PORT must be a valid integer, got: {raw_port}") from exc
    if parsed < 1 or parsed > 65535:
        raise RuntimeError(f"UI_PORT must be between 1 and 65535, got: {parsed}")
    return str(parsed)
