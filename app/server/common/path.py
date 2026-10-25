# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
import os
from pathlib import Path


###############################################################################
ROOT_DIR = Path(__file__).resolve().parents[3]
SERVER_DIR = ROOT_DIR / "app" / "server"
CACHE_PATH = (ROOT_DIR / "runtimes" / "cache").resolve()

###############################################################################
def _resolve_data_path(configured_path: str | None) -> Path:
    return resolve_runtime_path(configured_path, ROOT_DIR / "data")


###############################################################################
def resolve_runtime_path(configured_path: str | None, default_path: Path) -> Path:
    normalized = configured_path.strip() if configured_path else ""
    candidate = Path(normalized).expanduser() if normalized else default_path
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate.resolve()


###############################################################################
def paths_config_file() -> Path:
    override = os.getenv("TKBEN_PATHS_CONFIG", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return ROOT_DIR / "settings" / "paths-config.json"


###############################################################################
def read_path_configuration() -> tuple[str | None, str | None]:
    path = paths_config_file()
    if not path.is_file():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Path configuration file is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Path configuration file must contain an object: {path}")
    data_dir = payload.get("data_dir")
    log_dir = payload.get("log_dir")
    return (
        data_dir if isinstance(data_dir, str) else None,
        log_dir if isinstance(log_dir, str) else None,
    )


###############################################################################
def write_path_configuration(data_dir: str, log_dir: str) -> None:
    path = paths_config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"data_dir": data_dir, "log_dir": log_dir}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


DATA_PATH = _resolve_data_path(read_path_configuration()[0])
SOURCES_PATH = DATA_PATH / "sources"
DATASETS_PATH = SOURCES_PATH / "datasets"
TOKENIZERS_PATH = SOURCES_PATH / "tokenizers"
__all__ = [
    "CACHE_PATH",
    "DATASETS_PATH",
    "ROOT_DIR",
    "resolve_runtime_path",
    "SERVER_DIR",
    "TOKENIZERS_PATH",
    "paths_config_file",
    "read_path_configuration",
    "write_path_configuration",
]
