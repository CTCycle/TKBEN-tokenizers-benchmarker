# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import os
from pathlib import Path


###############################################################################
ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = ROOT_DIR / "app"
SERVER_DIR = APP_DIR / "server"
CLIENT_DIR = APP_DIR / "client"
TESTS_DIR = APP_DIR / "tests"
ASSETS_DIR = ROOT_DIR / "assets"
FIGURES_DIR = ASSETS_DIR / "figures"
SETTINGS_DIR = (ROOT_DIR / "settings").resolve()
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


DATA_PATH = _resolve_data_path(os.getenv("TKBEN_DATA_DIR"))
SOURCES_PATH = DATA_PATH / "sources"
DATASETS_PATH = SOURCES_PATH / "datasets"
TOKENIZERS_PATH = SOURCES_PATH / "tokenizers"
LOGS_PATH = resolve_runtime_path(os.getenv("TKBEN_LOG_DIR"), DATA_PATH / "logs")
TEMPLATES_PATH = DATA_PATH / "templates"
ENV_FILE_PATH = SETTINGS_DIR / ".env"
ENV_EXAMPLE_FILE_PATH = SETTINGS_DIR / ".env.example"
__all__ = [
    "APP_DIR",
    "ASSETS_DIR",
    "CACHE_PATH",
    "CLIENT_DIR",
    "DATASETS_PATH",
    "ENV_EXAMPLE_FILE_PATH",
    "ENV_FILE_PATH",
    "FIGURES_DIR",
    "LOGS_PATH",
    "DATA_PATH",
    "ROOT_DIR",
    "resolve_runtime_path",
    "SERVER_DIR",
    "SETTINGS_DIR",
    "SOURCES_PATH",
    "TEMPLATES_PATH",
    "TESTS_DIR",
    "TOKENIZERS_PATH",
]
