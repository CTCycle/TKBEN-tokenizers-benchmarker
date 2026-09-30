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
    if not configured_path:
        return (ROOT_DIR / "data").resolve()

    data_path = Path(configured_path).expanduser()
    if not data_path.is_absolute():
        data_path = ROOT_DIR / data_path
    return data_path.resolve()


DATA_PATH = _resolve_data_path(os.getenv("TKBEN_DATA_DIR"))
SOURCES_PATH = DATA_PATH / "sources"
DATASETS_PATH = SOURCES_PATH / "datasets"
TOKENIZERS_PATH = SOURCES_PATH / "tokenizers"
LOGS_PATH = Path(os.getenv("TKBEN_LOG_DIR", DATA_PATH / "logs")).resolve()
TEMPLATES_PATH = DATA_PATH / "templates"
ENV_FILE_PATH = SETTINGS_DIR / ".env"
ENV_EXAMPLE_FILE_PATH = SETTINGS_DIR / ".env.example"
DATABASE_PATH = DATA_PATH / "database.db"
__all__ = [
    "APP_DIR",
    "ASSETS_DIR",
    "CACHE_PATH",
    "CLIENT_DIR",
    "DATABASE_PATH",
    "DATASETS_PATH",
    "ENV_EXAMPLE_FILE_PATH",
    "ENV_FILE_PATH",
    "FIGURES_DIR",
    "LOGS_PATH",
    "DATA_PATH",
    "ROOT_DIR",
    "SERVER_DIR",
    "SETTINGS_DIR",
    "SOURCES_PATH",
    "TEMPLATES_PATH",
    "TESTS_DIR",
    "TOKENIZERS_PATH",
]
