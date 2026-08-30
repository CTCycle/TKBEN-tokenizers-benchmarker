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
QA_DIR = ROOT_DIR / "QA"
SETTINGS_DIR = (ROOT_DIR / "settings").resolve()


###############################################################################
def _resolve_resource_path(configured_path: str | None) -> Path:
    if not configured_path:
        return (APP_DIR / "resources").resolve()

    resource_path = Path(configured_path).expanduser()
    if not resource_path.is_absolute():
        resource_path = ROOT_DIR / resource_path
    return resource_path.resolve()


RESOURCES_PATH = _resolve_resource_path(os.getenv("TKBEN_DATA_DIR"))
SOURCES_PATH = RESOURCES_PATH / "sources"
DATASETS_PATH = SOURCES_PATH / "datasets"
TOKENIZERS_PATH = SOURCES_PATH / "tokenizers"
LOGS_PATH = Path(os.getenv("TKBEN_LOG_DIR", RESOURCES_PATH / "logs")).resolve()
TEMPLATES_PATH = RESOURCES_PATH / "templates"
ENV_FILE_PATH = SETTINGS_DIR / ".env"
ENV_EXAMPLE_FILE_PATH = SETTINGS_DIR / ".env.example"
CONFIGURATIONS_FILE = SETTINGS_DIR / "configurations.json"
DATABASE_PATH = RESOURCES_PATH / "database.db"
__all__ = [
    "APP_DIR",
    "ASSETS_DIR",
    "CLIENT_DIR",
    "CONFIGURATIONS_FILE",
    "DATABASE_PATH",
    "DATASETS_PATH",
    "ENV_EXAMPLE_FILE_PATH",
    "ENV_FILE_PATH",
    "FIGURES_DIR",
    "LOGS_PATH",
    "QA_DIR",
    "RESOURCES_PATH",
    "ROOT_DIR",
    "SERVER_DIR",
    "SETTINGS_DIR",
    "SOURCES_PATH",
    "TEMPLATES_PATH",
    "TESTS_DIR",
    "TOKENIZERS_PATH",
]
