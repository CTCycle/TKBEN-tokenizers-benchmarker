from __future__ import annotations

from pathlib import Path


###############################################################################
ROOT_DIR = Path(__file__).resolve().parents[3]
PROJECT_DIR = ROOT_DIR / "app"
SETTING_PATH = ROOT_DIR / "settings"
RESOURCES_PATH = PROJECT_DIR / "resources"
SOURCES_PATH = RESOURCES_PATH / "sources"
DATASETS_PATH = SOURCES_PATH / "datasets"
TOKENIZERS_PATH = SOURCES_PATH / "tokenizers"
LOGS_PATH = RESOURCES_PATH / "logs"
TEMPLATES_PATH = RESOURCES_PATH / "templates"
ENV_FILE_PATH = SETTING_PATH / ".env"
CONFIGURATIONS_FILE = SETTING_PATH / "configurations.json"
DATABASE_PATH = RESOURCES_PATH / "database.db"
CLIENT_DIST_PATH = PROJECT_DIR / "client" / "dist"
CLIENT_ASSETS_PATH = CLIENT_DIST_PATH / "assets"
CLIENT_INDEX_FILE_PATH = CLIENT_DIST_PATH / "index.html"
