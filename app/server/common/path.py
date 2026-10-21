# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

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


DATA_PATH = _resolve_data_path(os.getenv("TKBEN_DATA_DIR"))
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
]
