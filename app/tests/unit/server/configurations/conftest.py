from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


_RUNTIME_ENV_NAMES = (
    "TKBEN_DATA_DIR",
    "TKBEN_LOG_DIR",
    "FASTAPI_HOST",
    "FASTAPI_PORT",
    "UI_HOST",
    "UI_PORT",
    "VITE_API_BASE_URL",
    "DATABASE_EMBEDDED",
    "DATABASE_CONNECT_TIMEOUT",
    "DATABASE_INSERT_BATCH_SIZE",
    "DATABASE_HOST",
    "DATABASE_PORT",
    "DATABASE_NAME",
    "DATABASE_USERNAME",
    "DATABASE_PASSWORD",
    "DATABASE_SSL",
    "DATABASE_SSL_CA",
    "ALLOW_KEY_REVEAL",
    "HF_KEYS_ENCRYPTION_MATERIAL_FILE",
)
_BASE_ENVIRONMENT = {name: os.environ.get(name) for name in _RUNTIME_ENV_NAMES}


@pytest.fixture(autouse=True)
def restore_runtime_environment() -> Iterator[None]:
    """Keep dotenv-backed settings tests isolated from process environment leaks."""
    for name in _RUNTIME_ENV_NAMES:
        original = _BASE_ENVIRONMENT[name]
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original

    yield

    for name in _RUNTIME_ENV_NAMES:
        original = _BASE_ENVIRONMENT[name]
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original
