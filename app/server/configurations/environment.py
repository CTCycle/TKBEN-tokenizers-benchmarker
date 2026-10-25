# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import os
from threading import Lock

from dotenv import dotenv_values


ROOT_DIR = Path(__file__).resolve().parents[3]
SETTINGS_DIR = (ROOT_DIR / "settings").resolve()
ENV_FILE_PATH = SETTINGS_DIR / ".env"
ENV_EXAMPLE_FILE_PATH = SETTINGS_DIR / ".env.example"
_BOOTSTRAP_ENVIRONMENT_KEYS = frozenset(
    {
        "FASTAPI_HOST",
        "FASTAPI_PORT",
        "UI_HOST",
        "UI_PORT",
        "VITE_API_BASE_URL",
        "CORS_ORIGINS",
        "ALLOW_KEY_REVEAL",
    }
)

###############################################################################
@dataclass
class EnvironmentBootstrapState:
    lock: Lock = field(default_factory=Lock)
    loaded_values: dict[str, str] = field(default_factory=dict)
    bootstrapped: bool = False

###############################################################################
@lru_cache(maxsize=1)
def _bootstrap_state() -> EnvironmentBootstrapState:
    return EnvironmentBootstrapState()

###############################################################################
def ensure_environment_loaded(*, force: bool = False) -> Path | None:
    state = _bootstrap_state()
    with state.lock:
        env_path = ENV_FILE_PATH
        if state.bootstrapped and not force and env_path.is_file():
            return env_path

        _ensure_environment_file(env_path)
        if env_path.is_file():
            for name, value in dotenv_values(env_path).items():
                if value is None or name not in _BOOTSTRAP_ENVIRONMENT_KEYS:
                    continue
                current_value = os.environ.get(name)
                previously_loaded = state.loaded_values.get(name)
                if previously_loaded is not None and current_value == previously_loaded:
                    os.environ[name] = value
                    state.loaded_values[name] = value
                elif current_value is None:
                    os.environ[name] = value
                    state.loaded_values[name] = value
                else:
                    state.loaded_values.pop(name, None)

        state.bootstrapped = True
        return env_path if env_path.is_file() else None

###############################################################################
def _ensure_environment_file(env_path: Path) -> None:
    if env_path.is_file():
        return
    if not ENV_EXAMPLE_FILE_PATH.is_file():
        raise RuntimeError(f"Environment template not found: {ENV_EXAMPLE_FILE_PATH}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    template_bytes = ENV_EXAMPLE_FILE_PATH.read_bytes()
    try:
        with env_path.open("xb") as destination:
            destination.write(template_bytes)
    except FileExistsError:
        return

###############################################################################
def reset_environment_bootstrap_for_tests() -> None:
    state = _bootstrap_state()
    with state.lock:
        state.bootstrapped = False
