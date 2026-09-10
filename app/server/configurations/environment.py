from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

from server.common.constants import ALLOW_KEY_REVEAL_DEFAULT


ROOT_DIR = Path(__file__).resolve().parents[3]
SETTINGS_DIR = (ROOT_DIR / "settings").resolve()
ENV_FILE_PATH = SETTINGS_DIR / ".env"
ENV_EXAMPLE_FILE_PATH = SETTINGS_DIR / ".env.example"


###############################################################################
@dataclass
class EnvironmentBootstrapState:
    lock: Lock = field(default_factory=Lock)
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
        load_dotenv(dotenv_path=env_path, override=True)

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


###############################################################################
def is_key_reveal_enabled() -> bool:
    raw_value = os.getenv("ALLOW_KEY_REVEAL")
    if raw_value is None or raw_value.strip() == "":
        return ALLOW_KEY_REVEAL_DEFAULT

    normalized = raw_value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(
        f"ALLOW_KEY_REVEAL must be either 'true' or 'false', got: {raw_value}"
    )
