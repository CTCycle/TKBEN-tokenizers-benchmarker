from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from server.common.constants import ALLOW_KEY_REVEAL_DEFAULT
from server.common.utils.types import coerce_bool
from server.common.path import ENV_EXAMPLE_FILE_PATH, ENV_FILE_PATH
from server.common.utils.logger import logger
from server.domain.bootstrap import EnvironmentBootstrapState

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
        # .env is the active runtime profile and deliberately overrides process env.
        load_dotenv(dotenv_path=env_path, override=True)

        state.bootstrapped = True
        return env_path if env_path.is_file() else None

###############################################################################
def _ensure_environment_file(env_path: Path) -> None:
    if env_path.is_file():
        return
    if not ENV_EXAMPLE_FILE_PATH.is_file():
        raise RuntimeError(
            f"Environment template not found: {ENV_EXAMPLE_FILE_PATH}"
        )

    env_path.parent.mkdir(parents=True, exist_ok=True)
    template_bytes = ENV_EXAMPLE_FILE_PATH.read_bytes()
    try:
        with env_path.open("xb") as destination:
            destination.write(template_bytes)
    except FileExistsError:
        # Another process created the file after the existence check. Preserve it.
        return

    logger.info("Created environment file from template: %s", env_path)

###############################################################################
def reset_environment_bootstrap_for_tests() -> None:
    state = _bootstrap_state()
    with state.lock:
        state.bootstrapped = False

###############################################################################
def is_key_reveal_enabled() -> bool:
    return coerce_bool(
        os.getenv("ALLOW_KEY_REVEAL"),
        ALLOW_KEY_REVEAL_DEFAULT,
    )
