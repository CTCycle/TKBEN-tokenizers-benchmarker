from __future__ import annotations

from pathlib import Path
from threading import RLock

from pydantic import ValidationError

from server.common.path import CONFIGURATIONS_FILE
from server.configurations.environment import ensure_environment_loaded
from server.configurations.settings import (
    ApplicationConfiguration,
    ServerSettings,
    build_server_settings,
)


_DEFAULT_SETTINGS_LOCK = RLock()
_default_settings: ServerSettings | None = None


###############################################################################
def _resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        return CONFIGURATIONS_FILE
    return Path(config_path)


###############################################################################
def _load_server_settings(config_path: str | Path | None = None) -> ServerSettings:
    ensure_environment_loaded(force=True)
    path = _resolve_config_path(config_path)
    try:
        configuration = ApplicationConfiguration.from_path(path)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid application settings: {exc}") from exc
    return build_server_settings(configuration)


###############################################################################
def get_server_settings(config_path: str | Path | None = None) -> ServerSettings:
    global _default_settings
    if config_path is not None:
        return _load_server_settings(config_path)
    with _DEFAULT_SETTINGS_LOCK:
        if _default_settings is None:
            _default_settings = _load_server_settings()
        return _default_settings


###############################################################################
def is_key_reveal_enabled() -> bool:
    return get_server_settings().security.allow_key_reveal


###############################################################################
def reset_settings_cache_for_tests() -> None:
    global _default_settings
    with _DEFAULT_SETTINGS_LOCK:
        _default_settings = None
