from __future__ import annotations

from pathlib import Path
from threading import RLock

from server.common.path import CONFIGURATIONS_FILE
from server.configurations.environment import ensure_environment_loaded
from server.configurations.management import ConfigurationManager
from server.configurations.settings import ServerSettings

_DEFAULT_SETTINGS_LOCK = RLock()
_DEFAULT_SETTINGS: ServerSettings | None = None


###############################################################################
def _resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        return CONFIGURATIONS_FILE
    return Path(config_path)


###############################################################################
def get_configuration_manager(
    config_path: str | Path | None = None,
) -> ConfigurationManager:
    ensure_environment_loaded(force=True)
    return ConfigurationManager(config_path=_resolve_config_path(config_path)).load()


###############################################################################
def get_server_settings(config_path: str | Path | None = None) -> ServerSettings:
    global _DEFAULT_SETTINGS
    if config_path is not None:
        return get_configuration_manager(config_path).server_settings
    with _DEFAULT_SETTINGS_LOCK:
        if _DEFAULT_SETTINGS is None:
            _DEFAULT_SETTINGS = get_configuration_manager().server_settings
        return _DEFAULT_SETTINGS


###############################################################################
def reload_settings_for_tests(config_path: str | Path | None = None) -> ServerSettings:
    global _DEFAULT_SETTINGS
    if config_path is not None:
        return get_configuration_manager(config_path).server_settings
    with _DEFAULT_SETTINGS_LOCK:
        _DEFAULT_SETTINGS = None
        _DEFAULT_SETTINGS = get_configuration_manager().server_settings
        return _DEFAULT_SETTINGS
