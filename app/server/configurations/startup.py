from __future__ import annotations

from pathlib import Path
from threading import RLock

from server.common.path import CONFIGURATIONS_FILE
from server.configurations.environment import ensure_environment_loaded
from server.configurations.management import ConfigurationManager
from server.configurations.settings import ServerSettings

_DEFAULT_SETTINGS_LOCK = RLock()
_default_settings: ServerSettings | None = None


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
    global _default_settings
    if config_path is not None:
        return get_configuration_manager(config_path).server_settings
    with _DEFAULT_SETTINGS_LOCK:
        if _default_settings is None:
            _default_settings = get_configuration_manager().server_settings
        return _default_settings


###############################################################################
def reload_settings_for_tests(config_path: str | Path | None = None) -> ServerSettings:
    global _default_settings
    if config_path is not None:
        return get_configuration_manager(config_path).server_settings
    with _DEFAULT_SETTINGS_LOCK:
        _default_settings = None
        _default_settings = get_configuration_manager().server_settings
        return _default_settings
