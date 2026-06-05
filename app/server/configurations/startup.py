from __future__ import annotations

from pathlib import Path

from server.common.path import CONFIGURATIONS_FILE
from server.configurations.environment import ensure_environment_loaded
from server.configurations.management import ConfigurationManager
from server.domain.settings import ServerSettings


###############################################################################
def _resolve_config_path(config_path: str | Path | None = None) -> Path:
    return Path(config_path or CONFIGURATIONS_FILE)


###############################################################################
def get_configuration_manager(
    config_path: str | Path | None = None,
) -> ConfigurationManager:
    ensure_environment_loaded(force=True)
    return ConfigurationManager(config_path=_resolve_config_path(config_path)).load()


###############################################################################
def get_server_settings(config_path: str | Path | None = None) -> ServerSettings:
    return get_configuration_manager(config_path).server_settings


###############################################################################
def reload_settings_for_tests(config_path: str | Path | None = None) -> ServerSettings:
    return get_server_settings(config_path)
