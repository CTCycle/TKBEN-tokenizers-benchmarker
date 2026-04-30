from __future__ import annotations

from pathlib import Path

from TKBEN.server.common.constants import CONFIGURATIONS_FILE
from TKBEN.server.configurations.environment import ensure_environment_loaded
from TKBEN.server.configurations.management import ConfigurationManager
from TKBEN.server.domain.settings import ServerSettings


# -----------------------------------------------------------------------------
def _resolve_config_path(config_path: str | None = None) -> str:
    return str(Path(config_path or CONFIGURATIONS_FILE))


###############################################################################
def get_configuration_manager(config_path: str | None = None) -> ConfigurationManager:
    ensure_environment_loaded()
    return ConfigurationManager(config_path=_resolve_config_path(config_path)).load()


# -----------------------------------------------------------------------------
def get_server_settings(config_path: str | None = None) -> ServerSettings:
    return get_configuration_manager(config_path).server_settings


# -----------------------------------------------------------------------------
def reload_settings_for_tests(config_path: str | None = None) -> ServerSettings:
    return get_server_settings(config_path)
