from __future__ import annotations

import os

from TKBEN.server.configurations.environment import ensure_environment_loaded


###############################################################################
def get_env_variable(key: str, default: str | None = None) -> str | None:
    ensure_environment_loaded()
    return os.getenv(key, default)
