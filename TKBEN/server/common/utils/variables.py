from __future__ import annotations

import os

from TKBEN.server.configurations.bootstrap import ensure_environment_loaded


###############################################################################
class EnvironmentVariables:
    def __init__(self) -> None:
        self.env_path = ensure_environment_loaded()

    # -------------------------------------------------------------------------
    def get(self, key: str, default: str | None = None) -> str | None:
        return os.getenv(key, default)


env_variables = EnvironmentVariables()
