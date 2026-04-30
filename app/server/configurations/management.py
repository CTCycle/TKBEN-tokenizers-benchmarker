from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from TKBEN.server.common.constants import CONFIGURATIONS_FILE
from TKBEN.server.domain.settings import JsonConfiguration, ServerSettings


###############################################################################
class ConfigurationManager:
    def __init__(self, config_path: str | Path = CONFIGURATIONS_FILE) -> None:
        self.config_path = Path(config_path)
        self._payload: dict[str, Any] = {}
        self._configuration = JsonConfiguration()

    # -------------------------------------------------------------------------
    @property
    def configuration(self) -> JsonConfiguration:
        return self._configuration

    # -------------------------------------------------------------------------
    @property
    def server_settings(self) -> ServerSettings:
        return self._configuration.to_server_settings()

    # -------------------------------------------------------------------------
    def load(self) -> "ConfigurationManager":
        payload = self._read_payload()
        self._configuration = self._validate_configuration(payload)
        self._payload = payload
        return self

    # -------------------------------------------------------------------------
    def reload(self) -> "ConfigurationManager":
        return self.load()

    # -------------------------------------------------------------------------
    def update(self, payload: dict[str, Any], *, persist: bool = True) -> "ConfigurationManager":
        if not isinstance(payload, dict):
            raise RuntimeError("Configuration must be a JSON object.")
        self._configuration = self._validate_configuration(payload)
        self._payload = payload
        if persist:
            self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self

    # -------------------------------------------------------------------------
    def get_block(self, block_name: str) -> dict[str, Any]:
        block = self._payload.get(block_name, {})
        if isinstance(block, dict):
            return dict(block)
        return {}

    # -------------------------------------------------------------------------
    def get_value(self, block_name: str, key: str, default: Any = None) -> Any:
        return self.get_block(block_name).get(key, default)

    # -------------------------------------------------------------------------
    def _read_payload(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise RuntimeError(f"Configuration file not found: {self.config_path}")

        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to load configuration from {self.config_path}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Configuration must be a JSON object.")
        return payload

    # -------------------------------------------------------------------------
    def _validate_configuration(self, payload: dict[str, Any]) -> JsonConfiguration:
        try:
            return JsonConfiguration.from_payload(payload)
        except ValidationError as exc:
            raise RuntimeError(f"Invalid application settings: {exc}") from exc
