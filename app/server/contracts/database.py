# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator


class DatabaseConfigurationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    backend: Literal["sqlite", "postgresql"]
    sqlite_path: StrictStr | None = None
    host: StrictStr | None = None
    port: StrictInt | None = Field(default=None, ge=1, le=65535)
    database_name: StrictStr | None = None
    username: StrictStr | None = None
    password: StrictStr | None = Field(default=None, repr=False)
    clear_password: bool = False
    ssl: bool = False
    ssl_ca: StrictStr | None = None
    connect_timeout: StrictInt = Field(default=30, ge=1)
    insert_batch_size: StrictInt = Field(default=1000, ge=1)

    @model_validator(mode="after")
    def validate_selected_backend(self) -> "DatabaseConfigurationRequest":
        if self.backend == "sqlite":
            if not self.sqlite_path or not self.sqlite_path.strip():
                raise ValueError("SQLite database path is required.")
            if any(
                (
                    self.host,
                    self.port,
                    self.database_name,
                    self.username,
                    self.password,
                    self.clear_password,
                    self.ssl,
                    self.ssl_ca,
                )
            ):
                raise ValueError("PostgreSQL fields are only valid for PostgreSQL.")
            return self
        missing = [
            field
            for field, value in (
                ("host", self.host),
                ("database_name", self.database_name),
                ("username", self.username),
            )
            if not value or not value.strip()
        ]
        if missing:
            raise ValueError("PostgreSQL requires: " + ", ".join(missing))
        if self.sqlite_path:
            raise ValueError("SQLite path is only valid for SQLite.")
        if self.clear_password and self.password:
            raise ValueError("Set either a new password or clear the saved password.")
        return self


class DatabaseConfigurationResponse(BaseModel):
    backend: Literal["sqlite", "postgresql"]
    active: bool = True
    connection_status: Literal["connected", "unavailable"]
    sqlite_path: str | None = None
    host: str | None = None
    port: int | None = None
    database_name: str | None = None
    username: str | None = None
    password_configured: bool = False
    ssl: bool = False
    ssl_ca: str | None = None
    connect_timeout: int
    insert_batch_size: int
    message: str | None = None


class DatabaseConnectionTestResponse(BaseModel):
    backend: Literal["sqlite", "postgresql"]
    success: bool
    message: str
