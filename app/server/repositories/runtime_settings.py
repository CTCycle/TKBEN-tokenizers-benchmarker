# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from server.repositories.database.backend import TKBENDatabase, get_database
from server.repositories.schemas.models import RuntimeSettingsRecord


class DatabaseRuntimeSettingsPersistence:
    """Persist the active sparse settings document in one database row."""

    def __init__(self, database: TKBENDatabase | None = None) -> None:
        self.database = database or get_database()

    def load(self) -> dict[str, Any] | None:
        with Session(bind=self.database.backend.engine) as session:
            record = session.get(RuntimeSettingsRecord, 1)
            if record is None:
                return None
            return {
                "schema_version": record.schema_version,
                "revision": record.revision,
                "dashboard_revision": record.dashboard_revision,
                "overrides": record.overrides,
                "dashboard_preferences": record.dashboard_preferences,
            }

    def save(self, payload: dict[str, Any]) -> None:
        with Session(bind=self.database.backend.engine) as session:
            record = session.get(RuntimeSettingsRecord, 1)
            if record is None:
                record = RuntimeSettingsRecord(
                    id=1,
                    schema_version=int(payload["schema_version"]),
                    revision=int(payload["revision"]),
                    dashboard_revision=int(payload.get("dashboard_revision", 0)),
                    overrides=dict(payload["overrides"]),
                    dashboard_preferences=dict(
                        payload.get("dashboard_preferences", {})
                    ),
                )
                session.add(record)
            else:
                record.schema_version = int(payload["schema_version"])
                record.revision = int(payload["revision"])
                record.dashboard_revision = int(
                    payload.get("dashboard_revision", record.dashboard_revision)
                )
                record.overrides = dict(payload["overrides"])
                record.dashboard_preferences = dict(
                    payload.get("dashboard_preferences", record.dashboard_preferences)
                )
            session.commit()
