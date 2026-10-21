# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
from pathlib import Path

from server.configurations.runtime import RuntimeSettingsStore
from server.configurations.settings import DatabaseSettings, build_server_settings
from server.contracts.settings import DashboardUiPreferences
from server.repositories.database.initializer import initialize_database
from server.repositories.database.backend import TKBENDatabase
from server.repositories.runtime_settings import DatabaseRuntimeSettingsPersistence
from server.repositories.schemas.models import RuntimeSettingsRecord
from sqlalchemy.orm import Session


###############################################################################
def _database_settings(path: Path) -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=True,
        sqlite_path=path,
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=5,
        insert_batch_size=100,
    )


###############################################################################
def _initialized_store(path: Path, legacy_path: Path | None = None) -> RuntimeSettingsStore:
    database_settings = _database_settings(path)
    settings = build_server_settings(database_settings=database_settings)
    initialize_database(settings=settings)
    database = TKBENDatabase(database_settings)
    return RuntimeSettingsStore(
        settings,
        DatabaseRuntimeSettingsPersistence(database),
        legacy_path,
    )


###############################################################################
def test_runtime_settings_survive_restart_in_their_own_database(tmp_path: Path) -> None:
    first_path = tmp_path / "first.db"
    second_path = tmp_path / "second.db"
    first = _initialized_store(first_path)
    _initialized_store(second_path)

    updated = first.apply_patch(
        {"datasets": {"histogram_bins": 31}},
        expected_revision=0,
    )
    preferences = DashboardUiPreferences(
        ordered_widget_ids=["speed"],
        hidden_widget_ids=[],
        visualization_by_widget_id={"speed": "bar"},
        baseline_by_report_id={"7": "alpha"},
    )
    first.set_dashboard_preferences(preferences, expected_revision=0)
    restarted = _initialized_store(first_path)
    independent = _initialized_store(second_path)

    assert updated.revision == 1
    assert restarted.get_state().settings.datasets.histogram_bins == 31
    assert restarted.get_state().revision == 1
    assert restarted.get_dashboard_preferences_state().preferences == preferences
    assert restarted.get_dashboard_preferences_state().revision == 1
    assert independent.get_state().settings.datasets.histogram_bins == 20
    assert independent.get_state().revision == 0
    assert independent.get_dashboard_preferences_state().preferences == DashboardUiPreferences()
    assert independent.get_dashboard_preferences_state().revision == 0
    assert not (tmp_path / "runtime-settings.json").exists()


###############################################################################
def test_legacy_runtime_file_is_removed_only_after_database_commit(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "runtime-settings.json"
    legacy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "revision": 4,
                "overrides": {"datasets": {"histogram_bins": 28}},
            }
        ),
        encoding="utf-8",
    )

    migrated = _initialized_store(tmp_path / "migration.db", legacy_path)

    assert migrated.get_state().settings.datasets.histogram_bins == 28
    assert migrated.get_state().revision == 4
    assert not legacy_path.exists()
    with Session(bind=TKBENDatabase(_database_settings(tmp_path / "migration.db")).engine) as session:
        record = session.get(RuntimeSettingsRecord, 1)
        assert record is not None
        assert record.revision == 4
        assert record.overrides == {"datasets": {"histogram_bins": 28}}
        assert record.dashboard_revision == 0
        assert record.dashboard_preferences == {
            "ordered_widget_ids": [],
            "hidden_widget_ids": [],
            "visualization_by_widget_id": {},
            "baseline_by_report_id": {},
        }


###############################################################################
def test_failed_legacy_migration_keeps_source_file(tmp_path: Path) -> None:
    legacy_path = tmp_path / "runtime-settings.json"
    legacy_path.write_text("{broken", encoding="utf-8")

    migrated = _initialized_store(tmp_path / "migration.db", legacy_path)

    assert migrated.get_state().warning is not None
    assert legacy_path.read_text(encoding="utf-8") == "{broken"
