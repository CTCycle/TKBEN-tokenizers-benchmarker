# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

from server.configurations.runtime import RuntimeSettingsStore
from server.configurations.settings import DatabaseSettings, build_server_settings
from server.contracts.settings import DashboardUiPreferences
from server.repositories.database.initializer import initialize_database
from server.repositories.database.backend import TKBENDatabase
from server.repositories.runtime_settings import DatabaseRuntimeSettingsPersistence


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
def _initialized_store(path: Path) -> RuntimeSettingsStore:
    database_settings = _database_settings(path)
    settings = build_server_settings(database_settings=database_settings)
    initialize_database(settings=settings)
    database = TKBENDatabase(database_settings)
    return RuntimeSettingsStore(
        settings,
        DatabaseRuntimeSettingsPersistence(database),
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
