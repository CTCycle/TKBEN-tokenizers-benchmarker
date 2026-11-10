# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import pytest
from pathlib import Path
from server.repositories.runtime_settings import DatabaseRuntimeSettingsPersistence
from server.configurations.runtime import RuntimeSettingsStore
from server.configurations.settings import (
    DatabaseSettings as DatabaseSettings__runtime_settings_persistence,
)
from server.configurations.settings import build_server_settings
from server.contracts.settings import DashboardUiPreferences
from server.repositories.database.initializer import initialize_database
from server.repositories.database.backend import (
    TKBENDatabase as TKBENDatabase__runtime_settings_persistence,
)
import os
from collections.abc import Iterator
import json
from server.configurations.database_configuration import DATABASE_CONFIGURATION_FILENAME
from server.configurations.database_configuration import DatabaseConfigurationStore
from server.configurations.database_configuration import StoredDatabaseConfiguration
from server.contracts.database import DatabaseConfigurationRequest
from typing import get_args
from pydantic import BaseModel
from pydantic import ValidationError
from server.configurations import runtime as runtime_module
from server.configurations.runtime import RuntimeSettingsConflictError
from server.configurations.runtime import RuntimeSettingsPersistenceError
from server.configurations.runtime import RuntimeSettingsValidationError
from server.configurations.settings import BenchmarkSettings
from server.configurations.settings import (
    DatabaseSettings as DatabaseSettings__runtime_settings,
)
from server.configurations.settings import DatasetSettings
from server.configurations.settings import JobsSettings
from server.configurations.settings import NetworkSettings
from server.configurations.settings import PathSettings
from server.configurations.settings import SecuritySettings
from server.configurations.settings import ServerSettings
from server.configurations.settings import TokenizerSettings
from server.contracts.settings import RuntimeSettingsPatchRequest
from server.common.path import ROOT_DIR
from server.common.utils import encryption
from server.common.utils.encryption import (
    KeyMaterialNotInitializedError,
    get_hf_key_cipher,
    initialize_key_material,
    is_key_material_initialized,
)
from server.configurations import environment as bootstrap
from server.configurations.startup import get_server_settings
from server.configurations.startup import is_key_reveal_enabled
from server.configurations.startup import reset_settings_cache_for_tests


###############################################################################
def _database_settings(path: Path) -> DatabaseSettings__runtime_settings_persistence:
    return DatabaseSettings__runtime_settings_persistence(
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
    database = TKBENDatabase__runtime_settings_persistence(database_settings)
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
    assert (
        independent.get_dashboard_preferences_state().preferences
        == DashboardUiPreferences()
    )
    assert independent.get_dashboard_preferences_state().revision == 0


###############################################################################


_RUNTIME_ENV_NAMES = (
    "TKBEN_PATHS_CONFIG",
    "FASTAPI_HOST",
    "FASTAPI_PORT",
    "UI_HOST",
    "UI_PORT",
    "VITE_API_BASE_URL",
    "ALLOW_KEY_REVEAL",
)
_BASE_ENVIRONMENT = {name: os.environ.get(name) for name in _RUNTIME_ENV_NAMES}


###############################################################################
@pytest.fixture(autouse=True)
def restore_runtime_environment() -> Iterator[None]:
    """Keep dotenv-backed settings tests isolated from process environment leaks."""
    for name in _RUNTIME_ENV_NAMES:
        os.environ.pop(name, None)

    yield

    for name in _RUNTIME_ENV_NAMES:
        original = _BASE_ENVIRONMENT[name]
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


###############################################################################

# Copyright © 2026 Thomas Virdis


###############################################################################
def _store(tmp_path: Path) -> DatabaseConfigurationStore:
    material_path = tmp_path / "data" / "hf-key-material.json"
    initialize_key_material(material_path)
    return DatabaseConfigurationStore(tmp_path / "data", material_path)


###############################################################################
def test_default_configuration_is_file_based_and_independent(tmp_path: Path) -> None:
    store = _store(tmp_path)

    configuration = store.load()

    assert store.path.name == DATABASE_CONFIGURATION_FILENAME
    assert configuration.backend == "sqlite"
    assert (
        store.to_database_settings(configuration).sqlite_path
        == (tmp_path / "data" / "database.db").resolve()
    )
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert payload["backend"] == "sqlite"
    assert "FASTAPI_PORT" not in store.path.read_text(encoding="utf-8")


###############################################################################
def test_invalid_primary_file_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{broken", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Database configuration is invalid"):
        store.load()

    assert store.path.read_text(encoding="utf-8") == "{broken"


###############################################################################
def test_atomic_write_failure_preserves_last_valid_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    original = store.load()
    original_bytes = store.path.read_bytes()
    candidate = store.build_candidate(backend="sqlite", sqlite_path="alternate.db")

    def fail_replace(_source: str, _destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(
        "server.configurations.database_configuration.os.replace", fail_replace
    )
    with pytest.raises(RuntimeError, match="Unable to persist database configuration"):
        store.save(candidate)

    assert store.load() == original
    assert store.path.read_bytes() == original_bytes
    assert not list(store.path.parent.glob(".database-config.json.*.tmp"))


###############################################################################
def test_candidate_does_not_reuse_password_for_a_different_target(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    initial = store.build_candidate(
        backend="postgresql",
        host="db1.example.test",
        port=5432,
        database_name="one",
        username="alice",
        password="secret-one",
    )
    store.save(initial)

    other = store.build_candidate(
        backend="postgresql",
        host="db2.example.test",
        port=5432,
        database_name="two",
        username="bob",
    )
    same_target = store.build_candidate(
        backend="postgresql",
        host="db1.example.test",
        port=5432,
        database_name="one",
        username="alice",
    )

    assert other.encrypted_password is None
    assert same_target.encrypted_password == initial.encrypted_password


###############################################################################
def test_sqlite_configuration_rejects_blank_paths_and_postgresql_fields() -> None:
    with pytest.raises(ValueError, match="SQLite database path is required"):
        DatabaseConfigurationRequest(backend="sqlite", sqlite_path="  ")

    with pytest.raises(ValueError, match="PostgreSQL fields are only valid"):
        DatabaseConfigurationRequest(
            backend="sqlite",
            sqlite_path="database.db",
            port=5432,
        )

    with pytest.raises(ValueError, match="PostgreSQL fields are not allowed"):
        StoredDatabaseConfiguration(
            schema_version=1,
            backend="sqlite",
            sqlite_path="database.db",
            ssl=True,
        )


###############################################################################
class _FilePersistenceForStoreUnitTests:
    """Small test adapter; production persistence is the active SQL database."""

    # -------------------------------------------------------------------------
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fail_writes = False

    # -------------------------------------------------------------------------
    def load(self) -> dict[str, object] | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    # -------------------------------------------------------------------------
    def save(self, payload: dict[str, object]) -> None:
        if self.fail_writes:
            raise OSError("test persistence failure")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


###############################################################################
class _PersistedRuntimeSettingsStore(RuntimeSettingsStore):

    # -------------------------------------------------------------------------
    def __init__(self, defaults: ServerSettings, path: Path) -> None:
        self.persistence = _FilePersistenceForStoreUnitTests(path)
        self.path = path
        super().__init__(defaults, self.persistence)


###############################################################################
def _store__runtime_settings(
    defaults: ServerSettings, path: Path
) -> _PersistedRuntimeSettingsStore:
    return _PersistedRuntimeSettingsStore(defaults, path)


###############################################################################
def _non_finite_runtime_float_cases():
    cases = []
    for group, group_field in RuntimeSettingsPatchRequest.model_fields.items():
        if group == "expected_revision":
            continue
        patch_model = next(
            option
            for option in get_args(group_field.annotation)
            if isinstance(option, type) and issubclass(option, BaseModel)
        )
        settings_model = ServerSettings.model_fields[group].annotation
        for name in patch_model.model_fields:
            default = settings_model.model_fields[name].default
            if isinstance(default, float):
                for value in (float("nan"), float("inf"), -float("inf")):
                    cases.append((group, name, value))
    return cases


###############################################################################
def _server_settings(tmp_path: Path) -> ServerSettings:
    data = tmp_path / "data"
    return ServerSettings(
        database=DatabaseSettings__runtime_settings(
            embedded_database=True,
            sqlite_path=data / "database.db",
            host=None,
            port=None,
            database_name=None,
            username=None,
            password=None,
            ssl=False,
            ssl_ca=None,
            connect_timeout=30,
            insert_batch_size=1000,
        ),
        paths=PathSettings(
            data=data,
            sources=data / "sources",
            datasets=data / "sources/datasets",
            tokenizers=data / "sources/tokenizers",
            logs=data / "logs",
            templates=data / "templates",
        ),
        network=NetworkSettings(
            fastapi_host="127.0.0.1",
            fastapi_port=5000,
            ui_host="127.0.0.1",
            ui_port=8000,
            api_base_url="/api",
        ),
        security=SecuritySettings(
            allow_key_reveal=False,
            hf_keys_encryption_material_file=data / "hf-key-material.json",
        ),
        datasets=DatasetSettings(),
        tokenizers=TokenizerSettings(),
        benchmarks=BenchmarkSettings(),
        jobs=JobsSettings(),
    )


###############################################################################
def test_default_only_startup_and_missing_runtime_file(tmp_path: Path) -> None:
    defaults = _server_settings(tmp_path)
    path = tmp_path / "runtime.json"

    store = _store__runtime_settings(defaults, path)
    state = store.get_state()

    assert state.settings == defaults
    assert state.defaults == defaults
    assert state.revision == 0
    assert state.overridden_keys == ()
    assert state.warning is None
    assert not path.exists()


###############################################################################
def test_valid_partial_overrides_merge_and_persist_sparsely(tmp_path: Path) -> None:
    defaults = _server_settings(tmp_path)
    path = tmp_path / "runtime.json"
    store = _store__runtime_settings(defaults, path)

    first = store.apply_patch(
        {"datasets": {"histogram_bins": 30}},
        expected_revision=0,
    )
    second = store.apply_patch(
        {"jobs": {"polling_interval": 2.0}},
        expected_revision=1,
    )

    assert first.settings.datasets.histogram_bins == 30
    assert second.settings.datasets.histogram_bins == 30
    assert second.settings.jobs.polling_interval == 2.0
    assert second.overridden_keys == (
        "datasets.histogram_bins",
        "jobs.polling_interval",
    )
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "dashboard_preferences": {
            "baseline_by_report_id": {},
            "tokenizer_colors": {},
            "hidden_widget_ids": [],
            "ordered_widget_ids": [],
            "visualization_by_widget_id": {},
        },
        "dashboard_revision": 0,
        "overrides": {
            "datasets": {"histogram_bins": 30},
            "jobs": {"polling_interval": 2.0},
        },
        "revision": 2,
        "schema_version": 1,
    }
    assert "network" not in path.read_text(encoding="utf-8")


###############################################################################
def test_dashboard_preferences_have_an_independent_revision_and_survive_reload(
    tmp_path: Path,
) -> None:
    defaults = _server_settings(tmp_path)
    path = tmp_path / "runtime.json"
    store = _store__runtime_settings(defaults, path)
    store.apply_patch({"datasets": {"histogram_bins": 30}}, expected_revision=0)
    preferences = DashboardUiPreferences(
        ordered_widget_ids=["speed", "size"],
        hidden_widget_ids=["size"],
        visualization_by_widget_id={"speed": "horizontal_bar"},
        baseline_by_report_id={"42": "tokenizer-a"},
    )

    saved = store.set_dashboard_preferences(preferences, expected_revision=0)
    restarted = _store__runtime_settings(defaults, path)

    assert saved.revision == 1
    assert store.get_state().revision == 1
    assert restarted.get_state().settings.datasets.histogram_bins == 30
    assert restarted.get_dashboard_preferences_state().revision == 1
    assert restarted.get_dashboard_preferences_state().preferences == preferences


###############################################################################
@pytest.mark.parametrize(
    "patch",
    [
        {"datasets": {"histogram_bins": 4}},
        {"datasets": {"histogram_bins": "30"}},
        {"settings": {"histogram_bins": 30}},
        {"network": {"fastapi_port": 9999}},
        {"datasets": {"unknown": 30}},
    ],
)
def test_invalid_unknown_and_environment_fields_are_rejected(
    tmp_path: Path,
    patch: dict[str, dict[str, object]],
) -> None:
    store = _store__runtime_settings(
        _server_settings(tmp_path), tmp_path / "runtime.json"
    )

    with pytest.raises(RuntimeSettingsValidationError):
        store.apply_patch(patch, expected_revision=0)

    assert store.get_state().revision == 0
    assert not store.path.exists()


###############################################################################
def test_tokenizer_cross_field_validation_remains_authoritative(tmp_path: Path) -> None:
    store = _store__runtime_settings(
        _server_settings(tmp_path), tmp_path / "runtime.json"
    )

    with pytest.raises(RuntimeSettingsValidationError):
        store.apply_patch(
            {"tokenizers": {"max_discovery_limit": 10}},
            expected_revision=0,
        )

    with pytest.raises(RuntimeSettingsValidationError):
        store.apply_patch(
            {"tokenizers": {"max_discovery_candidates": 20}},
            expected_revision=0,
        )

    assert store.get_state().settings.tokenizers.max_discovery_limit == 250


###############################################################################
@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": 2,
            "revision": 4,
            "overrides": {},
            "dashboard_revision": 0,
            "dashboard_preferences": {},
        },
        {
            "schema_version": 1,
            "revision": 4,
            "overrides": {"network": {}},
            "dashboard_revision": 0,
            "dashboard_preferences": {},
        },
        {
            "schema_version": 1,
            "revision": 4,
            "overrides": {"jobs": {"polling_interval": 0.1}},
            "dashboard_revision": 0,
            "dashboard_preferences": {},
        },
    ],
)
def test_invalid_schema_or_override_file_falls_back_with_warning(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    original = path.read_bytes()

    store = _store__runtime_settings(_server_settings(tmp_path), path)
    state = store.get_state()

    assert state.settings.datasets.histogram_bins == 20
    assert state.revision == 0
    assert state.overridden_keys == ()
    assert state.warning == runtime_module.RUNTIME_SETTINGS_WARNING
    assert path.read_bytes() == original


###############################################################################
def test_malformed_runtime_json_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    path.write_text("{not-json", encoding="utf-8")

    store = _store__runtime_settings(_server_settings(tmp_path), path)

    assert store.get_state().warning is not None
    assert path.read_text(encoding="utf-8") == "{not-json"


###############################################################################
def test_successful_save_replaces_corrupt_stored_document(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    path.write_text("{not-json", encoding="utf-8")

    store = _store__runtime_settings(_server_settings(tmp_path), path)
    state = store.apply_patch(
        {"datasets": {"histogram_bins": 20}},
        expected_revision=0,
    )

    assert state.warning is None
    assert state.revision == 1
    assert state.overridden_keys == ()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "dashboard_preferences": {
            "baseline_by_report_id": {},
            "tokenizer_colors": {},
            "hidden_widget_ids": [],
            "ordered_widget_ids": [],
            "visualization_by_widget_id": {},
        },
        "dashboard_revision": 0,
        "schema_version": 1,
        "revision": 1,
        "overrides": {},
    }


###############################################################################
def test_revision_conflict_does_not_write(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    store = _store__runtime_settings(_server_settings(tmp_path), path)
    store.apply_patch({"datasets": {"histogram_bins": 30}}, expected_revision=0)
    original = path.read_bytes()

    with pytest.raises(RuntimeSettingsConflictError) as error:
        store.apply_patch(
            {"datasets": {"histogram_bins": 31}},
            expected_revision=0,
        )

    assert error.value.actual_revision == 1
    assert path.read_bytes() == original
    assert store.get_state().settings.datasets.histogram_bins == 30


###############################################################################
def test_reset_one_value_and_equal_default_remove_the_override(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    store = _store__runtime_settings(_server_settings(tmp_path), path)
    store.apply_patch(
        {
            "datasets": {"histogram_bins": 30, "streaming_batch_size": 20_000},
            "jobs": {"polling_interval": 2.0},
        },
        expected_revision=0,
    )

    reset_one = store.reset(
        expected_revision=1,
        keys=["datasets.histogram_bins"],
    )
    assert reset_one.settings.datasets.histogram_bins == 20
    assert reset_one.settings.datasets.streaming_batch_size == 20_000
    assert reset_one.overridden_keys == (
        "datasets.streaming_batch_size",
        "jobs.polling_interval",
    )

    reset_to_default = store.apply_patch(
        {"datasets": {"streaming_batch_size": 10_000}},
        expected_revision=2,
    )
    assert reset_to_default.settings.datasets.streaming_batch_size == 10_000
    assert reset_to_default.overridden_keys == ("jobs.polling_interval",)


###############################################################################
def test_reset_all_persists_an_empty_database_document(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    store = _store__runtime_settings(_server_settings(tmp_path), path)
    store.apply_patch({"datasets": {"histogram_bins": 30}}, expected_revision=0)

    state = store.reset(expected_revision=1)

    assert state.settings.datasets.histogram_bins == 20
    assert state.overridden_keys == ()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "dashboard_preferences": {
            "baseline_by_report_id": {},
            "tokenizer_colors": {},
            "hidden_widget_ids": [],
            "ordered_widget_ids": [],
            "visualization_by_widget_id": {},
        },
        "dashboard_revision": 0,
        "schema_version": 1,
        "revision": 2,
        "overrides": {},
    }


###############################################################################
def test_reset_all_keeps_revision_of_an_unchanged_empty_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.json"
    payload = {
        "schema_version": 1,
        "revision": 3,
        "overrides": {},
        "dashboard_revision": 0,
        "dashboard_preferences": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    store = _store__runtime_settings(_server_settings(tmp_path), path)

    state = store.reset(expected_revision=3)

    assert state.revision == 3
    assert state.overridden_keys == ()
    assert json.loads(path.read_text(encoding="utf-8")) == payload


###############################################################################
def test_persistence_failure_rolls_back_state_and_stored_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.json"
    store = _store__runtime_settings(_server_settings(tmp_path), path)
    store.apply_patch({"datasets": {"histogram_bins": 30}}, expected_revision=0)
    original = path.read_bytes()

    store.persistence.fail_writes = True
    with pytest.raises(RuntimeSettingsPersistenceError):
        store.apply_patch(
            {"datasets": {"histogram_bins": 31}},
            expected_revision=1,
        )

    assert path.read_bytes() == original
    assert store.get_state().settings.datasets.histogram_bins == 30
    assert store.get_state().revision == 1


###############################################################################
def test_process_restart_and_reload_load_persisted_overrides(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    defaults = _server_settings(tmp_path)
    first = _store__runtime_settings(defaults, path)
    first.apply_patch(
        {"datasets": {"histogram_bins": defaults.datasets.histogram_bins + 1}},
        expected_revision=0,
    )

    sparse_file = json.loads(path.read_text(encoding="utf-8"))
    assert sparse_file["overrides"] == {
        "datasets": {"histogram_bins": defaults.datasets.histogram_bins + 1}
    }

    restarted = _store__runtime_settings(defaults, path)
    restored = restarted.get_state()
    assert (
        restored.settings.datasets.histogram_bins
        == defaults.datasets.histogram_bins + 1
    )
    assert (
        restored.settings.datasets.model_copy(
            update={"histogram_bins": defaults.datasets.histogram_bins}
        )
        == defaults.datasets
    )
    assert restored.settings.tokenizers == defaults.tokenizers
    assert restored.settings.benchmarks == defaults.benchmarks
    assert restored.settings.jobs == defaults.jobs
    assert restarted.get_state().revision == 1
    assert restored.warning is None


###############################################################################
@pytest.mark.parametrize(
    ("group", "name", "value"),
    _non_finite_runtime_float_cases(),
)
def test_non_finite_float_update_preserves_persisted_snapshot(
    tmp_path: Path,
    group: str,
    name: str,
    value: float,
) -> None:
    defaults = _server_settings(tmp_path)
    path = tmp_path / "runtime.json"
    store = _store__runtime_settings(defaults, path)
    store.apply_patch(
        {
            "benchmarks": {
                "default_batch_size": defaults.benchmarks.default_batch_size + 1
            }
        },
        expected_revision=0,
    )
    before = store.get_state()
    persisted_before = path.read_bytes()
    patch = {group: {name: value}}
    patch.setdefault("datasets", {})["histogram_bins"] = (
        defaults.datasets.histogram_bins + 1
    )

    with pytest.raises(RuntimeSettingsValidationError):
        store.apply_patch(patch, expected_revision=before.revision)

    assert store.get_state() == before
    assert path.read_bytes() == persisted_before


###############################################################################
def test_runtime_override_models_are_frozen(tmp_path: Path) -> None:
    store = _store__runtime_settings(
        _server_settings(tmp_path), tmp_path / "runtime.json"
    )
    state = store.apply_patch(
        {"datasets": {"histogram_bins": 30}},
        expected_revision=0,
    )

    with pytest.raises(ValidationError):
        state.settings.datasets.histogram_bins = 20


###############################################################################
def test_benchmark_defaults_persist_sparsely_and_reset_independently(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.json"
    store = _store__runtime_settings(_server_settings(tmp_path), path)

    saved = store.apply_patch(
        {
            "benchmarks": {
                "default_max_documents": 2500,
                "default_batch_size": 32,
                "default_parallelism": 4,
            }
        },
        expected_revision=0,
    )

    assert saved.settings.benchmarks.default_max_documents == 2500
    assert saved.settings.benchmarks.default_batch_size == 32
    assert saved.settings.benchmarks.default_parallelism == 4
    assert saved.overridden_keys == (
        "benchmarks.default_max_documents",
        "benchmarks.default_batch_size",
        "benchmarks.default_parallelism",
    )
    assert json.loads(path.read_text(encoding="utf-8"))["overrides"] == {
        "benchmarks": {
            "default_batch_size": 32,
            "default_max_documents": 2500,
            "default_parallelism": 4,
        }
    }

    reset = store.reset(
        expected_revision=1,
        keys=["benchmarks.default_batch_size"],
    )
    assert reset.settings.benchmarks.default_batch_size == 16
    assert reset.settings.benchmarks.default_max_documents == 2500
    assert reset.settings.benchmarks.default_parallelism == 4


###############################################################################
@pytest.fixture(autouse=True)
def reset_configuration_state() -> Iterator[None]:
    previous_cors_origins = os.environ.pop("CORS_ORIGINS", None)
    reset_settings_cache_for_tests()
    bootstrap.reset_environment_bootstrap_for_tests()
    yield
    reset_settings_cache_for_tests()
    bootstrap.reset_environment_bootstrap_for_tests()
    os.environ.pop("CORS_ORIGINS", None)
    if previous_cors_origins is not None:
        os.environ["CORS_ORIGINS"] = previous_cors_origins


###############################################################################
def _write_env(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


###############################################################################
def _configure_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data_dir: str | Path,
    log_dir: str | Path | None = None,
) -> None:
    config_path = tmp_path / "paths-config.json"
    payload = {"data_dir": str(data_dir)}
    if log_dir is not None:
        payload["log_dir"] = str(log_dir)
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("TKBEN_PATHS_CONFIG", str(config_path))


###############################################################################
def _configure_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_lines: list[str],
    *,
    data_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
) -> None:
    _configure_paths(
        tmp_path, monkeypatch, data_dir or tmp_path / "runtime-data", log_dir
    )
    env_path = tmp_path / ".env"
    _write_env(env_path, env_lines)
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)
    reset_settings_cache_for_tests()
    bootstrap.reset_environment_bootstrap_for_tests()


###############################################################################
def test_bootstrap_preserves_existing_process_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    _write_env(
        env_path,
        ["FASTAPI_HOST=from_dotenv", "UI_HOST=from-dotenv-ui"],
    )
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)
    monkeypatch.setenv("FASTAPI_HOST", "from_process")
    monkeypatch.setenv("UI_HOST", "from-process-ui")

    bootstrap.ensure_environment_loaded()

    assert os.getenv("FASTAPI_HOST") == "from_process"
    assert os.getenv("UI_HOST") == "from-process-ui"


###############################################################################
def test_missing_environment_is_created_from_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    template_path = tmp_path / ".env.example"
    template_bytes = b"FASTAPI_HOST=from_template\n"
    template_path.write_bytes(template_bytes)
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)
    monkeypatch.setattr(bootstrap, "ENV_EXAMPLE_FILE_PATH", template_path)

    assert bootstrap.ensure_environment_loaded() == env_path
    assert env_path.read_bytes() == template_bytes
    assert os.getenv("FASTAPI_HOST") == "from_template"


###############################################################################
def test_environment_template_exposes_canonical_runtime_inputs() -> None:
    example = (ROOT_DIR / "settings/.env.example").read_text(encoding="utf-8")

    for removed in (
        "TKBEN_DATA_DIR",
        "TKBEN_LOG_DIR",
        "HF_KEYS_ENCRYPTION_MATERIAL_FILE",
        "KERAS_BACKEND",
        "MPLBACKEND",
    ):
        assert removed not in example
    assert "UI_HOST=127.0.0.1" in example
    assert "CORS_ORIGINS=\n" in example
    assert "DATABASE_" not in example
    assert "ALLOW_KEY_REVEAL=false" in example


###############################################################################
def test_default_runtime_paths_use_repository_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path, [])
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)
    monkeypatch.setenv("TKBEN_PATHS_CONFIG", str(tmp_path / "missing-paths.json"))
    reset_settings_cache_for_tests()
    bootstrap.reset_environment_bootstrap_for_tests()

    settings = get_server_settings()
    default_root = (ROOT_DIR / "data").resolve()

    assert settings.paths.data == default_root
    assert settings.paths.logs == (default_root / "logs").resolve()
    assert settings.database.sqlite_path == (default_root / "database.db").resolve()
    assert settings.network.cors_origins is None
    assert (
        settings.security.hf_keys_encryption_material_file
        == (default_root / "hf-key-material.json").resolve()
    )


###############################################################################
def test_bootstrap_is_idempotent_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path, ["FASTAPI_HOST=first"])
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)

    bootstrap.ensure_environment_loaded()
    _write_env(env_path, ["FASTAPI_HOST=second"])
    bootstrap.ensure_environment_loaded()

    assert os.getenv("FASTAPI_HOST") == "first"


###############################################################################
def test_settings_initialization_refreshes_after_explicit_bootstrap_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path, ["FASTAPI_PORT=5101"])
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", env_path)

    first = get_server_settings()

    _write_env(env_path, ["FASTAPI_PORT=5102"])
    reset_settings_cache_for_tests()
    bootstrap.reset_environment_bootstrap_for_tests()

    refreshed = get_server_settings()

    assert first.network.fastapi_port == 5101
    assert refreshed.network.fastapi_port == 5102


###############################################################################
def test_typed_defaults_resolve_without_structured_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "runtime-data"
    _configure_environment(tmp_path, monkeypatch, [], data_dir=data_root)

    settings = get_server_settings()

    assert settings.tokenizers.default_discovery_limit == 50
    assert settings.tokenizers.max_discovery_limit == 250
    assert settings.tokenizers.max_discovery_candidates == 750
    assert settings.tokenizers.metadata_candidate_multiplier == 3
    assert settings.tokenizers.max_upload_bytes == 10 * 1024 * 1024
    assert settings.datasets.histogram_bins == 20
    assert settings.datasets.streaming_batch_size == 10_000
    assert settings.datasets.max_upload_bytes == 25 * 1024 * 1024
    assert settings.datasets.download_timeout_seconds == 180.0
    assert settings.datasets.download_retry_attempts == 3
    assert settings.datasets.download_retry_backoff_seconds == 2.0
    assert settings.benchmarks.streaming_batch_size == 1000
    assert settings.jobs.polling_interval == 1.0


###############################################################################
def test_runtime_environment_resolves_into_one_settings_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "runtime-data"
    log_root = tmp_path / "runtime-logs"
    _configure_environment(
        tmp_path,
        monkeypatch,
        [
            "FASTAPI_HOST=0.0.0.0",
            "FASTAPI_PORT=5100",
            "UI_HOST=localhost",
            "UI_PORT=8100",
            "VITE_API_BASE_URL=/api",
            "CORS_ORIGINS= https://Example.com, http://localhost:8000,https://example.com ",
            "ALLOW_KEY_REVEAL=true",
        ],
        data_dir=data_root,
        log_dir=log_root,
    )

    settings = get_server_settings()

    assert settings.paths.data == data_root.resolve()
    assert settings.paths.datasets == (data_root / "sources/datasets").resolve()
    assert settings.paths.tokenizers == (data_root / "sources/tokenizers").resolve()
    assert settings.paths.logs == log_root.resolve()
    assert settings.database.embedded_database is True
    assert settings.database.sqlite_path == (data_root / "database.db").resolve()
    assert settings.network.fastapi_host == "0.0.0.0"
    assert settings.network.fastapi_port == 5100
    assert settings.network.ui_host == "localhost"
    assert settings.network.ui_port == 8100
    assert settings.network.cors_origins == (
        "https://example.com",
        "http://localhost:8000",
    )
    assert settings.security.allow_key_reveal is True
    assert (
        settings.security.hf_keys_encryption_material_file
        == (data_root / "hf-key-material.json").resolve()
    )
    saved = (data_root / "database-config.json").read_text(encoding="utf-8")
    assert "secret" not in saved
    assert "DATABASE_" not in (tmp_path / ".env").read_text(encoding="utf-8")


###############################################################################
def test_blank_cors_origins_leave_origins_derived_from_ui_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_environment(
        tmp_path,
        monkeypatch,
        ["CORS_ORIGINS=", "UI_HOST=localhost", "UI_PORT=8100"],
    )

    settings = get_server_settings()

    assert settings.network.cors_origins is None
    assert settings.network.ui_host == "localhost"
    assert settings.network.ui_port == 8100


###############################################################################
@pytest.mark.parametrize(
    "value",
    [
        "*",
        "ftp://example.com",
        "https://example.com/api",
        "http://user@example.com",
        "http://example.com:70000",
        "http://example.com,,https://other.example",
    ],
)
def test_invalid_cors_origins_fail_during_settings_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    _configure_environment(tmp_path, monkeypatch, [f"CORS_ORIGINS={value}"])

    with pytest.raises(RuntimeError, match="CORS_ORIGINS"):
        get_server_settings()


###############################################################################
def test_relative_runtime_paths_are_repository_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_environment(
        tmp_path,
        monkeypatch,
        [],
        data_dir="custom/runtime-data",
        log_dir="custom/runtime-logs",
    )

    bootstrap.ensure_environment_loaded()
    settings = build_server_settings()

    assert settings.paths.data == (ROOT_DIR / "custom/runtime-data").resolve()
    assert settings.paths.logs == (ROOT_DIR / "custom/runtime-logs").resolve()


###############################################################################
def test_boolean_aliases_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_environment(
        tmp_path,
        monkeypatch,
        ["ALLOW_KEY_REVEAL=on"],
    )

    with pytest.raises(RuntimeError, match="ALLOW_KEY_REVEAL"):
        get_server_settings()


###############################################################################
def test_settings_models_are_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_environment(tmp_path, monkeypatch, [])
    settings = get_server_settings()

    with pytest.raises(ValidationError):
        settings.datasets.histogram_bins = 99


###############################################################################
def test_runtime_setting_models_keep_cross_field_and_polling_validation() -> None:
    with pytest.raises(ValidationError):
        TokenizerSettings(default_discovery_limit=251)
    with pytest.raises(ValidationError):
        TokenizerSettings(default_discovery_limit=20, max_discovery_limit=10)
    with pytest.raises(ValidationError):
        TokenizerSettings(max_discovery_limit=300, max_discovery_candidates=300)
    with pytest.raises(ValidationError):
        JobsSettings(polling_interval=0.1)


###############################################################################
def test_key_reveal_policy_uses_canonical_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_environment(
        tmp_path,
        monkeypatch,
        ["ALLOW_KEY_REVEAL=true"],
    )

    assert is_key_reveal_enabled() is True


###############################################################################
def test_hf_key_cipher_requires_explicit_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material_path = tmp_path / "hf-key-material.json"
    monkeypatch.setattr(encryption, "DATA_PATH", tmp_path)

    assert is_key_material_initialized() is False
    with pytest.raises(KeyMaterialNotInitializedError):
        get_hf_key_cipher()
    assert not material_path.exists()

    initialize_key_material()
    cipher = get_hf_key_cipher()

    assert is_key_material_initialized() is True
    assert cipher.decrypt(cipher.encrypt("hf_test")) == "hf_test"
    with pytest.raises(FileExistsError):
        initialize_key_material()
