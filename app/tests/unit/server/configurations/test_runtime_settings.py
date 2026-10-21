# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import pytest
from pydantic import BaseModel, ValidationError

from server.configurations import runtime as runtime_module
from server.configurations.runtime import (
    RuntimeSettingsConflictError,
    RuntimeSettingsPersistenceError,
    RuntimeSettingsStore,
    RuntimeSettingsValidationError,
)
from server.configurations.settings import (
    BenchmarkSettings,
    DatabaseSettings,
    DatasetSettings,
    JobsSettings,
    NetworkSettings,
    PathSettings,
    SecuritySettings,
    ServerSettings,
    TokenizerSettings,
)
from server.contracts.settings import DashboardUiPreferences
from server.contracts.settings import RuntimeSettingsPatchRequest

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
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


###############################################################################
class _PersistedRuntimeSettingsStore(RuntimeSettingsStore):

    # -------------------------------------------------------------------------
    def __init__(self, defaults: ServerSettings, path: Path) -> None:
        self.persistence = _FilePersistenceForStoreUnitTests(path)
        self.path = path
        super().__init__(defaults, self.persistence)


###############################################################################
def _store(defaults: ServerSettings, path: Path) -> _PersistedRuntimeSettingsStore:
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
        database=DatabaseSettings(
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

    store = _store(defaults, path)
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
    store = _store(defaults, path)

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
    store = _store(defaults, path)
    store.apply_patch({"datasets": {"histogram_bins": 30}}, expected_revision=0)
    preferences = DashboardUiPreferences(
        ordered_widget_ids=["speed", "size"],
        hidden_widget_ids=["size"],
        visualization_by_widget_id={"speed": "horizontal_bar"},
        baseline_by_report_id={"42": "tokenizer-a"},
    )

    saved = store.set_dashboard_preferences(preferences, expected_revision=0)
    restarted = _store(defaults, path)

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
    store = _store(_server_settings(tmp_path), tmp_path / "runtime.json")

    with pytest.raises(RuntimeSettingsValidationError):
        store.apply_patch(patch, expected_revision=0)

    assert store.get_state().revision == 0
    assert not store.path.exists()

###############################################################################
def test_tokenizer_cross_field_validation_remains_authoritative(tmp_path: Path) -> None:
    store = _store(_server_settings(tmp_path), tmp_path / "runtime.json")

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

    store = _store(_server_settings(tmp_path), path)
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

    store = _store(_server_settings(tmp_path), path)

    assert store.get_state().warning is not None
    assert path.read_text(encoding="utf-8") == "{not-json"

###############################################################################
def test_successful_save_replaces_corrupt_stored_document(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    path.write_text("{not-json", encoding="utf-8")

    store = _store(_server_settings(tmp_path), path)
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
    store = _store(_server_settings(tmp_path), path)
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
    store = _store(_server_settings(tmp_path), path)
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
    store = _store(_server_settings(tmp_path), path)
    store.apply_patch({"datasets": {"histogram_bins": 30}}, expected_revision=0)

    state = store.reset(expected_revision=1)

    assert state.settings.datasets.histogram_bins == 20
    assert state.overridden_keys == ()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "dashboard_preferences": {
            "baseline_by_report_id": {},
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
def test_reset_all_keeps_revision_of_an_unchanged_empty_document(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    payload = {
        "schema_version": 1,
        "revision": 3,
        "overrides": {},
        "dashboard_revision": 0,
        "dashboard_preferences": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    store = _store(_server_settings(tmp_path), path)

    state = store.reset(expected_revision=3)

    assert state.revision == 3
    assert state.overridden_keys == ()
    assert json.loads(path.read_text(encoding="utf-8")) == payload

###############################################################################
def test_persistence_failure_rolls_back_state_and_stored_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.json"
    store = _store(_server_settings(tmp_path), path)
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
    first = _store(defaults, path)
    first.apply_patch(
        {"datasets": {"histogram_bins": defaults.datasets.histogram_bins + 1}},
        expected_revision=0,
    )

    sparse_file = json.loads(path.read_text(encoding="utf-8"))
    assert sparse_file["overrides"] == {
        "datasets": {"histogram_bins": defaults.datasets.histogram_bins + 1}
    }

    restarted = _store(defaults, path)
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
    store = _store(defaults, path)
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
    store = _store(_server_settings(tmp_path), tmp_path / "runtime.json")
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
    store = _store(_server_settings(tmp_path), path)

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
