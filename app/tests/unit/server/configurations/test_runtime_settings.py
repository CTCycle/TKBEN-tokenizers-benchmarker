from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

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


###############################################################################
def _server_settings(tmp_path: Path) -> ServerSettings:
    resources = tmp_path / "resources"
    return ServerSettings(
        database=DatabaseSettings(
            embedded_database=True,
            sqlite_path=resources / "database.db",
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
            resources=resources,
            sources=resources / "sources",
            datasets=resources / "sources/datasets",
            tokenizers=resources / "sources/tokenizers",
            logs=resources / "logs",
            templates=resources / "templates",
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
            hf_keys_encryption_material_file=resources / "hf-key-material.json",
        ),
        datasets=DatasetSettings(),
        tokenizers=TokenizerSettings(),
        benchmarks=BenchmarkSettings(),
        jobs=JobsSettings(),
    )


###############################################################################
def test_default_only_startup_and_missing_runtime_file(tmp_path: Path) -> None:
    defaults = _server_settings(tmp_path)
    path = tmp_path / "runtime-settings.json"

    store = RuntimeSettingsStore(defaults, path)
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
    path = tmp_path / "runtime-settings.json"
    store = RuntimeSettingsStore(defaults, path)

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
        "overrides": {
            "datasets": {"histogram_bins": 30},
            "jobs": {"polling_interval": 2.0},
        },
        "revision": 2,
        "schema_version": 1,
    }
    assert "network" not in path.read_text(encoding="utf-8")


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
    store = RuntimeSettingsStore(_server_settings(tmp_path), tmp_path / "runtime.json")

    with pytest.raises(RuntimeSettingsValidationError):
        store.apply_patch(patch, expected_revision=0)

    assert store.get_state().revision == 0
    assert not store.path.exists()


###############################################################################
def test_tokenizer_cross_field_validation_remains_authoritative(tmp_path: Path) -> None:
    store = RuntimeSettingsStore(_server_settings(tmp_path), tmp_path / "runtime.json")

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
        {"schema_version": 2, "revision": 4, "overrides": {}},
        {"schema_version": 1, "revision": 4, "overrides": {"network": {}}},
        {"schema_version": 1, "revision": 4, "overrides": {"jobs": {"polling_interval": 0.1}}},
    ],
)
def test_invalid_schema_or_override_file_falls_back_with_warning(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    path = tmp_path / "runtime-settings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    original = path.read_bytes()

    store = RuntimeSettingsStore(_server_settings(tmp_path), path)
    state = store.get_state()

    assert state.settings.datasets.histogram_bins == 20
    assert state.revision == 0
    assert state.overridden_keys == ()
    assert state.warning == runtime_module.RUNTIME_SETTINGS_WARNING
    assert path.read_bytes() == original


###############################################################################
def test_malformed_runtime_json_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "runtime-settings.json"
    path.write_text("{not-json", encoding="utf-8")

    store = RuntimeSettingsStore(_server_settings(tmp_path), path)

    assert store.get_state().warning is not None
    assert path.read_text(encoding="utf-8") == "{not-json"


###############################################################################
def test_successful_save_replaces_corrupt_runtime_file(tmp_path: Path) -> None:
    path = tmp_path / "runtime-settings.json"
    path.write_text("{not-json", encoding="utf-8")

    store = RuntimeSettingsStore(_server_settings(tmp_path), path)
    state = store.apply_patch(
        {"datasets": {"histogram_bins": 20}},
        expected_revision=0,
    )

    assert state.warning is None
    assert state.revision == 1
    assert state.overridden_keys == ()
    assert not path.exists()

###############################################################################
def test_revision_conflict_does_not_write(tmp_path: Path) -> None:
    path = tmp_path / "runtime-settings.json"
    store = RuntimeSettingsStore(_server_settings(tmp_path), path)
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
    path = tmp_path / "runtime-settings.json"
    store = RuntimeSettingsStore(_server_settings(tmp_path), path)
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
def test_reset_all_deletes_empty_runtime_file(tmp_path: Path) -> None:
    path = tmp_path / "runtime-settings.json"
    store = RuntimeSettingsStore(_server_settings(tmp_path), path)
    store.apply_patch({"datasets": {"histogram_bins": 30}}, expected_revision=0)

    state = store.reset(expected_revision=1)

    assert state.settings.datasets.histogram_bins == 20
    assert state.overridden_keys == ()
    assert not path.exists()


###############################################################################
def test_reset_all_removes_a_valid_empty_runtime_file(tmp_path: Path) -> None:
    path = tmp_path / "runtime-settings.json"
    path.write_text(
        json.dumps({"schema_version": 1, "revision": 3, "overrides": {}}),
        encoding="utf-8",
    )
    store = RuntimeSettingsStore(_server_settings(tmp_path), path)

    state = store.reset(expected_revision=3)

    assert state.revision == 4
    assert state.overridden_keys == ()
    assert not path.exists()


###############################################################################
def test_atomic_write_and_persistence_failure_roll_back_state_and_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "runtime-settings.json"
    store = RuntimeSettingsStore(_server_settings(tmp_path), path)
    store.apply_patch({"datasets": {"histogram_bins": 30}}, expected_revision=0)
    original = path.read_bytes()

    def fail_replace(_source: str, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(runtime_module.os, "replace", fail_replace)
    with pytest.raises(RuntimeSettingsPersistenceError):
        store.apply_patch(
            {"datasets": {"histogram_bins": 31}},
            expected_revision=1,
        )

    assert path.read_bytes() == original
    assert store.get_state().settings.datasets.histogram_bins == 30
    assert store.get_state().revision == 1
    assert not list(tmp_path.glob(".runtime-settings.json.*.tmp"))


###############################################################################
def test_process_restart_and_reload_load_persisted_overrides(tmp_path: Path) -> None:
    path = tmp_path / "runtime-settings.json"
    defaults = _server_settings(tmp_path)
    first = RuntimeSettingsStore(defaults, path)
    first.apply_patch({"jobs": {"polling_interval": 2.5}}, expected_revision=0)

    restarted = RuntimeSettingsStore(defaults, path)
    assert restarted.get_state().settings.jobs.polling_interval == 2.5
    assert restarted.get_state().revision == 1
    assert restarted.get_state().warning is None


###############################################################################
def test_runtime_override_models_are_frozen(tmp_path: Path) -> None:
    store = RuntimeSettingsStore(_server_settings(tmp_path), tmp_path / "runtime.json")
    state = store.apply_patch(
        {"datasets": {"histogram_bins": 30}},
        expected_revision=0,
    )

    with pytest.raises(ValidationError):
        state.settings.datasets.histogram_bins = 20
