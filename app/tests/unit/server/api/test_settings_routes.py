from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import pytest
from pydantic import BaseModel
from fastapi.testclient import TestClient

from server.app import app
from server.configurations import startup
from server.configurations.runtime import (
    RuntimeSettingsPersistenceError,
    RuntimeSettingsStore,
)
from server.configurations.settings import ServerSettings
from server.contracts.settings import RuntimeSettingsPatchRequest


def _runtime_setting_fields():
    """Derive editable fields and constraints from the public request schema."""
    for group, group_field in RuntimeSettingsPatchRequest.model_fields.items():
        if group == "expected_revision":
            continue
        group_models = [
            option
            for option in get_args(group_field.annotation)
            if isinstance(option, type) and issubclass(option, BaseModel)
        ]
        assert len(group_models) == 1
        settings_model = ServerSettings.model_fields[group].annotation
        for name, request_field in group_models[0].model_fields.items():
            yield (
                group,
                name,
                request_field,
                settings_model.model_fields[name],
            )


def _constraint(field, name: str):
    return next(
        (
            getattr(metadata, name)
            for metadata in field.metadata
            if hasattr(metadata, name)
        ),
        None,
    )


def _runtime_setting_spec(group: str, name: str):
    return next(
        spec
        for spec in _runtime_setting_fields()
        if spec[0] == group and spec[1] == name
    )


def _alternate_schema_value(group: str, name: str):
    _group, _name, request_field, settings_field = _runtime_setting_spec(group, name)
    current = settings_field.default
    step = 1 if isinstance(current, int) else 0.25
    candidate = current + step
    maximum = _constraint(request_field, "le")
    if maximum is not None and candidate > maximum:
        candidate = current - step
    minimum = _constraint(request_field, "ge")
    assert minimum is not None and candidate >= minimum
    assert maximum is None or candidate <= maximum
    return candidate


def _boundary_patch(group: str, name: str, value: object) -> dict[str, object]:
    values = {name: value}
    if group == "tokenizers":
        defaults = startup.get_server_settings().tokenizers
        if name == "default_discovery_limit":
            values["max_discovery_limit"] = max(
                defaults.max_discovery_limit,
                int(value),
            )
            values["max_discovery_candidates"] = max(
                defaults.max_discovery_candidates,
                int(values["max_discovery_limit"]),
            )
        elif name == "max_discovery_limit":
            values["default_discovery_limit"] = min(
                defaults.default_discovery_limit,
                int(value),
            )
            values["max_discovery_candidates"] = max(
                defaults.max_discovery_candidates,
                int(value),
            )
        elif name == "max_discovery_candidates":
            compatible_limit = min(defaults.max_discovery_limit, int(value))
            values["max_discovery_limit"] = compatible_limit
            values["default_discovery_limit"] = min(
                defaults.default_discovery_limit,
                compatible_limit,
            )
    return {group: values}


def _invalid_setting_cases():
    cases = []
    for group, name, request_field, settings_field in _runtime_setting_fields():
        default = settings_field.default
        is_integer = isinstance(default, int) and not isinstance(default, bool)
        wrong_types = [None, "", "1", [], True]
        for value in wrong_types:
            cases.append((group, name, value, "type/null"))

        minimum = _constraint(request_field, "ge")
        maximum = _constraint(request_field, "le")
        if minimum is not None:
            delta = 1 if is_integer else 0.01
            cases.append((group, name, minimum - delta, "below-minimum"))
        if maximum is not None:
            delta = 1 if is_integer else 0.01
            cases.append((group, name, maximum + delta, "above-maximum"))
        if is_integer:
            cases.append((group, name, default + 0.5, "fractional-integer"))
    return cases


def _numeric_boundary_cases():
    cases = []
    for group, name, request_field, settings_field in _runtime_setting_fields():
        for boundary_name, value in (
            ("minimum", _constraint(request_field, "ge")),
            ("maximum", _constraint(request_field, "le")),
        ):
            if value is not None:
                cases.append((group, name, value, boundary_name))
    return cases


def _companion_override(group: str) -> tuple[str, str, object, object]:
    companion_group, companion_name = (
        ("datasets", "histogram_bins")
        if group == "jobs"
        else ("jobs", "polling_interval")
    )
    model = ServerSettings.model_fields[companion_group].annotation
    field = model.model_fields[companion_name]
    original = field.default
    return companion_group, companion_name, original + 1, original + 2


###############################################################################
@pytest.fixture
def settings_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    defaults = startup.get_server_settings()
    store = RuntimeSettingsStore(defaults, tmp_path / "runtime-settings.json")
    monkeypatch.setattr(startup, "_runtime_settings_store", store)
    monkeypatch.setattr(startup, "_default_settings", defaults)
    return TestClient(app)


###############################################################################
def test_get_settings_returns_only_typed_runtime_fields(
    settings_client: TestClient,
) -> None:
    response = settings_client.get("/api/settings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["revision"] == 0
    assert set(payload["settings"]) == {"tokenizers", "datasets", "benchmarks", "jobs"}
    assert set(payload["settings"]["tokenizers"]) == {
        "default_discovery_limit",
        "max_discovery_limit",
        "max_discovery_candidates",
        "metadata_candidate_multiplier",
        "max_upload_bytes",
    }
    assert set(payload["settings"]["datasets"]) == {
        "histogram_bins",
        "streaming_batch_size",
        "max_upload_bytes",
        "download_timeout_seconds",
        "download_retry_attempts",
        "download_retry_backoff_seconds",
    }
    assert set(payload["settings"]["benchmarks"]) == {
        "default_max_documents",
        "default_batch_size",
        "default_parallelism",
        "streaming_batch_size",
    }
    assert set(payload["settings"]["jobs"]) == {"polling_interval"}
    assert payload["overridden_keys"] == []
    assert payload["warning"] is None
    assert "DATABASE_PASSWORD" not in response.text
    assert "fastapi_host" not in response.text
    assert "paths" not in response.text


def test_public_runtime_matrix_matches_typed_backend_schema() -> None:
    fields = list(_runtime_setting_fields())
    api_keys = {f"{group}.{name}" for group, name, _, _ in fields}

    assert len(api_keys) == 16
    assert api_keys == set(RuntimeSettingsStore._EDITABLE_KEYS)
    for _group, _name, request_field, settings_field in fields:
        assert _constraint(request_field, "ge") == _constraint(settings_field, "ge")
        assert _constraint(request_field, "le") == _constraint(settings_field, "le")


@pytest.mark.parametrize(
    ("group", "name", "value", "boundary"),
    _numeric_boundary_cases(),
)
def test_patch_accepts_every_runtime_numeric_boundary(
    settings_client: TestClient,
    group: str,
    name: str,
    value: object,
    boundary: str,
) -> None:
    values = _boundary_patch(group, name, value)[group]
    response = settings_client.patch(
        "/api/settings",
        json={"expected_revision": 0, group: values},
    )

    assert response.status_code == 200, f"{group}.{name} {boundary}: {response.text}"
    assert response.json()["settings"][group][name] == value


@pytest.mark.parametrize(("group", "name", "value", "case"), _invalid_setting_cases())
def test_rejected_runtime_setting_update_preserves_persisted_snapshot(
    settings_client: TestClient,
    group: str,
    name: str,
    value: object,
    case: str,
) -> None:
    companion_group, companion_name, initial_companion, replacement_companion = (
        _companion_override(group)
    )
    initial = settings_client.patch(
        "/api/settings",
        json={
            "expected_revision": 0,
            companion_group: {companion_name: initial_companion},
        },
    )
    assert initial.status_code == 200, initial.text
    before = settings_client.get("/api/settings").json()
    store = startup.get_runtime_settings_store()
    persisted_before = store.path.read_bytes()

    payload = {
        "expected_revision": before["revision"],
        group: {name: value},
        companion_group: {companion_name: replacement_companion},
    }
    rejected = settings_client.patch("/api/settings", json=payload)

    assert rejected.status_code == 422, f"{group}.{name} {case}: {rejected.text}"
    assert settings_client.get("/api/settings").json() == before
    assert store.path.read_bytes() == persisted_before


def test_invalid_tokenizer_relationships_do_not_partially_persist(
    settings_client: TestClient,
) -> None:
    minimum = _constraint(
        next(
            field
            for group, name, field, _settings_field in _runtime_setting_fields()
            if group == "tokenizers" and name == "default_discovery_limit"
        ),
        "ge",
    )
    assert minimum is not None
    cases = (
        {
            "default_discovery_limit": minimum + 1,
            "max_discovery_limit": minimum,
        },
        {
            "default_discovery_limit": minimum,
            "max_discovery_limit": minimum + 1,
            "max_discovery_candidates": minimum,
        },
        {
            "default_discovery_limit": minimum + 2,
            "max_discovery_limit": minimum + 1,
            "max_discovery_candidates": minimum,
        },
    )
    store = startup.get_runtime_settings_store()
    histogram_default = (
        ServerSettings.model_fields["datasets"]
        .annotation.model_fields["histogram_bins"]
        .default
    )
    baseline = settings_client.patch(
        "/api/settings",
        json={
            "expected_revision": 0,
            "datasets": {"histogram_bins": histogram_default + 1},
        },
    )
    assert baseline.status_code == 200, baseline.text
    before = settings_client.get("/api/settings").json()
    persisted_before = store.path.read_bytes()

    for tokenizer_settings in cases:
        rejected = settings_client.patch(
            "/api/settings",
            json={
                "expected_revision": before["revision"],
                "tokenizers": tokenizer_settings,
                "datasets": {"histogram_bins": histogram_default + 2},
            },
        )
        assert rejected.status_code == 422, rejected.text
        assert settings_client.get("/api/settings").json() == before
        assert store.path.read_bytes() == persisted_before


def test_new_workflows_use_saved_runtime_settings(
    settings_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    from types import SimpleNamespace

    from server.services import datasets as datasets_module
    from server.services import tokenizers as tokenizers_module
    from server.services.benchmarks import BenchmarkService
    from server.services.dataset_statistics import LengthStatistics
    from server.services.datasets import DatasetService
    from server.services.managed_jobs import ManagedJobService, ManagedJobSpec

    def minimum(group: str, name: str):
        _group, _name, field, _settings_field = _runtime_setting_spec(group, name)
        result = _constraint(field, "ge")
        assert result is not None
        return result

    target_histogram_bins = minimum("datasets", "histogram_bins") + 1
    target_dataset_batch = _alternate_schema_value("datasets", "streaming_batch_size")
    target_dataset_upload = minimum("datasets", "max_upload_bytes")
    target_timeout = _alternate_schema_value("datasets", "download_timeout_seconds")
    target_retry_attempts = minimum("datasets", "download_retry_attempts") + 1
    target_retry_backoff = minimum("datasets", "download_retry_backoff_seconds") + 0.25
    target_discovery_default = minimum("tokenizers", "default_discovery_limit")
    target_discovery_max = target_discovery_default + 1
    target_candidate_cap = target_discovery_max + 1
    target_metadata_multiplier = (
        minimum("tokenizers", "metadata_candidate_multiplier") + 1
    )
    target_tokenizer_upload = minimum("tokenizers", "max_upload_bytes")
    target_benchmark_batch = _alternate_schema_value(
        "benchmarks", "streaming_batch_size"
    )
    target_poll_interval = _alternate_schema_value("jobs", "polling_interval")
    assert target_discovery_max <= _constraint(
        _runtime_setting_spec("tokenizers", "max_discovery_limit")[2], "le"
    )

    saved = settings_client.patch(
        "/api/settings",
        json={
            "expected_revision": 0,
            "datasets": {
                "histogram_bins": target_histogram_bins,
                "streaming_batch_size": target_dataset_batch,
                "max_upload_bytes": target_dataset_upload,
                "download_timeout_seconds": target_timeout,
                "download_retry_attempts": target_retry_attempts,
                "download_retry_backoff_seconds": target_retry_backoff,
            },
            "tokenizers": {
                "default_discovery_limit": target_discovery_default,
                "max_discovery_limit": target_discovery_max,
                "max_discovery_candidates": target_candidate_cap,
                "metadata_candidate_multiplier": target_metadata_multiplier,
                "max_upload_bytes": target_tokenizer_upload,
            },
            "benchmarks": {"streaming_batch_size": target_benchmark_batch},
            "jobs": {"polling_interval": target_poll_interval},
        },
    )
    assert saved.status_code == 200, saved.text

    dataset_service = DatasetService()
    lengths = list(range(target_histogram_bins))
    stats = LengthStatistics()
    for length in lengths:
        stats.update(length)
    histogram = dataset_service.histogram_from_counts(
        stats,
        {length: 1 for length in lengths},
    )
    assert len(histogram["bins"]) == target_histogram_bins

    observed_dataset_batches: list[int] = []

    def fake_database_lengths(_dataset_name: str, batch_size: int):
        observed_dataset_batches.append(batch_size)
        return iter([1, 2])

    monkeypatch.setattr(
        dataset_service,
        "_iterate_database_lengths",
        fake_database_lengths,
    )
    assert list(dataset_service.database_length_stream("new-dataset")()) == [1, 2]
    assert observed_dataset_batches == [target_dataset_batch]
    assert dataset_service.retry_delay_seconds(1) == target_retry_backoff
    assert dataset_service.should_retry_download(
        "network_or_transient",
        target_retry_attempts - 1,
        target_retry_attempts,
    )
    assert not dataset_service.should_retry_download(
        "network_or_transient",
        target_retry_attempts,
        target_retry_attempts,
    )

    joined_timeouts: list[float | None] = []

    class CapturingThread(threading.Thread):
        def join(self, timeout: float | None = None) -> None:
            joined_timeouts.append(timeout)
            super().join(timeout=timeout)

    monkeypatch.setattr(
        datasets_module,
        "threading",
        SimpleNamespace(Event=threading.Event, Thread=CapturingThread),
    )

    def finish_download(result_holder, _error_holder, *_args) -> None:
        result_holder["dataset"] = "loaded"

    monkeypatch.setattr(dataset_service, "_load_dataset_worker", finish_download)
    monkeypatch.setattr(
        dataset_service,
        "_limit_loaded_dataset",
        lambda dataset, _maximum: dataset,
    )
    assert (
        dataset_service.load_dataset_with_progress(
            "dataset-id",
            None,
            "cache",
            None,
            None,
        )
        == "loaded"
    )
    assert joined_timeouts == [target_timeout]

    captured_candidate_limits: list[int] = []

    class FakeHfApi:
        def __init__(self, token: str | None) -> None:
            del token

        def list_models(self, **kwargs):
            captured_candidate_limits.append(kwargs["limit"])
            return []

    monkeypatch.setattr(tokenizers_module, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        tokenizers_module.HFAccessKeyService,
        "get_active_key",
        lambda _service: None,
    )
    default_discovery = settings_client.get("/api/tokenizers/discover")
    assert default_discovery.status_code == 200, default_discovery.text
    assert captured_candidate_limits[-1] == min(
        target_candidate_cap,
        target_discovery_default * target_metadata_multiplier,
    )
    maximum_discovery = settings_client.get(
        f"/api/tokenizers/discover?limit={target_discovery_max}"
    )
    assert maximum_discovery.status_code == 200, maximum_discovery.text
    assert captured_candidate_limits[-1] == min(
        target_candidate_cap,
        target_discovery_max * target_metadata_multiplier,
    )
    assert (
        settings_client.get(
            f"/api/tokenizers/discover?limit={target_discovery_max + 1}"
        ).status_code
        == 422
    )

    dataset_upload = settings_client.post(
        "/api/datasets/upload",
        files={"file": ("too-large.csv", b"ab", "text/csv")},
    )
    tokenizer_upload = settings_client.post(
        "/api/tokenizers/upload",
        files={"file": ("too-large.json", b"{}", "application/json")},
    )
    assert dataset_upload.status_code == 413
    assert tokenizer_upload.status_code == 413

    benchmark_service = BenchmarkService()
    observed_benchmark_batches: list[int] = []

    def fake_benchmark_rows(*, dataset_name: str, batch_size: int):
        assert dataset_name == "new-dataset"
        observed_benchmark_batches.append(batch_size)
        return iter([(1, "first document")])

    monkeypatch.setattr(
        benchmark_service.dataset_repository,
        "iterate_dataset_rows_for_benchmarks",
        fake_benchmark_rows,
    )
    assert list(benchmark_service.stream_dataset_rows_from_database("new-dataset")) == [
        (1, "first document")
    ]
    assert observed_benchmark_batches == [target_benchmark_batch]

    class NewJobManager:
        def is_job_running(self, _job_type: str | None = None) -> bool:
            return False

        def start_job(self, job_type, runner, args=(), kwargs=None):
            del runner, args, kwargs
            self.job_type = job_type
            return "settings-workflow-job"

        def get_job_status(self, _job_id: str):
            return {"job_type": self.job_type, "status": "pending"}

    job = ManagedJobService().start(
        NewJobManager(),
        ManagedJobSpec(
            job_type="settings-workflow",
            runner=lambda **_kwargs: {},
            kwargs={},
            conflict_detail="already running",
            initialization_detail="failed to initialize",
            message="started",
        ),
    )
    assert job.poll_interval == target_poll_interval


###############################################################################
def test_patch_updates_multiple_fields_and_keeps_partial_overrides(
    settings_client: TestClient,
) -> None:
    response = settings_client.patch(
        "/api/settings",
        json={
            "expected_revision": 0,
            "datasets": {"histogram_bins": 30},
            "benchmarks": {"default_max_documents": 2500},
            "jobs": {"polling_interval": 2.0},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["revision"] == 1
    assert payload["settings"]["datasets"]["histogram_bins"] == 30
    assert payload["settings"]["datasets"]["streaming_batch_size"] == 10_000
    assert payload["settings"]["benchmarks"]["default_max_documents"] == 2500
    assert payload["settings"]["jobs"]["polling_interval"] == 2.0
    assert payload["overridden_keys"] == [
        "datasets.histogram_bins",
        "benchmarks.default_max_documents",
        "jobs.polling_interval",
    ]

    follow_up = settings_client.patch(
        "/api/settings",
        json={
            "expected_revision": 1,
            "datasets": {"download_retry_attempts": 5},
        },
    )
    assert follow_up.status_code == 200
    assert follow_up.json()["settings"]["datasets"]["histogram_bins"] == 30
    assert follow_up.json()["settings"]["datasets"]["download_retry_attempts"] == 5
    assert follow_up.json()["settings"]["benchmarks"]["default_max_documents"] == 2500


###############################################################################
def test_reset_one_and_reset_all_return_defaults(
    settings_client: TestClient,
) -> None:
    patched = settings_client.patch(
        "/api/settings",
        json={
            "expected_revision": 0,
            "datasets": {"histogram_bins": 30, "max_upload_bytes": 40 * 1024 * 1024},
        },
    ).json()

    reset_one = settings_client.post(
        "/api/settings/reset",
        json={
            "expected_revision": patched["revision"],
            "keys": ["datasets.histogram_bins"],
        },
    )
    assert reset_one.status_code == 200
    assert reset_one.json()["settings"]["datasets"]["histogram_bins"] == 20
    assert reset_one.json()["overridden_keys"] == ["datasets.max_upload_bytes"]

    reset_all = settings_client.post(
        "/api/settings/reset",
        json={"expected_revision": reset_one.json()["revision"], "all": True},
    )
    assert reset_all.status_code == 200
    assert reset_all.json()["overridden_keys"] == []
    assert (
        reset_all.json()["settings"]["datasets"]["max_upload_bytes"] == 25 * 1024 * 1024
    )


###############################################################################
@pytest.mark.parametrize(
    "payload",
    [
        {"expected_revision": 0, "datasets": {"histogram_bins": 4}},
        {"expected_revision": 0, "datasets": {"histogram_bins": "30"}},
        {"expected_revision": 0, "unknown": {"histogram_bins": 30}},
        {"expected_revision": 0, "network": {"fastapi_port": 9000}},
        {"expected_revision": 0, "tokenizers": {"max_discovery_limit": 10}},
        {"expected_revision": 0, "benchmarks": {"default_max_documents": 0}},
        {"expected_revision": 0, "benchmarks": {"default_batch_size": 4097}},
        {"expected_revision": 0, "benchmarks": {"default_parallelism": 129}},
        {"expected_revision": 0, "datasets": None},
    ],
)
def test_invalid_unknown_environment_and_incompatible_fields_fail(
    settings_client: TestClient,
    payload: dict[str, object],
) -> None:
    response = settings_client.patch("/api/settings", json=payload)

    assert response.status_code == 422
    assert settings_client.get("/api/settings").json()["revision"] == 0


###############################################################################
def test_stale_revision_returns_conflict_without_overwriting(
    settings_client: TestClient,
) -> None:
    first = settings_client.patch(
        "/api/settings",
        json={"expected_revision": 0, "jobs": {"polling_interval": 2.0}},
    )
    assert first.status_code == 200

    stale = settings_client.patch(
        "/api/settings",
        json={"expected_revision": 0, "jobs": {"polling_interval": 3.0}},
    )
    assert stale.status_code == 409
    assert (
        settings_client.get("/api/settings").json()["settings"]["jobs"][
            "polling_interval"
        ]
        == 2.0
    )


###############################################################################
def test_persistence_error_keeps_previous_authoritative_state(
    settings_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = startup.get_runtime_settings_store()
    original = store.get_state()

    def fail_persist(_overrides, _revision) -> None:
        raise RuntimeSettingsPersistenceError("persistence failure")

    monkeypatch.setattr(store, "_persist", fail_persist)
    response = settings_client.patch(
        "/api/settings",
        json={"expected_revision": 0, "datasets": {"histogram_bins": 30}},
    )

    assert response.status_code == 500
    current = settings_client.get("/api/settings").json()
    assert current["revision"] == original.revision
    assert current["settings"]["datasets"]["histogram_bins"] == 20


###############################################################################
def test_corrupt_file_warning_is_sanitized_and_cleared_by_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = startup.get_server_settings()
    path = tmp_path / "runtime-settings.json"
    path.write_text("{broken", encoding="utf-8")
    store = RuntimeSettingsStore(defaults, path)
    monkeypatch.setattr(startup, "_runtime_settings_store", store)
    monkeypatch.setattr(startup, "_default_settings", defaults)
    client = TestClient(app)

    warning = client.get("/api/settings")
    assert warning.status_code == 200
    assert warning.json()["warning"]
    assert "runtime-settings.json" not in warning.json()["warning"]

    saved = client.patch(
        "/api/settings",
        json={"expected_revision": 0, "datasets": {"histogram_bins": 30}},
    )
    assert saved.status_code == 200
    assert saved.json()["warning"] is None
    assert (
        json.loads(path.read_text(encoding="utf-8"))["overrides"]["datasets"][
            "histogram_bins"
        ]
        == 30
    )
