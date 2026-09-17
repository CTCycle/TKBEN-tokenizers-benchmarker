from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.configurations import startup
from server.configurations.runtime import RuntimeSettingsPersistenceError, RuntimeSettingsStore


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
        json={"expected_revision": 0, "datasets": {"histogram_bins": 30, "max_upload_bytes": 40 * 1024 * 1024}},
    ).json()

    reset_one = settings_client.post(
        "/api/settings/reset",
        json={"expected_revision": patched["revision"], "keys": ["datasets.histogram_bins"]},
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
    assert reset_all.json()["settings"]["datasets"]["max_upload_bytes"] == 25 * 1024 * 1024


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
    assert settings_client.get("/api/settings").json()["settings"]["jobs"]["polling_interval"] == 2.0


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
    assert json.loads(path.read_text(encoding="utf-8"))["overrides"]["datasets"]["histogram_bins"] == 30
