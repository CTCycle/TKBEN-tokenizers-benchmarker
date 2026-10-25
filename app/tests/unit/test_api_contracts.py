# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest
from server.app import app
from types import SimpleNamespace
from time import time
from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
from typing import cast
from typing import get_args
from pydantic import BaseModel
from fastapi import FastAPI
from server.api.datasets import router as datasets_router
from server.api.settings import router as settings_router
from server.api.tokenizers import router as tokenizers_router
from server.configurations import startup
from server.configurations.runtime import RuntimeSettingsPersistenceError
from server.configurations.runtime import RuntimeSettingsStore
from server.configurations.settings import ServerSettings
from server.contracts.settings import RuntimeSettingsPatchRequest

###############################################################################


###############################################################################
class DummyJobManager:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.last_job_type = ""

    # -------------------------------------------------------------------------
    def is_job_running(self, job_type: str | None = None) -> bool:
        return False

    # -------------------------------------------------------------------------
    def start_job(self, job_type, runner, args=(), kwargs=None):
        del runner, args, kwargs
        self.last_job_type = str(job_type)
        return "job-bench"

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str):
        del job_id
        return {"job_type": self.last_job_type, "status": "pending"}

###############################################################################
def test_benchmark_run_route_returns_202(monkeypatch) -> None:
    manager = DummyJobManager()
    monkeypatch.setattr(app.state, "job_manager", manager)

    from server.services.benchmarks import BenchmarkService

    monkeypatch.setattr(
        BenchmarkService,
        "prepare_run",
        lambda self, payload: payload.model_dump(),
    )

    client = TestClient(app)
    resp = client.post(
        "/api/benchmarks/run",
        json={
            "tokenizers": ["bert-base-uncased"],
            "dataset_name": "custom/sample",
            "config": {
                "max_documents": 10,
                "add_special_tokens": True,
                "padding": True,
                "truncation": True,
                "max_length": 64,
                "store_per_document_stats": False,
                "per_document_sample_size": 32,
            },
        },
    )

    assert resp.status_code == 202
    assert resp.json()["job_id"] == "job-bench"

###############################################################################
def test_benchmark_run_rejects_removed_custom_tokenizer_field() -> None:
    response = TestClient(app).post(
        "/api/benchmarks/run",
        json={
            "tokenizers": ["CUSTOM_demo"],
            "dataset_name": "custom/sample",
            "custom_tokenizer_name": "CUSTOM_demo",
        },
    )

    assert response.status_code == 422
    assert "custom_tokenizer_name" in response.text

###############################################################################
def test_benchmark_list_and_by_id(monkeypatch) -> None:
    from server.contracts.benchmarks import BenchmarkReportListResponse
    from server.services.benchmark_reports import BenchmarkReportService

    captured = {}

    def fake_list(self, query):
        del self
        captured.update(query.model_dump())
        return BenchmarkReportListResponse.model_validate(
            {
                "reports": [
                    {
                        "report_id": 1,
                        "report_version": 5,
                        "created_at": "2026-01-01T00:00:00Z",
                        "run_name": "run",
                        "dataset_name": "custom/sample",
                        "documents_processed": 2,
                        "tokenizers_count": 1,
                        "tokenizers_processed": ["bert-base-uncased"],
                        "selected_metric_keys": ["global.tokenization_speed_tps"],
                    }
                ],
                "total": 1,
                "offset": 0,
                "limit": 25,
            }
        )

    monkeypatch.setattr(
        BenchmarkReportService,
        "list_benchmark_reports",
        fake_list,
    )
    monkeypatch.setattr(
        BenchmarkReportService,
        "load_benchmark_report_by_id",
        lambda self, report_id: {
            "status": "success",
            "schema_version": 3,
            "methodology_version": "semantic_honesty",
            "report_id": report_id,
            "report_version": 5,
            "created_at": "2026-01-01T00:00:00Z",
            "dataset_name": "custom/sample",
            "documents_processed": 2,
            "tokenizers_count": 1,
            "tokenizers_processed": ["bert-base-uncased"],
            "selected_metric_keys": [],
            "config": {
                "max_documents": 0,
                "warmup_trials": 2,
                "timed_trials": 8,
                "batch_size": 16,
                "seed": 42,
                "parallelism": 1,
                "add_special_tokens": False,
                "padding": False,
                "truncation": False,
                "max_length": None,
                "store_per_document_stats": True,
                "per_document_sample_size": 500,
            },
            "hardware_profile": {
                "runtime": "",
                "os": "",
                "cpu_model": None,
                "cpu_logical_cores": None,
                "memory_total_mb": None,
            },
            "trial_summary": {"warmup_trials": 2, "timed_trials": 8},
            "tokenizer_results": [],
            "dashboard": {"widgets": []},
            "per_document_stats": [],
            "runtime_metadata": {},
            "raw_observations": {},
        },
    )

    client = TestClient(app)

    listed = client.get(
        "/api/benchmarks/reports?search= run &sort=oldest&offset=5&limit=10"
    )
    assert listed.status_code == 200
    assert listed.json()["reports"][0]["report_id"] == 1
    assert listed.json()["total"] == 1
    assert captured == {"search": "run", "sort": "oldest", "offset": 5, "limit": 10}

    by_id = client.get("/api/benchmarks/reports/1")
    assert by_id.status_code == 200
    assert by_id.json()["report_id"] == 1

###############################################################################
def test_benchmark_report_delete_route_returns_204_or_404(monkeypatch) -> None:
    from server.services.benchmark_reports import BenchmarkReportService

    monkeypatch.setattr(
        BenchmarkReportService,
        "delete_benchmark_report",
        lambda self, report_id: report_id == 4,
    )
    client = TestClient(app)
    assert client.delete("/api/benchmarks/reports/4").status_code == 204
    missing = client.delete("/api/benchmarks/reports/9")
    assert missing.status_code == 404

###############################################################################
def test_benchmark_report_tags_route_normalizes_and_returns_tags(monkeypatch) -> None:
    from server.contracts.benchmarks import BenchmarkReportTagsResponse
    from server.services.benchmark_reports import BenchmarkReportService

    captured = {}

    def fake_update(self, report_id, tags):
        del self
        captured["report_id"] = report_id
        captured["tags"] = tags
        return BenchmarkReportTagsResponse(report_id=report_id, tags=tags)

    monkeypatch.setattr(
        BenchmarkReportService,
        "update_benchmark_report_tags",
        fake_update,
    )
    response = TestClient(app).patch(
        "/api/benchmarks/reports/7/tags",
        json={"tags": [" Production ", "cpu", "production", ""]},
    )

    assert response.status_code == 200
    assert captured == {"report_id": 7, "tags": ["Production", "cpu"]}
    assert response.json() == {"report_id": 7, "tags": ["Production", "cpu"]}

###############################################################################
def test_benchmark_report_tags_route_returns_404_for_missing_report(monkeypatch) -> None:
    from server.services.benchmark_reports import BenchmarkReportService

    monkeypatch.setattr(
        BenchmarkReportService,
        "update_benchmark_report_tags",
        lambda self, report_id, tags: None,
    )
    response = TestClient(app).patch(
        "/api/benchmarks/reports/404/tags", json={"tags": []}
    )
    assert response.status_code == 404

###############################################################################
@pytest.mark.parametrize(
    "tags",
    [
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
        ["x" * 33],
        ["valid\ninvalid"],
    ],
)
def test_benchmark_report_tags_route_rejects_invalid_tags(tags) -> None:
    response = TestClient(app).patch(
        "/api/benchmarks/reports/7/tags", json={"tags": tags}
    )
    assert response.status_code == 422

###############################################################################
def test_benchmark_by_id_accepts_cancelled_contract(monkeypatch) -> None:
    from server.services.benchmark_reports import BenchmarkReportService

    monkeypatch.setattr(
        BenchmarkReportService,
        "load_benchmark_report_by_id",
        lambda self, report_id: {
            "status": "cancelled",
            "schema_version": 3,
            "methodology_version": "semantic_honesty",
            "report_id": report_id,
            "report_version": 5,
            "created_at": "2026-01-01T00:00:00Z",
            "dataset_name": "custom/sample",
            "documents_processed": 0,
            "tokenizers_count": 0,
            "tokenizers_processed": [],
            "selected_metric_keys": [],
            "config": {
                "max_documents": 0,
                "warmup_trials": 2,
                "timed_trials": 8,
                "batch_size": 16,
                "seed": 42,
                "parallelism": 1,
                "add_special_tokens": False,
                "padding": False,
                "truncation": False,
                "max_length": None,
                "store_per_document_stats": True,
                "per_document_sample_size": 500,
            },
            "hardware_profile": {
                "runtime": "",
                "os": "",
                "cpu_model": None,
                "cpu_logical_cores": None,
                "memory_total_mb": None,
            },
            "trial_summary": {"warmup_trials": 0, "timed_trials": 0},
            "tokenizer_results": [],
            "dashboard": {"widgets": []},
            "per_document_stats": [],
            "runtime_metadata": {},
            "raw_observations": {},
        },
    )

    client = TestClient(app)
    by_id = client.get("/api/benchmarks/reports/9")
    assert by_id.status_code == 200
    payload = by_id.json()
    assert payload["status"] == "cancelled"
    assert payload["report_id"] == 9

###############################################################################


###############################################################################
class DummyJobManager__datasets_routes:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.last_job_type = ""
        self.started_jobs = 0

    # -------------------------------------------------------------------------
    def is_job_running(self, job_type: str | None = None) -> bool:
        return False

    # -------------------------------------------------------------------------
    def start_job(self, job_type, runner, args=(), kwargs=None):
        del runner, args, kwargs
        self.started_jobs += 1
        self.last_job_type = str(job_type)
        return "job-123"

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str):
        del job_id
        return {"job_type": self.last_job_type, "status": "pending"}

###############################################################################
def test_dataset_job_start_routes_return_202(monkeypatch) -> None:
    manager = DummyJobManager__datasets_routes()
    monkeypatch.setattr(app.state, "job_manager", manager)

    from server.services.datasets import DatasetService

    monkeypatch.setattr(
        DatasetService,
        "is_dataset_in_database",
        lambda self, dataset_name: bool(dataset_name),
    )

    client = TestClient(app)

    download_resp = client.post(
        "/api/datasets/download",
        json={"corpus": "wikitext", "configs": {"configuration": "wikitext-2-v1"}},
    )
    assert download_resp.status_code == 202
    assert download_resp.json()["job_id"] == "job-123"

    upload_resp = client.post(
        "/api/datasets/upload",
        files={"file": ("sample.csv", b"text\nhello\n", "text/csv")},
    )
    assert upload_resp.status_code == 202
    assert upload_resp.json()["job_id"] == "job-123"

    analyze_resp = client.post(
        "/api/datasets/analyze",
        json={"dataset_name": "custom/sample"},
    )
    assert analyze_resp.status_code == 202
    assert analyze_resp.json()["job_id"] == "job-123"

###############################################################################
def test_dataset_analysis_rejects_inverted_length_filter(monkeypatch) -> None:
    manager = DummyJobManager__datasets_routes()
    monkeypatch.setattr(app.state, "job_manager", manager)

    response = TestClient(app).post(
        "/api/datasets/analyze",
        json={
            "dataset_name": "custom/sample",
            "filters": {"min_length": 100, "max_length": 1},
        },
    )

    assert response.status_code == 422
    assert "Minimum document length must not exceed maximum document length." in response.text
    assert manager.started_jobs == 0

###############################################################################
def test_dataset_validation_preview_rejects_inverted_length_filter() -> None:
    response = TestClient(app).post(
        "/api/datasets/validation/preview",
        json={
            "dataset_name": "custom/sample",
            "filters": {"min_length": 100, "max_length": 1},
        },
    )

    assert response.status_code == 422
    assert "Minimum document length must not exceed maximum document length." in response.text

###############################################################################
def test_dataset_upload_accepts_limit_and_rejects_one_byte_over(monkeypatch) -> None:
    limit = 25 * 1024 * 1024
    manager = DummyJobManager__datasets_routes()
    captured_upload_sizes: list[int] = []
    original_start_job = manager.start_job

    def capture_start_job(job_type, runner, args=(), kwargs=None):
        if job_type == "dataset_upload" and kwargs is not None:
            captured_upload_sizes.append(len(kwargs["file_content"]))
        return original_start_job(job_type, runner, args=args, kwargs=kwargs)

    monkeypatch.setattr(manager, "start_job", capture_start_job)
    monkeypatch.setattr(app.state, "job_manager", manager)

    from server.api import datasets as datasets_api

    ###############################################################################
    class _DatasetCfg:
        allowed_extensions = (".csv", ".xlsx")
        max_upload_bytes = limit

    ###############################################################################
    class _Settings:
        datasets = _DatasetCfg()
        jobs = type("JobsCfg", (), {"polling_interval": 1.0})()

    monkeypatch.setattr(datasets_api, "get_server_settings", lambda: _Settings())

    payload = b"text\n" + b"x" * (limit - len(b"text\n"))
    client = TestClient(app)
    accepted = client.post(
        "/api/datasets/upload",
        files={"file": ("near-limit.csv", payload, "text/csv")},
    )

    assert accepted.status_code == 202
    assert captured_upload_sizes == [limit]

    rejected = client.post(
        "/api/datasets/upload",
        files={"file": ("near-limit.csv", payload + b"x", "text/csv")},
    )

    assert rejected.status_code == 413
    assert manager.started_jobs == 1

###############################################################################
def test_dataset_list_passes_catalog_filters_to_service(monkeypatch) -> None:
    from server.services.datasets import DatasetService

    captured: dict[str, object] = {}

    def fake_previews(self, **filters):
        del self
        captured.update(filters)
        return [{"dataset_name": "custom/demo", "document_count": 4}]

    monkeypatch.setattr(DatasetService, "get_dataset_previews", fake_previews)

    response = TestClient(app).get(
        "/api/datasets/list?search= demo &source=custom"
        "&document_count_operator=at_most&document_count=5"
    )

    assert response.status_code == 200
    assert response.json() == {
        "datasets": [{"dataset_name": "custom/demo", "document_count": 4}],
        "count": 1,
    }
    assert captured == {
        "search": " demo ",
        "source": "custom",
        "document_count_operator": "at_most",
        "document_count": 5,
    }

###############################################################################
def test_dataset_delete_normalizes_identifier_and_returns_not_found_for_repeat(
    monkeypatch,
) -> None:
    from server.services.datasets import DatasetService

    calls: list[tuple[str, str]] = []
    available = {"custom/foo"}

    monkeypatch.setattr(
        DatasetService,
        "is_dataset_in_database",
        lambda self, name: calls.append(("exists", name)) or name in available,
    )
    monkeypatch.setattr(
        DatasetService,
        "remove_dataset",
        lambda self, name: (calls.append(("remove", name)), available.remove(name)),
    )

    client = TestClient(app)
    deleted = client.delete(
        "/api/datasets/delete",
        params={"dataset_name": "custom/foo"},
    )

    assert deleted.status_code == 200
    assert deleted.json() == {
        "status": "success",
        "dataset_name": "custom/foo",
        "message": "Dataset removed.",
    }
    assert calls == [("exists", "custom/foo"), ("remove", "custom/foo")]

    repeated = client.delete(
        "/api/datasets/delete",
        params={"dataset_name": "custom/foo"},
    )
    assert repeated.status_code == 404
    assert repeated.json()["detail"] == "Dataset 'custom/foo' not found."

###############################################################################


###############################################################################
def _export_payload() -> dict[str, object]:
    return {
        "dashboard_type": "benchmark",
        "report_name": "demo",
        "file_name": "demo",
        "dashboard_payload": {"report_id": 1},
    }

###############################################################################
def test_export_route_returns_pdf_headers_and_bytes(monkeypatch) -> None:
    from server.api import exports as exports_api

    ###############################################################################
    class FakeExportService:

        # -------------------------------------------------------------------------
        def export_dashboard_pdf(self, **kwargs):
            assert kwargs["dashboard_type"] == "benchmark"
            return SimpleNamespace(
                file_name="demo.pdf",
                page_count=2,
                pdf_bytes=b"%PDF-test",
            )

    monkeypatch.setattr(exports_api, "DashboardExportService", FakeExportService)

    response = TestClient(app).post(
        "/api/exports/dashboard/pdf", json=_export_payload()
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == 'attachment; filename="demo.pdf"'
    assert response.headers["x-export-page-count"] == "2"
    assert response.content == b"%PDF-test"

###############################################################################
def test_export_route_maps_expected_and_unexpected_service_failures(
    monkeypatch,
) -> None:
    from server.api import exports as exports_api

    ###############################################################################
    class ValueErrorService:

        # -------------------------------------------------------------------------
        def export_dashboard_pdf(self, **kwargs):
            del kwargs
            raise ValueError("unsupported visualization")

    monkeypatch.setattr(exports_api, "DashboardExportService", ValueErrorService)
    invalid = TestClient(app).post("/api/exports/dashboard/pdf", json=_export_payload())
    assert invalid.status_code == 400
    assert invalid.json()["detail"] == "unsupported visualization"

    ###############################################################################
    class FailingService:

        # -------------------------------------------------------------------------
        def export_dashboard_pdf(self, **kwargs):
            del kwargs
            raise RuntimeError("private rendering details")

    monkeypatch.setattr(exports_api, "DashboardExportService", FailingService)
    failed = TestClient(app).post("/api/exports/dashboard/pdf", json=_export_payload())
    assert failed.status_code == 500
    assert failed.json()["detail"] == "Failed to export dashboard as PDF."
    assert "private rendering" not in failed.text

###############################################################################


###############################################################################
def _job_status(job_id: str, status: str) -> dict[str, object]:
    return {
        "job_id": job_id,
        "job_type": "benchmark",
        "status": status,
        "progress": 42.0,
        "result": None,
        "error": None,
    }

###############################################################################
def test_job_routes_distinguish_missing_terminal_and_running_jobs(monkeypatch) -> None:

    ###############################################################################
    class FakeJobManager:

        # -------------------------------------------------------------------------
        def get_job_status(self, job_id: str):
            if job_id == "missing":
                return None
            return _job_status(job_id, "completed" if job_id == "done" else "running")

        # -------------------------------------------------------------------------
        def request_stop(self, job_id: str):
            if job_id == "missing":
                return None
            if job_id == "done":
                return _job_status(job_id, "completed")
            return _job_status(job_id, "stopping")

    monkeypatch.setattr(app.state, "job_manager", FakeJobManager())
    client = TestClient(app)

    assert client.get("/api/jobs/missing").status_code == 404
    terminal = client.post("/api/jobs/done/cancel")
    running = client.post("/api/jobs/active/cancel")

    assert terminal.status_code == 409
    assert "already completed" in terminal.json()["detail"]
    assert running.status_code == 200
    assert running.json()["status"] == "stopping"

###############################################################################
def test_reconciled_job_remains_addressable_and_rejects_cancellation(
    monkeypatch,
) -> None:

    ###############################################################################
    class MemoryJobRepository:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            now = time()
            self.rows = {
                "restart1": {
                    "job_id": "restart1",
                    "job_type": "benchmark",
                    "status": "running",
                    "progress": 20.0,
                    "result": None,
                    "error": None,
                    "failure_reason": None,
                    "created_at": now - 1,
                    "completed_at": None,
                }
            }

        # -------------------------------------------------------------------------
        def delete_expired_jobs(self, now: float, retention_seconds: float) -> int:
            del now, retention_seconds
            return 0

        # -------------------------------------------------------------------------
        def list_jobs(self) -> list[dict[str, object]]:
            return list(self.rows.values())

        # -------------------------------------------------------------------------
        def save_job(self, snapshot: dict[str, object]) -> None:
            self.rows[snapshot["job_id"]] = snapshot

    from server.services.jobs import JobManager

    manager = JobManager(repository=MemoryJobRepository())
    manager.restore_persisted_jobs()
    monkeypatch.setattr(app.state, "job_manager", manager)
    client = TestClient(app)

    status = client.get("/api/jobs/restart1")
    cancellation = client.post("/api/jobs/restart1/cancel")

    assert status.status_code == 200
    assert status.json()["status"] == "failed"
    assert status.json()["error"] == (
        "Job interrupted because the application restarted while it was running."
    )
    assert "failure_reason" not in status.json()
    assert cancellation.status_code == 409
    assert "already failed" in cancellation.json()["detail"]

###############################################################################


###############################################################################
def _key_item(key_id: int = 1) -> dict[str, object]:
    return {
        "id": key_id,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "is_active": False,
        "masked_preview": "********",
    }

###############################################################################
def test_key_routes_keep_raw_values_out_of_list_and_map_lifecycle_calls(
    monkeypatch,
) -> None:
    from server.api import keys as keys_api

    calls: list[tuple[str, object]] = []

    ###############################################################################
    class FakeKeyService:

        # -------------------------------------------------------------------------
        def add_key(self, raw_key: str):
            calls.append(("add", raw_key))
            return _key_item()

        # -------------------------------------------------------------------------
        def list_keys(self):
            return [_key_item()]

        # -------------------------------------------------------------------------
        def set_active_key(self, key_id: int) -> None:
            calls.append(("activate", key_id))

        # -------------------------------------------------------------------------
        def clear_active_key(self, key_id: int) -> None:
            calls.append(("deactivate", key_id))

        # -------------------------------------------------------------------------
        def delete_key(self, key_id: int, confirm: bool) -> None:
            calls.append(("delete", (key_id, confirm)))

    monkeypatch.setattr(keys_api, "HFAccessKeyService", FakeKeyService)
    client = TestClient(app)

    created = client.post("/api/keys", json={"key_value": "hf_example"})
    listed = client.get("/api/keys")
    activated = client.post("/api/keys/1/activate")
    deactivated = client.post("/api/keys/1/deactivate")
    deleted = client.delete("/api/keys/1?confirm=true")

    assert created.status_code == 201
    assert listed.status_code == 200
    assert "key_value" not in created.json()
    assert "key_value" not in listed.json()["keys"][0]
    assert activated.json()["status"] == "success"
    assert deactivated.json()["message"] == "Active key cleared."
    assert deleted.json()["status"] == "success"
    assert calls == [
        ("add", "hf_example"),
        ("activate", 1),
        ("deactivate", 1),
        ("delete", (1, True)),
    ]

###############################################################################
def test_key_reveal_is_blocked_when_server_policy_disables_reveal(monkeypatch) -> None:
    from server.api import keys as keys_api

    monkeypatch.setattr(keys_api, "is_key_reveal_enabled", lambda: False)

    response = TestClient(app).post("/api/keys/1/reveal")

    assert response.status_code == 403
    assert response.json()["detail"] == "Key reveal is disabled by server policy."

###############################################################################


###############################################################################
class _MemoryPersistence:

    # -------------------------------------------------------------------------
    def __init__(self, document: dict[str, object] | None = None) -> None:
        self.document = document

    # -------------------------------------------------------------------------
    def load(self) -> dict[str, object] | None:
        return json.loads(json.dumps(self.document)) if self.document is not None else None

    # -------------------------------------------------------------------------
    def save(self, payload: dict[str, object]) -> None:
        self.document = json.loads(json.dumps(payload))


###############################################################################
def _settings_client(
    defaults: ServerSettings,
    persistence: _MemoryPersistence,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    store = RuntimeSettingsStore(defaults, persistence)
    monkeypatch.setattr(startup, "_runtime_settings_store", store)
    monkeypatch.setattr(startup, "_default_settings", store.get_state().settings)
    application = FastAPI()
    application.include_router(datasets_router, prefix="/api")
    application.include_router(tokenizers_router, prefix="/api")
    application.include_router(settings_router, prefix="/api")
    application.state.settings = store.get_state().settings
    return TestClient(application)

###############################################################################
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

###############################################################################
def _constraint(field, name: str):
    return next(
        (
            getattr(metadata, name)
            for metadata in field.metadata
            if hasattr(metadata, name)
        ),
        None,
    )

###############################################################################
def _runtime_setting_spec(group: str, name: str):
    return next(
        spec
        for spec in _runtime_setting_fields()
        if spec[0] == group and spec[1] == name
    )

###############################################################################
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

###############################################################################
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

###############################################################################
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

###############################################################################
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

###############################################################################
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
    return _settings_client(defaults, _MemoryPersistence(), monkeypatch)

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
    assert "cors_origins" not in response.text
    assert "paths" not in response.text

###############################################################################
def test_dashboard_preferences_are_database_scoped_and_revisioned(
    settings_client: TestClient,
) -> None:
    original = settings_client.get("/api/settings/dashboard")
    assert original.status_code == 200
    assert original.json() == {
        "revision": 0,
        "preferences": {
            "ordered_widget_ids": [],
            "hidden_widget_ids": [],
            "visualization_by_widget_id": {},
            "baseline_by_report_id": {},
        },
    }

    saved = settings_client.put(
        "/api/settings/dashboard",
        json={
            "expected_revision": 0,
            "preferences": {
                "ordered_widget_ids": ["speed"],
                "hidden_widget_ids": [],
                "visualization_by_widget_id": {"speed": "horizontal_bar"},
                "baseline_by_report_id": {"12": "alpha"},
            },
        },
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["revision"] == 1
    assert saved.json()["preferences"]["visualization_by_widget_id"] == {
        "speed": "horizontal_bar"
    }
    assert settings_client.get("/api/settings").json()["revision"] == 0

    stale = settings_client.put(
        "/api/settings/dashboard",
        json={
            "expected_revision": 0,
            "preferences": saved.json()["preferences"],
        },
    )
    assert stale.status_code == 409

###############################################################################
def test_dashboard_preferences_reject_invalid_visualizations(
    settings_client: TestClient,
) -> None:
    response = settings_client.put(
        "/api/settings/dashboard",
        json={
            "expected_revision": 0,
            "preferences": {
                "ordered_widget_ids": ["speed"],
                "hidden_widget_ids": [],
                "visualization_by_widget_id": {"speed": "unknown"},
                "baseline_by_report_id": {},
            },
        },
    )

    assert response.status_code == 422

###############################################################################
def test_public_runtime_matrix_matches_typed_backend_schema() -> None:
    fields = list(_runtime_setting_fields())
    api_keys = {f"{group}.{name}" for group, name, _, _ in fields}

    assert len(api_keys) == 16
    assert api_keys == set(RuntimeSettingsStore._EDITABLE_KEYS)
    for _group, _name, request_field, settings_field in fields:
        assert _constraint(request_field, "ge") == _constraint(settings_field, "ge")
        assert _constraint(request_field, "le") == _constraint(settings_field, "le")

###############################################################################
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

###############################################################################
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
    persistence = cast(_MemoryPersistence, store._persistence)
    persisted_before = json.dumps(persistence.document, sort_keys=True)

    payload = {
        "expected_revision": before["revision"],
        group: {name: value},
        companion_group: {companion_name: replacement_companion},
    }
    rejected = settings_client.patch("/api/settings", json=payload)

    assert rejected.status_code == 422, f"{group}.{name} {case}: {rejected.text}"
    assert settings_client.get("/api/settings").json() == before
    assert json.dumps(persistence.document, sort_keys=True) == persisted_before

###############################################################################
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
    persistence = cast(_MemoryPersistence, store._persistence)
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
    persisted_before = json.dumps(persistence.document, sort_keys=True)

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
        assert json.dumps(persistence.document, sort_keys=True) == persisted_before

###############################################################################
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

    ###############################################################################
    class CapturingThread(threading.Thread):

        # -------------------------------------------------------------------------
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

    ###############################################################################
    class FakeHfApi:

        # -------------------------------------------------------------------------
        def __init__(self, token: str | None) -> None:
            del token

        # -------------------------------------------------------------------------
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

    ###############################################################################
    class NewJobManager:

        # -------------------------------------------------------------------------
        def is_job_running(self, _job_type: str | None = None) -> bool:
            return False

        # -------------------------------------------------------------------------
        def start_job(self, job_type, runner, args=(), kwargs=None):
            del runner, args, kwargs
            self.job_type = job_type
            return "settings-workflow-job"

        # -------------------------------------------------------------------------
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
        {"expected_revision": 0, "unknown": {"histogram_bins": 30}},
        {"expected_revision": 0, "network": {"fastapi_port": 9000}},
        {"expected_revision": 0, "cors_origins": ["https://example.com"]},
        {"expected_revision": 0, "tokenizers": {"max_discovery_limit": 10}},
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
def test_dashboard_persistence_error_keeps_previous_preferences(
    settings_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = startup.get_runtime_settings_store()

    def fail_persist(_overrides, _revision, **_kwargs) -> None:
        raise RuntimeSettingsPersistenceError("persistence failure")

    monkeypatch.setattr(store, "_persist", fail_persist)
    response = settings_client.put(
        "/api/settings/dashboard",
        json={
            "expected_revision": 0,
            "preferences": {
                "ordered_widget_ids": ["speed"],
                "hidden_widget_ids": [],
                "visualization_by_widget_id": {"speed": "bar"},
                "baseline_by_report_id": {},
            },
        },
    )

    assert response.status_code == 500
    current = settings_client.get("/api/settings/dashboard").json()
    assert current["revision"] == 0
    assert current["preferences"]["ordered_widget_ids"] == []

###############################################################################
def test_invalid_database_settings_warning_is_sanitized_and_cleared_by_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = startup.get_server_settings()
    persistence = _MemoryPersistence(
        {
            "schema_version": 2,
            "revision": 0,
            "overrides": {},
            "dashboard_revision": 0,
            "dashboard_preferences": {},
        }
    )
    client = _settings_client(defaults, persistence, monkeypatch)

    warning = client.get("/api/settings")
    assert warning.status_code == 200
    assert warning.json()["warning"]

    saved = client.patch(
        "/api/settings",
        json={"expected_revision": 0, "datasets": {"histogram_bins": 30}},
    )
    assert saved.status_code == 200
    assert saved.json()["warning"] is None
    assert persistence.document is not None
    assert persistence.document["overrides"]["datasets"]["histogram_bins"] == 30

###############################################################################


###############################################################################
class DummyJobManager__tokenizers_routes:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.last_job_type = ""

    # -------------------------------------------------------------------------
    def is_job_running(self, job_type: str | None = None) -> bool:
        return False

    # -------------------------------------------------------------------------
    def start_job(self, job_type, runner, args=(), kwargs=None):
        del runner, args, kwargs
        self.last_job_type = str(job_type)
        return "job-xyz"

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str):
        del job_id
        return {"job_type": self.last_job_type, "status": "pending"}

###############################################################################
def test_tokenizer_upload_validation(monkeypatch) -> None:
    client = TestClient(app)

    invalid_ext = client.post(
        "/api/tokenizers/upload",
        files={"file": ("tokenizer.txt", b"x", "text/plain")},
    )
    assert invalid_ext.status_code == 400

    empty = client.post(
        "/api/tokenizers/upload",
        files={"file": ("tokenizer.json", b"", "application/json")},
    )
    assert empty.status_code == 400

    from server.api import tokenizers as tokenizers_api

    ###############################################################################
    class _TokenizerCfg:
        max_upload_bytes = 1

    ###############################################################################
    class _Settings:
        tokenizers = _TokenizerCfg()
        jobs = type("JobsCfg", (), {"polling_interval": 1.0})()

    monkeypatch.setattr(tokenizers_api, "get_server_settings", lambda: _Settings())

    oversized = client.post(
        "/api/tokenizers/upload",
        files={"file": ("tokenizer.json", b"{}", "application/json")},
    )
    assert oversized.status_code == 413
    monkeypatch.setattr(
        tokenizers_api,
        "get_server_settings",
        lambda: type(
            "Settings",
            (),
            {
                "tokenizers": type(
                    "TokenizerCfg", (), {"max_upload_bytes": 10_000_000}
                )(),
                "jobs": type("JobsCfg", (), {"polling_interval": 1.0})(),
            },
        )(),
    )

    def fake_upload(self, content: bytes, normalized_filename: str, safe_stem: str):
        del self, content, normalized_filename, safe_stem
        return {
            "status": "success",
            "tokenizer_name": "CUSTOM_demo",
            "is_compatible": True,
        }

    monkeypatch.setattr(
        tokenizers_api.TokenizersService,
        "register_custom_tokenizer_from_upload",
        fake_upload,
    )

    ok_upload = client.post(
        "/api/tokenizers/upload",
        files={"file": ("tokenizer.json", b"{}", "application/json")},
    )
    assert ok_upload.status_code == 200
    assert ok_upload.json()["is_compatible"] is True

###############################################################################
def test_tokenizer_delete_supports_encoded_and_custom_names_and_returns_404(
    monkeypatch,
) -> None:
    from server.api import tokenizers as tokenizers_api

    deleted: list[str] = []

    available = {"google-bert/bert-base-uncased", "CUSTOM_sample"}

    def fake_remove(self, name: str) -> bool:
        del self
        deleted.append(name)
        if name in available:
            available.remove(name)
            return True
        return False

    monkeypatch.setattr(
        tokenizers_api.TokenizersService, "remove_tokenizer", fake_remove
    )

    client = TestClient(app)
    downloaded = client.delete(
        "/api/tokenizers/delete",
        params={"tokenizer_name": "google-bert/bert-base-uncased"},
    )
    custom = client.delete(
        "/api/tokenizers/delete",
        params={"tokenizer_name": "CUSTOM_sample"},
    )
    missing = client.delete(
        "/api/tokenizers/delete",
        params={"tokenizer_name": "CUSTOM_sample"},
    )

    assert downloaded.status_code == 200
    assert custom.status_code == 200
    assert missing.status_code == 404
    assert downloaded.json()["tokenizer_name"] == "google-bert/bert-base-uncased"
    assert missing.json()["detail"] == "Tokenizer 'CUSTOM_sample' is not persisted."
    assert deleted == [
        "google-bert/bert-base-uncased",
        "CUSTOM_sample",
        "CUSTOM_sample",
    ]

###############################################################################
def test_tokenizer_job_routes_return_202(monkeypatch) -> None:
    manager = DummyJobManager__tokenizers_routes()
    monkeypatch.setattr(app.state, "job_manager", manager)

    from server.services.keys import HFAccessKeyService
    from server.services.tokenizers import TokenizersService

    monkeypatch.setattr(HFAccessKeyService, "get_active_key", lambda self: "token")
    monkeypatch.setattr(
        TokenizersService,
        "has_available_tokenizer",
        lambda self, tokenizer_name: tokenizer_name == "bert-base-uncased",
    )

    client = TestClient(app)

    download_resp = client.post(
        "/api/tokenizers/download",
        json={"tokenizers": ["bert-base-uncased"]},
    )
    assert download_resp.status_code == 202
    assert download_resp.json()["job_id"] == "job-xyz"

    report_resp = client.post(
        "/api/tokenizers/reports/generate",
        json={"tokenizer_name": "bert-base-uncased"},
    )
    assert report_resp.status_code == 202
    assert report_resp.json()["job_id"] == "job-xyz"

###############################################################################
def test_tokenizer_discovery_returns_sanitized_500_on_upstream_failure(
    monkeypatch,
) -> None:
    from server.services.tokenizers import TokenizersService

    def fail_discovery(self, query):
        del self, query
        raise RuntimeError("private upstream credentials and response details")

    monkeypatch.setattr(TokenizersService, "discover_tokenizers", fail_discovery)

    response = TestClient(app).get("/api/tokenizers/discover")

    assert response.status_code == 500
    assert (
        response.json()["detail"] == "Failed to discover tokenizers from HuggingFace."
    )
    assert "private upstream" not in response.text

###############################################################################
def test_tokenizer_discovery_validates_combined_query_and_structured_response(
    monkeypatch,
) -> None:
    from server.contracts.tokenizers import TokenizerDiscoveryResponse
    from server.services.tokenizers import TokenizersService

    captured = {}

    def fake_discovery(self, query):
        del self
        captured.update(query.model_dump())
        return TokenizerDiscoveryResponse.model_validate(
            {
                "items": [
                    {
                        "identifier": "bert-base-uncased",
                        "pipeline_tag": "fill-mask",
                        "library_name": "transformers",
                        "downloads": 10,
                        "likes": 2,
                        "last_modified": None,
                        "gated": False,
                        "tags": ["core"],
                        "vocabulary_size": None,
                    }
                ],
                "count": 1,
                "fetched_count": 3,
            }
        )

    monkeypatch.setattr(TokenizersService, "discover_tokenizers", fake_discovery)
    response = TestClient(app).get(
        "/api/tokenizers/discover?search=%20bert%20&limit=5&pipeline_tag=fill-mask"
        "&author=google&include_tags=core&exclude_tags=audio&access=public&sort=downloads"
    )

    assert response.status_code == 200
    assert response.json()["items"][0]["identifier"] == "bert-base-uncased"
    assert response.json()["count"] == 1
    assert captured["search"] == "bert"
    assert captured["limit"] == 5
    assert captured["pipeline_tag"] == "fill-mask"
    assert captured["author"] == "google"
    assert captured["include_tags"] == ["core"]
    assert captured["exclude_tags"] == ["audio"]
    assert captured["access"] == "public"

###############################################################################
def test_tokenizer_discovery_uses_runtime_default(monkeypatch) -> None:
    from server.api import tokenizers as tokenizers_api
    from server.contracts.tokenizers import TokenizerDiscoveryResponse
    from server.services.tokenizers import TokenizersService

    ###############################################################################
    class _TokenizerCfg:
        default_discovery_limit = 7
        max_discovery_limit = 9
        max_discovery_candidates = 30
        metadata_candidate_multiplier = 2

    ###############################################################################
    class _Settings:
        tokenizers = _TokenizerCfg()

    captured: dict[str, int] = {}

    def fake_discovery(self, query):
        del self
        captured["limit"] = query.limit
        return TokenizerDiscoveryResponse(items=[], count=0, fetched_count=0)

    monkeypatch.setattr(tokenizers_api, "get_server_settings", lambda: _Settings())
    monkeypatch.setattr(TokenizersService, "discover_tokenizers", fake_discovery)

    client = TestClient(app)
    response = client.get("/api/tokenizers/discover")

    assert response.status_code == 200
    assert captured["limit"] == 7

###############################################################################
def test_tokenizer_discovery_enforces_runtime_maximum(monkeypatch) -> None:
    from server.api import tokenizers as tokenizers_api

    ###############################################################################
    class _TokenizerCfg:
        default_discovery_limit = 7
        max_discovery_limit = 9

    ###############################################################################
    class _Settings:
        tokenizers = _TokenizerCfg()

    monkeypatch.setattr(tokenizers_api, "get_server_settings", lambda: _Settings())

    response = TestClient(app).get("/api/tokenizers/discover?limit=10")

    assert response.status_code == 422
    assert "configured maximum (9)" in response.json()["detail"]

###############################################################################
def test_tokenizer_discovery_rejects_invalid_query(monkeypatch) -> None:
    from server.services.tokenizers import TokenizersService

    called = {"value": False}

    def fail_discovery(self, query):
        del self, query
        called["value"] = True
        return None

    monkeypatch.setattr(TokenizersService, "discover_tokenizers", fail_discovery)
    client = TestClient(app)
    assert (
        client.get(
            "/api/tokenizers/discover?pipeline_tag=image-tokenization"
        ).status_code
        == 422
    )
    assert (
        client.get("/api/tokenizers/discover?vocabulary_operator=at_least").status_code
        == 422
    )
    assert client.get("/api/tokenizers/discover?vocabulary_size=-1").status_code == 422
    assert (
        client.get(
            "/api/tokenizers/discover?include_tags=shared&exclude_tags=shared"
        ).status_code
        == 422
    )
    assert called["value"] is False

###############################################################################
def test_tokenizer_preview_enforces_input_limits(monkeypatch) -> None:
    from server.contracts.tokenizers import TokenizerPreviewResult
    from server.services.tokenizer_preview import TokenizerPreviewService

    def fake_preview(self, request):
        del self
        return [
            TokenizerPreviewResult(
                tokenizer_name=name, status="success", token_count=0
            )
            for name in request.tokenizers
        ]

    monkeypatch.setattr(TokenizerPreviewService, "preview", fake_preview)
    client = TestClient(app)
    invalid_payloads = [
        {"text": "", "tokenizers": ["gpt2"]},
        {"text": "   ", "tokenizers": ["gpt2"]},
        {"text": "x" * 2001, "tokenizers": ["gpt2"]},
        {"text": "hello", "tokenizers": []},
        {"text": "hello", "tokenizers": ["a", "b", "c", "d"]},
    ]
    for payload in invalid_payloads:
        assert client.post("/api/tokenizers/preview", json=payload).status_code == 422

    response = client.post(
        "/api/tokenizers/preview",
        json={"text": "hello", "tokenizers": ["gpt2", "gpt2"]},
    )
    assert response.status_code == 200
    assert [item["tokenizer_name"] for item in response.json()["results"]] == ["gpt2"]

###############################################################################
def test_tokenizer_list_passes_catalog_filters_to_service(monkeypatch) -> None:
    from server.services.tokenizers import TokenizersService

    captured: dict[str, object] = {}

    def fake_catalog(self, **filters):
        del self
        captured.update(filters)
        return [
            {
                "tokenizer_name": "CUSTOM_demo",
                "source": "custom",
                "has_report": False,
                "vocabulary_size": 3,
            }
        ]

    monkeypatch.setattr(TokenizersService, "list_tokenizer_catalog", fake_catalog)

    response = TestClient(app).get(
        "/api/tokenizers/list?search= demo &source=custom"
        "&vocabulary_size_operator=at_most&vocabulary_size=5"
    )

    assert response.status_code == 200
    assert response.json() == {
        "tokenizers": [
            {
                "tokenizer_name": "CUSTOM_demo",
                "source": "custom",
                "has_report": False,
                "vocabulary_size": 3,
            }
        ],
        "count": 1,
    }
    assert captured == {
        "search": " demo ",
        "source": "custom",
        "vocabulary_size_operator": "at_most",
        "vocabulary_size": 5,
    }

###############################################################################
def test_key_encryption_must_be_initialized_before_adding_keys(
    tmp_path: Path, monkeypatch
) -> None:
    from server.common.utils import encryption

    monkeypatch.setattr(encryption, "DATA_PATH", tmp_path)
    client = TestClient(app)

    assert client.get("/api/keys/encryption").json() == {"initialized": False}
    blocked = client.post("/api/keys", json={"key_value": "hf_example_key"})
    assert blocked.status_code == 409
    assert client.post("/api/keys/encryption/initialize").status_code == 201
    assert client.get("/api/keys/encryption").json() == {"initialized": True}
    assert client.post("/api/keys/encryption/initialize").status_code == 409

###############################################################################
def test_path_settings_persist_and_flag_restart(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TKBEN_PATHS_CONFIG", str(tmp_path / "paths-config.json"))
    client = TestClient(app)
    new_data = tmp_path / "new-data"

    response = client.put(
        "/api/settings/paths",
        json={"data_dir": str(new_data), "log_dir": str(new_data / "logs")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["data_dir"] == str(new_data)
    assert body["restart_required"] is True
    assert (new_data / "logs").is_dir()
    assert client.get("/api/settings/paths").json()["log_dir"] == str(new_data / "logs")
