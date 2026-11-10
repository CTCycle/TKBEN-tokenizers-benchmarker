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
def test_benchmark_report_tags_route_returns_404_for_missing_report(
    monkeypatch,
) -> None:
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
    assert (
        "Minimum document length must not exceed maximum document length."
        in response.text
    )
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
    assert (
        "Minimum document length must not exceed maximum document length."
        in response.text
    )


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
        def prune_jobs(self, max_jobs: int) -> None:
            del max_jobs

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
