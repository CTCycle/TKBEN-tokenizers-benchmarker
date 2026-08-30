from __future__ import annotations

from fastapi.testclient import TestClient

from server.app import app


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
def test_benchmark_run_accepts_selected_persisted_custom_tokenizer(
    monkeypatch,
) -> None:
    manager = DummyJobManager()
    monkeypatch.setattr(app.state, "job_manager", manager)

    from server.services.benchmarks import BenchmarkService

    prepared_payload = {}

    def fake_prepare(self, payload):
        del self
        prepared_payload.update(payload.model_dump())
        return payload.model_dump()

    monkeypatch.setattr(BenchmarkService, "prepare_run", fake_prepare)

    client = TestClient(app)
    resp = client.post(
        "/api/benchmarks/run",
        json={
            "tokenizers": ["CUSTOM_demo"],
            "dataset_name": "custom/sample",
            "config": {"max_documents": 2},
        },
    )

    assert resp.status_code == 202
    assert prepared_payload["tokenizers"] == ["CUSTOM_demo"]


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
                "include_lm_metrics": False,
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
                "include_lm_metrics": False,
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
