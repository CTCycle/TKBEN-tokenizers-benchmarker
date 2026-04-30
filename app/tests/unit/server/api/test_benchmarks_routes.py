from __future__ import annotations

from fastapi.testclient import TestClient

from TKBEN.server.app import app


class DummyJobManager:
    def __init__(self) -> None:
        self.last_job_type = ""

    def is_job_running(self, job_type: str | None = None) -> bool:
        return False

    def start_job(self, job_type, runner, args=(), kwargs=None):
        del runner, args, kwargs
        self.last_job_type = str(job_type)
        return "job-bench"

    def get_job_status(self, job_id: str):
        del job_id
        return {"job_type": self.last_job_type, "status": "pending"}


def test_benchmark_run_route_returns_202(monkeypatch) -> None:
    manager = DummyJobManager()
    monkeypatch.setattr(app.state, "job_manager", manager)

    from TKBEN.server.services.benchmarks import BenchmarkService

    monkeypatch.setattr(BenchmarkService, "resolve_custom_tokenizer_selection", lambda self, name: {})
    monkeypatch.setattr(BenchmarkService, "get_dataset_document_count", lambda self, dataset_name: 3)
    monkeypatch.setattr(BenchmarkService, "get_missing_persisted_tokenizers", lambda self, tokenizers: [])

    client = TestClient(app)
    resp = client.post(
        "/api/benchmarks/run",
        json={
            "tokenizers": ["bert-base-uncased"],
            "dataset_name": "custom/sample",
            "config": {"max_documents": 10},
        },
    )

    assert resp.status_code == 202
    assert resp.json()["job_id"] == "job-bench"


def test_benchmark_list_and_by_id(monkeypatch) -> None:
    from TKBEN.server.services.benchmarks import BenchmarkService

    monkeypatch.setattr(
        BenchmarkService,
        "list_benchmark_reports",
        lambda self, limit=200: [
            {
                "report_id": 1,
                "report_version": 2,
                "created_at": "2026-01-01T00:00:00Z",
                "run_name": "run",
                "dataset_name": "custom/sample",
                "documents_processed": 2,
                "tokenizers_count": 1,
                "tokenizers_processed": ["bert-base-uncased"],
                "selected_metric_keys": ["global.tokenization_speed_tps"],
            }
        ],
    )
    monkeypatch.setattr(
        BenchmarkService,
        "load_benchmark_report_by_id",
        lambda self, report_id: {
            "status": "success",
            "report_id": report_id,
            "report_version": 2,
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
            },
            "hardware_profile": {"runtime": "", "os": "", "cpu_model": None, "cpu_logical_cores": None, "memory_total_mb": None},
            "trial_summary": {"warmup_trials": 2, "timed_trials": 8},
            "tokenizer_results": [],
            "chart_data": {
                "efficiency": [],
                "fidelity": [],
                "vocabulary": [],
                "fragmentation": [],
                "latency_or_memory_distribution": [],
            },
            "per_document_stats": [],
        },
    )

    client = TestClient(app)

    listed = client.get("/api/benchmarks/reports")
    assert listed.status_code == 200
    assert listed.json()["reports"][0]["report_id"] == 1

    by_id = client.get("/api/benchmarks/reports/1")
    assert by_id.status_code == 200
    assert by_id.json()["report_id"] == 1
