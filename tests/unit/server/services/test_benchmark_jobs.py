from __future__ import annotations

from TKBEN.server.services.benchmark_jobs import BenchmarkJobService


class DummyJobManager:
    def should_stop(self, job_id: str) -> bool:
        del job_id
        return False

    def update_progress(self, job_id: str, value: float) -> None:
        del job_id, value


def test_run_benchmark_job_builds_and_saves_report(monkeypatch) -> None:
    class FakeBenchmarkService:
        def __init__(self, max_documents: int = 0):
            self.max_documents = max_documents

        def run_benchmarks(self, **kwargs):
            del kwargs
            return {
                "dataset_name": "custom/sample",
                "documents_processed": 2,
                "tokenizers_processed": ["bert-base-uncased"],
                "tokenizers_count": 1,
                "global_metrics": [],
                "per_document_stats": [],
            }

        def save_benchmark_report(self, payload):
            assert payload["dataset_name"] == "custom/sample"
            return 11

    monkeypatch.setattr("TKBEN.server.services.benchmark_jobs.BenchmarkService", FakeBenchmarkService)

    service = BenchmarkJobService()
    result = service.run_benchmark_job(
        request_payload={"dataset_name": "custom/sample", "tokenizers": ["bert-base-uncased"], "config": {}},
        job_manager=DummyJobManager(),
        job_id="job1",
    )

    assert result["report_id"] == 11
    assert result["report_version"] == 2
