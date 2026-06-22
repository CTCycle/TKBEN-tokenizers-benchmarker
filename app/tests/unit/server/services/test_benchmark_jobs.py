from __future__ import annotations

from server.services.benchmark_jobs import BenchmarkJobService

###############################################################################
class DummyJobManager:

    # -------------------------------------------------------------------------
    def __init__(self, *, stopped: bool = False) -> None:
        self.stopped = stopped

    # -------------------------------------------------------------------------
    def should_stop(self, job_id: str) -> bool:
        del job_id
        return self.stopped

    # -------------------------------------------------------------------------
    def update_progress(self, job_id: str, value: float) -> None:
        del job_id, value

###############################################################################
def test_run_benchmark_job_builds_and_saves_report(monkeypatch) -> None:

    ###############################################################################
    class FakeBenchmarkService:

        # -------------------------------------------------------------------------
        def __init__(self, max_documents: int = 0):
            self.max_documents = max_documents

        # -------------------------------------------------------------------------
        def run_benchmarks(self, **kwargs):
            del kwargs
            return {
                "status": "success",
                "schema_version": 1,
                "methodology_version": "v1_observed_trials",
                "dataset_name": "custom/sample",
                "documents_processed": 2,
                "tokenizers_processed": ["bert-base-uncased"],
                "tokenizers_count": 1,
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
                    "runtime": "3.14",
                    "os": "test",
                    "cpu_model": None,
                    "cpu_logical_cores": None,
                    "memory_total_mb": None,
                },
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
                "runtime_metadata": {},
                "raw_observations": {},
            }

        # -------------------------------------------------------------------------
        def save_benchmark_report(self, payload):
            assert payload["dataset_name"] == "custom/sample"
            return 11

    monkeypatch.setattr(
        "server.services.benchmark_jobs.BenchmarkService", FakeBenchmarkService
    )

    service = BenchmarkJobService()
    result = service.run_benchmark_job(
        request_payload={
            "dataset_name": "custom/sample",
            "tokenizers": ["bert-base-uncased"],
            "config": {},
        },
        job_manager=DummyJobManager(),
        job_id="job1",
    )

    assert result["report_id"] == 11
    assert result["report_version"] == 2

###############################################################################
def test_run_benchmark_job_returns_cancelled_payload_without_persist(
    monkeypatch,
) -> None:

    ###############################################################################
    class FakeBenchmarkService:

        # -------------------------------------------------------------------------
        def __init__(self, max_documents: int = 0):
            self.max_documents = max_documents

        # -------------------------------------------------------------------------
        def run_benchmarks(self, **kwargs):
            del kwargs
            return {
                "status": "cancelled",
                "schema_version": 1,
                "methodology_version": "v1_observed_trials",
                "dataset_name": "custom/sample",
                "documents_processed": 0,
                "tokenizers_processed": [],
                "tokenizers_count": 0,
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
                "chart_data": {
                    "efficiency": [],
                    "fidelity": [],
                    "vocabulary": [],
                    "fragmentation": [],
                    "latency_or_memory_distribution": [],
                },
                "per_document_stats": [],
                "runtime_metadata": {},
                "raw_observations": {},
            }

        # -------------------------------------------------------------------------
        def save_benchmark_report(self, payload):
            raise AssertionError(
                "save_benchmark_report must not be called for cancelled runs"
            )

    monkeypatch.setattr(
        "server.services.benchmark_jobs.BenchmarkService", FakeBenchmarkService
    )

    service = BenchmarkJobService()
    result = service.run_benchmark_job(
        request_payload={
            "dataset_name": "custom/sample",
            "tokenizers": ["bert-base-uncased"],
            "config": {},
        },
        job_manager=DummyJobManager(stopped=True),
        job_id="job1",
    )

    assert result["status"] == "cancelled"
    assert result["report_id"] is None
