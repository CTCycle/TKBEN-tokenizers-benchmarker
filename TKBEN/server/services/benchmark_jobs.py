from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from TKBEN.server.services.benchmarks import BenchmarkService
from TKBEN.server.services.benchmark_payloads import BenchmarkPayloadBuilder
from TKBEN.server.services.jobs import JobManager, JobProgressReporter, JobStopChecker


###############################################################################
class BenchmarkJobService:
    def __init__(self) -> None:
        self.payload_builder = BenchmarkPayloadBuilder()

    # -------------------------------------------------------------------------
    def run_benchmark_job(
        self,
        request_payload: dict[str, Any],
        job_manager: JobManager,
        job_id: str,
    ) -> dict[str, Any]:
        config = request_payload.get("config", {})
        if not isinstance(config, dict):
            config = {}

        service = BenchmarkService(max_documents=config.get("max_documents", 0))
        progress_callback = JobProgressReporter(job_manager, job_id)
        should_stop = JobStopChecker(job_manager, job_id)

        result = service.run_benchmarks(
            dataset_name=request_payload.get("dataset_name", ""),
            tokenizer_ids=request_payload.get("tokenizers", []),
            custom_tokenizers=request_payload.get("custom_tokenizers", {}),
            run_name=request_payload.get("run_name"),
            selected_metric_keys=request_payload.get("selected_metric_keys"),
            progress_callback=progress_callback,
            should_stop=should_stop,
        )
        if job_manager.should_stop(job_id):
            return {}

        payload = self.payload_builder.build_benchmark_payload(
            result,
            request_payload.get("dataset_name", ""),
            config_payload=config,
        )
        payload["created_at"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        payload["report_version"] = 2
        report_id = service.save_benchmark_report(payload)
        payload["report_id"] = int(report_id)
        return payload
