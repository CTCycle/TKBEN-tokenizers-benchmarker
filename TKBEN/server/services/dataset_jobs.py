from __future__ import annotations

from typing import Any

from TKBEN.server.services.datasets import DatasetService
from TKBEN.server.services.jobs import JobManager, JobProgressReporter, JobStopChecker


###############################################################################
class DatasetJobService:
    def build_histogram_payload(self, histogram_data: dict[str, Any]) -> dict[str, Any]:
        return {
            "bins": histogram_data.get("bins", []),
            "counts": histogram_data.get("counts", []),
            "bin_edges": histogram_data.get("bin_edges", []),
            "min_length": histogram_data.get("min_length", 0),
            "max_length": histogram_data.get("max_length", 0),
            "mean_length": histogram_data.get("mean_length", 0.0),
            "median_length": histogram_data.get("median_length", 0.0),
        }

    # -------------------------------------------------------------------------
    def build_dataset_mutation_payload(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "success",
            "dataset_name": result.get("dataset_name", ""),
            "text_column": result.get("text_column", ""),
            "document_count": result.get("document_count", 0),
            "saved_count": result.get("saved_count", 0),
            "histogram": self.build_histogram_payload(result.get("histogram", {})),
        }

    # -------------------------------------------------------------------------
    def build_download_payload(self, result: dict[str, Any]) -> dict[str, Any]:
        return self.build_dataset_mutation_payload(result)

    # -------------------------------------------------------------------------
    def build_upload_payload(self, result: dict[str, Any]) -> dict[str, Any]:
        return self.build_dataset_mutation_payload(result)

    # -------------------------------------------------------------------------
    def extract_configuration(self, request_payload: dict[str, Any]) -> str | None:
        configs = request_payload.get("configs")
        if isinstance(configs, dict):
            value = configs.get("configuration")
            if isinstance(value, str) and value.strip():
                return value.strip()
            return None
        return None

    # -------------------------------------------------------------------------
    def build_analysis_payload(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "success",
            "report_id": result.get("report_id"),
            "report_version": result.get("report_version", 2),
            "created_at": result.get("created_at"),
            "dataset_name": result.get("dataset_name", ""),
            "session_name": result.get("session_name"),
            "selected_metric_keys": result.get("selected_metric_keys", []),
            "session_parameters": result.get("session_parameters", {}),
            "document_count": result.get("document_count", 0),
            "document_length_histogram": self.build_histogram_payload(
                result.get("document_length_histogram", {})
            ),
            "word_length_histogram": self.build_histogram_payload(
                result.get("word_length_histogram", {})
            ),
            "min_document_length": result.get("min_document_length", 0),
            "max_document_length": result.get("max_document_length", 0),
            "most_common_words": result.get("most_common_words", []),
            "least_common_words": result.get("least_common_words", []),
            "longest_words": result.get("longest_words", []),
            "shortest_words": result.get("shortest_words", []),
            "word_cloud_terms": result.get("word_cloud_terms", []),
            "aggregate_statistics": result.get("aggregate_statistics", {}),
            "per_document_stats": result.get("per_document_stats", {}),
        }

    # -------------------------------------------------------------------------
    def run_download_job(
        self,
        request_payload: dict[str, Any],
        job_manager: JobManager,
        job_id: str,
    ) -> dict[str, Any]:
        service = DatasetService()
        progress_callback = JobProgressReporter(job_manager, job_id)
        should_stop = JobStopChecker(job_manager, job_id)
        result = service.download_and_persist(
            corpus=request_payload.get("corpus", ""),
            config=self.extract_configuration(request_payload),
            remove_invalid=True,
            progress_callback=progress_callback,
            should_stop=should_stop,
            job_id=job_id,
        )
        if job_manager.should_stop(job_id):
            return {}
        return self.build_download_payload(result)

    # -------------------------------------------------------------------------
    def run_upload_job(
        self,
        file_content: bytes,
        filename: str,
        job_manager: JobManager,
        job_id: str,
    ) -> dict[str, Any]:
        service = DatasetService()
        progress_callback = JobProgressReporter(job_manager, job_id)
        should_stop = JobStopChecker(job_manager, job_id)
        result = service.upload_and_persist(
            file_content=file_content,
            filename=filename,
            remove_invalid=True,
            progress_callback=progress_callback,
            should_stop=should_stop,
        )
        if job_manager.should_stop(job_id):
            return {}
        return self.build_upload_payload(result)

    # -------------------------------------------------------------------------
    def run_analysis_job(
        self,
        request_payload: dict[str, Any],
        job_manager: JobManager,
        job_id: str,
    ) -> dict[str, Any]:
        service = DatasetService()
        progress_callback = JobProgressReporter(job_manager, job_id)
        should_stop = JobStopChecker(job_manager, job_id)
        result = service.analyze_dataset(
            dataset_name=str(request_payload.get("dataset_name", "")),
            session_name=request_payload.get("session_name"),
            selected_metric_keys=request_payload.get("selected_metric_keys"),
            sampling=request_payload.get("sampling"),
            filters=request_payload.get("filters"),
            metric_parameters=request_payload.get("metric_parameters"),
            progress_callback=progress_callback,
            should_stop=should_stop,
        )
        if job_manager.should_stop(job_id):
            return {}
        return self.build_analysis_payload(result)
