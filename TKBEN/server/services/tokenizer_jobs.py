from __future__ import annotations

from typing import Any

from TKBEN.server.services.jobs import JobManager, JobProgressReporter, JobStopChecker
from TKBEN.server.services.tokenizers import TokenizersService


###############################################################################
class TokenizerJobService:
    # -------------------------------------------------------------------------
    def run_download_job(
        self,
        request_payload: dict[str, Any],
        job_manager: JobManager,
        job_id: str,
    ) -> dict[str, Any]:
        service = TokenizersService()
        progress_callback = JobProgressReporter(job_manager, job_id)
        should_stop = JobStopChecker(job_manager, job_id)
        tokenizers = request_payload.get("tokenizers", [])
        if not isinstance(tokenizers, list):
            tokenizers = []
        result = service.download_and_persist(
            tokenizers=tokenizers,
            progress_callback=progress_callback,
            should_stop=should_stop,
        )
        if job_manager.should_stop(job_id):
            return {}
        return result

    # -------------------------------------------------------------------------
    def run_report_job(
        self,
        request_payload: dict[str, Any],
        job_manager: JobManager,
        job_id: str,
    ) -> dict[str, Any]:
        service = TokenizersService()
        progress_callback = JobProgressReporter(job_manager, job_id)
        should_stop = JobStopChecker(job_manager, job_id)
        tokenizer_name = str(request_payload.get("tokenizer_name", "")).strip()
        result = service.generate_and_store_report(
            tokenizer_name=tokenizer_name,
            progress_callback=progress_callback,
            should_stop=should_stop,
        )
        if job_manager.should_stop(job_id):
            return {}
        return {"status": "success", **result}

    # -------------------------------------------------------------------------
    def upload_custom_tokenizer(
        self,
        file_content: bytes,
        normalized_filename: str,
        safe_stem: str,
    ) -> dict[str, Any]:
        service = TokenizersService()
        return service.register_custom_tokenizer_from_upload(
            file_content=file_content,
            normalized_filename=normalized_filename,
            safe_stem=safe_stem,
        )

    # -------------------------------------------------------------------------
    def clear_custom_tokenizers(self) -> None:
        TokenizersService().clear_custom_tokenizers()
