from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from server.configurations import get_server_settings
from server.contracts.jobs import JobStartResponse
from server.services.jobs import JobManager


###############################################################################
class ManagedJobError(Exception):
    """Base error for managed-job initialization decisions."""


###############################################################################
class ManagedJobConflictError(ManagedJobError):
    """Raised when another job of the same type is active."""


###############################################################################
class ManagedJobInitializationError(ManagedJobError):
    """Raised when a started job cannot be observed immediately."""


###############################################################################
@dataclass(frozen=True)
class ManagedJobSpec:
    job_type: str
    runner: Callable[..., dict[str, Any]]
    kwargs: dict[str, Any]
    conflict_detail: str
    initialization_detail: str
    message: str
    check_conflict: bool = True


###############################################################################
class ManagedJobService:
    """Owns shared conflict, start, and initialization handling for jobs."""

    # -------------------------------------------------------------------------
    def start(self, job_manager: JobManager, spec: ManagedJobSpec) -> JobStartResponse:
        if spec.check_conflict and job_manager.is_job_running(spec.job_type):
            raise ManagedJobConflictError(spec.conflict_detail)

        job_id = job_manager.start_job(
            job_type=spec.job_type,
            runner=spec.runner,
            kwargs={**spec.kwargs, "job_manager": job_manager},
        )
        job_status = job_manager.get_job_status(job_id)
        if job_status is None:
            raise ManagedJobInitializationError(spec.initialization_detail)

        return JobStartResponse(
            job_id=job_id,
            job_type=job_status["job_type"],
            status=job_status["status"],
            message=spec.message,
            poll_interval=get_server_settings().jobs.polling_interval,
        )
