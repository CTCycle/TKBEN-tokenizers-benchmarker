from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from TKBEN.server.entities.jobs import JobCancelResponse, JobListResponse, JobStatusResponse
from TKBEN.server.common.constants import API_ROUTE_JOBS_STATUS, API_ROUTER_PREFIX_JOBS
from TKBEN.server.services.jobs import job_manager


router = APIRouter(prefix=API_ROUTER_PREFIX_JOBS, tags=["jobs"])


###############################################################################
@router.get(
    "",
    response_model=JobListResponse,
    status_code=status.HTTP_200_OK,
)
def list_jobs(job_type: str | None = Query(default=None)) -> JobListResponse:
    jobs = job_manager.list_jobs(job_type=job_type)
    return JobListResponse(jobs=[JobStatusResponse(**job) for job in jobs])


###############################################################################
@router.get(
    API_ROUTE_JOBS_STATUS,
    response_model=JobStatusResponse,
    status_code=status.HTTP_200_OK,
)
def get_job_status(job_id: str) -> JobStatusResponse:
    job_status = job_manager.get_job_status(job_id)
    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    return JobStatusResponse(**job_status)


###############################################################################
@router.delete(
    API_ROUTE_JOBS_STATUS,
    response_model=JobCancelResponse,
    status_code=status.HTTP_200_OK,
)
def cancel_job(job_id: str) -> JobCancelResponse:
    job_status = job_manager.get_job_status(job_id)
    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    success = job_manager.cancel_job(job_id)

    return JobCancelResponse(
        job_id=job_id,
        success=success,
        message="Cancellation requested" if success else "Job cannot be cancelled",
    )

