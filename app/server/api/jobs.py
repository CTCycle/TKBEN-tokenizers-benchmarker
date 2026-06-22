from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from server.domain.jobs import JobStatusResponse
from server.common.constants import API_ROUTE_JOBS_STATUS, API_ROUTER_PREFIX_JOBS

router = APIRouter(prefix=API_ROUTER_PREFIX_JOBS, tags=["jobs"])

###############################################################################
@router.get(
    API_ROUTE_JOBS_STATUS,
    response_model=JobStatusResponse,
    status_code=status.HTTP_200_OK,
)
def get_job_status(request: Request, job_id: str) -> JobStatusResponse:
    job_status = request.app.state.job_manager.get_job_status(job_id)
    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    return JobStatusResponse(**job_status)
