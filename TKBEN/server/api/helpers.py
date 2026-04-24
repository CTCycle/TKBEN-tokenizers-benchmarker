from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, UploadFile, status

from TKBEN.server.common.utils.security import normalize_upload_stem
from TKBEN.server.configurations import get_server_settings
from TKBEN.server.domain.jobs import JobStartResponse


def start_managed_job(
    request: Request,
    *,
    job_type: str,
    runner: Callable[..., Any],
    kwargs: dict[str, Any],
    conflict_detail: str,
    init_failure_detail: str,
    message: str,
    check_conflict: bool = True,
) -> JobStartResponse:
    job_manager = request.app.state.job_manager
    if check_conflict and job_manager.is_job_running(job_type):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=conflict_detail)

    job_id = job_manager.start_job(
        job_type=job_type,
        runner=runner,
        kwargs={**kwargs, "job_manager": job_manager},
    )
    job_status = job_manager.get_job_status(job_id)
    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=init_failure_detail,
        )

    return JobStartResponse(
        job_id=job_id,
        job_type=job_status["job_type"],
        status=job_status["status"],
        message=message,
        poll_interval=get_server_settings().jobs.polling_interval,
    )


def validate_upload_filename(
    file: UploadFile,
    *,
    extension_allowed: Callable[[str], bool],
    unsupported_detail: Callable[[str], str],
    validate_stem_before_extension: bool = False,
) -> tuple[str, str]:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    normalized_filename = os.path.basename(file.filename.strip().replace("\\", "/"))
    if not normalized_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    safe_stem = ""
    if validate_stem_before_extension:
        try:
            safe_stem = normalize_upload_stem(normalized_filename)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    extension = os.path.splitext(normalized_filename)[1].lower()
    if not extension_allowed(extension):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=unsupported_detail(extension),
        )

    if not validate_stem_before_extension:
        try:
            safe_stem = normalize_upload_stem(normalized_filename)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    return normalized_filename, safe_stem


def validate_upload_size(content: bytes, max_upload_bytes: int) -> None:
    if len(content) > max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds max allowed size ({max_upload_bytes} bytes).",
        )
