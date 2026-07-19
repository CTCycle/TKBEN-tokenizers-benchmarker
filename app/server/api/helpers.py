from __future__ import annotations

from pathlib import Path, PurePosixPath
from collections.abc import Callable

from fastapi import HTTPException, Request, UploadFile, status

from server.common.utils.security import normalize_upload_stem
from server.domain.jobs import JobStartResponse
from server.services.managed_jobs import (
    ManagedJobConflictError,
    ManagedJobInitializationError,
    ManagedJobService,
    ManagedJobSpec,
)

UPLOAD_CHUNK_SIZE = 1024 * 1024

###############################################################################
class ManagedJobHttpAdapter:
    """Maps service-level job lifecycle failures to HTTP responses."""

    # -------------------------------------------------------------------------
    @staticmethod
    def start(request: Request, spec: ManagedJobSpec) -> JobStartResponse:
        try:
            return ManagedJobService().start(request.app.state.job_manager, spec)
        except ManagedJobConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        except ManagedJobInitializationError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc

###############################################################################
def validate_upload_filename(
    file: UploadFile,
    *,
    extension_allowed: Callable[[str], bool],
    unsupported_detail: Callable[[str], str],
    validate_stem_before_extension: bool = False,
) -> tuple[str, str]:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided.")

    normalized_filename = PurePosixPath(file.filename.strip().replace("\\", "/")).name
    if not normalized_filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename.")

    safe_stem = ""
    if validate_stem_before_extension:
        safe_stem = _normalize_upload_stem(normalized_filename)
    extension = Path(normalized_filename).suffix.lower()
    if not extension_allowed(extension):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=unsupported_detail(extension),
        )
    if not validate_stem_before_extension:
        safe_stem = _normalize_upload_stem(normalized_filename)
    return normalized_filename, safe_stem

###############################################################################
def _normalize_upload_stem(filename: str) -> str:
    try:
        return normalize_upload_stem(filename)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

###############################################################################
async def read_upload_limited(
    file: UploadFile,
    max_upload_bytes: int,
    *,
    chunk_size: int = UPLOAD_CHUNK_SIZE,
) -> bytes:
    chunks: list[bytes] = []
    total_size = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Uploaded file exceeds max allowed size ({max_upload_bytes} bytes).",
            )
        chunks.append(chunk)
    return b"".join(chunks)

###############################################################################
def validate_upload_size(content: bytes, max_upload_bytes: int) -> None:
    if len(content) > max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded file exceeds max allowed size ({max_upload_bytes} bytes).",
        )
