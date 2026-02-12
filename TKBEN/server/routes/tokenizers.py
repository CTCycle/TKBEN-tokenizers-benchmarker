from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

import anyio
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from tokenizers import Tokenizer

from TKBEN.server.entities.tokenizers import (
    TokenizerDownloadRequest,
    TokenizerListItem,
    TokenizerListResponse,
    TokenizerScanResponse,
    TokenizerSettingsResponse,
    TokenizerUploadResponse,
)
from TKBEN.server.entities.jobs import JobStartResponse
from TKBEN.server.configurations.server import server_settings
from TKBEN.server.common.constants import (
    API_ROUTE_TOKENIZERS_CUSTOM,
    API_ROUTE_TOKENIZERS_DOWNLOAD,
    API_ROUTE_TOKENIZERS_LIST,
    API_ROUTE_TOKENIZERS_SCAN,
    API_ROUTE_TOKENIZERS_SETTINGS,
    API_ROUTE_TOKENIZERS_UPLOAD,
    API_ROUTER_PREFIX_TOKENIZERS,
)
from TKBEN.server.common.utils.logger import logger
from TKBEN.server.services.jobs import JobProgressReporter, JobStopChecker, job_manager
from TKBEN.server.services.benchmarks import BenchmarkTools
from TKBEN.server.services.tokenizers import TokenizersService

router = APIRouter(prefix=API_ROUTER_PREFIX_TOKENIZERS, tags=["tokenizers"])


###############################################################################
@router.get(
    API_ROUTE_TOKENIZERS_SETTINGS,
    response_model=TokenizerSettingsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_tokenizer_settings() -> TokenizerSettingsResponse:
    """
    Get tokenizer configuration settings.

    Returns:
        TokenizerSettingsResponse with default, min, and max scan limits.
    """
    return TokenizerSettingsResponse(
        default_scan_limit=server_settings.tokenizers.default_scan_limit,
        max_scan_limit=server_settings.tokenizers.max_scan_limit,
        min_scan_limit=server_settings.tokenizers.min_scan_limit,
    )


###############################################################################
@router.get(
    API_ROUTE_TOKENIZERS_SCAN,
    response_model=TokenizerScanResponse,
    status_code=status.HTTP_200_OK,
)
async def scan_tokenizers(
    limit: int = Query(default=None),
    hf_access_token: str | None = Query(default=None),
) -> TokenizerScanResponse:
    """
    Scan HuggingFace for the most popular tokenizer identifiers.

    Args:
        limit: Maximum number of tokenizers to fetch. Defaults to configured value.
        hf_access_token: Optional HuggingFace access token for authenticated requests.

    Returns:
        TokenizerScanResponse containing the list of tokenizer identifiers.
    """
    # Use configured defaults if limit not provided, and clamp to configured bounds
    min_limit = server_settings.tokenizers.min_scan_limit
    max_limit = server_settings.tokenizers.max_scan_limit
    default_limit = server_settings.tokenizers.default_scan_limit

    if limit is None:
        limit = default_limit
    else:
        limit = max(min_limit, min(limit, max_limit))

    logger.info("Scanning HuggingFace for tokenizers (limit=%s)", limit)

    service = TokenizersService(hf_access_token)

    try:
        identifiers = await asyncio.to_thread(service.get_tokenizer_identifiers, limit)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to scan tokenizers from HuggingFace")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tokenizers from HuggingFace.",
        ) from exc

    logger.info("Successfully fetched %s tokenizer identifiers", len(identifiers))
    return TokenizerScanResponse(
        status="success",
        identifiers=identifiers,
        count=len(identifiers),
    )


###############################################################################
@router.get(
    API_ROUTE_TOKENIZERS_LIST,
    response_model=TokenizerListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_tokenizers() -> TokenizerListResponse:
    service = TokenizersService()
    tokenizers = await asyncio.to_thread(service.list_downloaded_tokenizers)
    return TokenizerListResponse(
        tokenizers=[TokenizerListItem(tokenizer_name=name) for name in tokenizers],
        count=len(tokenizers),
    )


###############################################################################
def run_tokenizer_download_job(
    request_payload: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    raw_token = request_payload.get("hf_access_token")
    hf_access_token = raw_token if isinstance(raw_token, str) and raw_token else None
    service = TokenizersService(hf_access_token=hf_access_token)
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


###############################################################################
@router.post(
    API_ROUTE_TOKENIZERS_DOWNLOAD,
    response_model=JobStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def download_tokenizers(request: TokenizerDownloadRequest) -> JobStartResponse:
    if not request.tokenizers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one tokenizer must be specified.",
        )

    if job_manager.is_job_running("tokenizer_download"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Tokenizer download is already in progress.",
        )

    request_payload = request.model_dump()
    job_id = job_manager.start_job(
        job_type="tokenizer_download",
        runner=run_tokenizer_download_job,
        kwargs={"request_payload": request_payload},
    )

    job_status = job_manager.get_job_status(job_id)
    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize tokenizer download job.",
        )

    return JobStartResponse(
        job_id=job_id,
        job_type=job_status["job_type"],
        status=job_status["status"],
        message="Tokenizer download job started.",
        poll_interval=server_settings.jobs.polling_interval,
    )


###############################################################################
# Store uploaded custom tokenizers in memory for the session
custom_tokenizers: dict[str, Tokenizer] = {}


def get_custom_tokenizers() -> dict[str, Tokenizer]:
    """Get the currently loaded custom tokenizers."""
    return custom_tokenizers


def clear_custom_tokenizers() -> None:
    """Clear all custom tokenizers."""
    custom_tokenizers.clear()


def _create_temp_tokenizer_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    return path


def _unlink_temp_tokenizer_path(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


###############################################################################
@router.post(
    API_ROUTE_TOKENIZERS_UPLOAD,
    response_model=TokenizerUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_custom_tokenizer(
    file: UploadFile = File(...),
) -> TokenizerUploadResponse:
    """
    Upload a custom tokenizer.json file.

    The tokenizer will be loaded and validated for compatibility.
    If valid, it will be stored and can be used in benchmark runs.

    Args:
        file: The tokenizer.json file to upload.

    Returns:
        TokenizerUploadResponse with the assigned tokenizer name.
    """
    # Validate file extension
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a .json file (tokenizer.json)",
        )

    # Read file content
    content = await file.read()

    # Save to temp file and load
    tmp_path = None
    try:
        tmp_path = await anyio.to_thread.run_sync(_create_temp_tokenizer_path)
        async with await anyio.open_file(tmp_path, "wb") as tmp:
            await tmp.write(content)

        tokenizer = await anyio.to_thread.run_sync(Tokenizer.from_file, tmp_path)
    except Exception as exc:
        logger.warning("Failed to load tokenizer from uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load tokenizer: {exc}",
        ) from exc
    finally:
        if tmp_path:
            await anyio.to_thread.run_sync(_unlink_temp_tokenizer_path, tmp_path)

    # Check compatibility
    tools = BenchmarkTools()
    is_compatible = tools.is_tokenizer_compatible(tokenizer)

    # Generate name from filename
    base_name = os.path.splitext(file.filename)[0]
    tokenizer_name = f"CUSTOM_{base_name}"

    if is_compatible:
        custom_tokenizers[tokenizer_name] = tokenizer
        logger.info("Loaded custom tokenizer: %s", tokenizer_name)
    else:
        logger.warning("Custom tokenizer %s is not compatible", tokenizer_name)

    return TokenizerUploadResponse(
        status="success",
        tokenizer_name=tokenizer_name,
        is_compatible=is_compatible,
    )


###############################################################################
@router.delete(
    API_ROUTE_TOKENIZERS_CUSTOM,
    status_code=status.HTTP_200_OK,
)
async def delete_custom_tokenizers() -> dict:
    """
    Clear all uploaded custom tokenizers.

    Returns:
        Status message.
    """
    clear_custom_tokenizers()
    return {"status": "success", "message": "Custom tokenizers cleared"}

