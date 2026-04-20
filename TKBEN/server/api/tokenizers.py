from __future__ import annotations

import asyncio
import os
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status

from TKBEN.server.domain.jobs import JobStartResponse
from TKBEN.server.domain.tokenizers import (
    CustomTokenizersDeleteResponse,
    TokenizerDownloadRequest,
    TokenizerListItem,
    TokenizerListResponse,
    TokenizerReportGenerateRequest,
    TokenizerReportResponse,
    TokenizerScanResponse,
    TokenizerSettingsResponse,
    TokenizerUploadResponse,
    TokenizerVocabularyPageResponse,
)
from TKBEN.server.configurations import get_server_settings
from TKBEN.server.common.constants import (
    API_ROUTE_TOKENIZERS_CUSTOM,
    API_ROUTE_TOKENIZERS_DOWNLOAD,
    API_ROUTE_TOKENIZERS_LIST,
    API_ROUTE_TOKENIZERS_REPORT_BY_ID,
    API_ROUTE_TOKENIZERS_REPORT_GENERATE,
    API_ROUTE_TOKENIZERS_REPORT_LATEST,
    API_ROUTE_TOKENIZERS_REPORT_VOCABULARY,
    API_ROUTE_TOKENIZERS_SCAN,
    API_ROUTE_TOKENIZERS_SETTINGS,
    API_ROUTE_TOKENIZERS_UPLOAD,
    API_ROUTER_PREFIX_TOKENIZERS,
)
from TKBEN.server.common.utils.logger import logger
from TKBEN.server.common.utils.security import (
    normalize_identifier,
    normalize_upload_stem,
)
from TKBEN.server.services.keys import (
    HFAccessKeyService,
    HFAccessKeyValidationError,
)
from TKBEN.server.services.tokenizer_jobs import TokenizerJobService
from TKBEN.server.services.tokenizers import TokenizersService


router = APIRouter(prefix=API_ROUTER_PREFIX_TOKENIZERS, tags=["tokenizers"])
tokenizer_job_service = TokenizerJobService()


###############################################################################
@router.get(
    API_ROUTE_TOKENIZERS_SETTINGS,
    response_model=TokenizerSettingsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_tokenizer_settings() -> TokenizerSettingsResponse:
    return TokenizerSettingsResponse(
        default_scan_limit=get_server_settings().tokenizers.default_scan_limit,
        max_scan_limit=get_server_settings().tokenizers.max_scan_limit,
        min_scan_limit=get_server_settings().tokenizers.min_scan_limit,
    )


###############################################################################
@router.get(
    API_ROUTE_TOKENIZERS_SCAN,
    response_model=TokenizerScanResponse,
    status_code=status.HTTP_200_OK,
)
async def scan_tokenizers(
    limit: Annotated[int | None, Query()] = None,
) -> TokenizerScanResponse:
    min_limit = get_server_settings().tokenizers.min_scan_limit
    max_limit = get_server_settings().tokenizers.max_scan_limit
    default_limit = get_server_settings().tokenizers.default_scan_limit

    if limit is None:
        limit = default_limit
    else:
        limit = max(min_limit, min(limit, max_limit))

    logger.info("Scanning HuggingFace for tokenizers (limit=%s)", limit)

    service = TokenizersService()
    try:
        identifiers = await asyncio.to_thread(service.get_tokenizer_identifiers, limit)
    except HFAccessKeyValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to scan tokenizers from HuggingFace")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tokenizers from HuggingFace.",
        ) from exc

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
@router.post(
    API_ROUTE_TOKENIZERS_DOWNLOAD,
    response_model=JobStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def download_tokenizers(
    request: Request,
    payload: TokenizerDownloadRequest,
) -> JobStartResponse:
    if not payload.tokenizers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one tokenizer must be specified.",
        )

    key_service = HFAccessKeyService()
    try:
        key_service.get_active_key()
    except HFAccessKeyValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    job_manager = request.app.state.job_manager
    if job_manager.is_job_running("tokenizer_download"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Tokenizer download is already in progress.",
        )

    request_payload = payload.model_dump()
    job_id = job_manager.start_job(
        job_type="tokenizer_download",
        runner=tokenizer_job_service.run_download_job,
        kwargs={
            "request_payload": request_payload,
            "job_manager": job_manager,
        },
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
        poll_interval=get_server_settings().jobs.polling_interval,
    )


###############################################################################
@router.post(
    API_ROUTE_TOKENIZERS_REPORT_GENERATE,
    response_model=JobStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_tokenizer_report(
    request: Request,
    payload: TokenizerReportGenerateRequest,
) -> JobStartResponse:
    tokenizer_name = payload.tokenizer_name.strip()
    if not tokenizer_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tokenizer name must be specified.",
        )

    service = TokenizersService()
    if not service.resolve_cached_tokenizer_existence(tokenizer_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Tokenizer '{tokenizer_name}' is not downloaded. "
                "Download it before generating a report."
            ),
        )

    job_manager = request.app.state.job_manager
    if job_manager.is_job_running("tokenizer_report"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Tokenizer report generation is already in progress.",
        )

    request_payload = payload.model_dump()
    job_id = job_manager.start_job(
        job_type="tokenizer_report",
        runner=tokenizer_job_service.run_report_job,
        kwargs={
            "request_payload": request_payload,
            "job_manager": job_manager,
        },
    )

    job_status = job_manager.get_job_status(job_id)
    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize tokenizer report job.",
        )

    return JobStartResponse(
        job_id=job_id,
        job_type=job_status["job_type"],
        status=job_status["status"],
        message="Tokenizer report job started.",
        poll_interval=get_server_settings().jobs.polling_interval,
    )


###############################################################################
@router.get(
    API_ROUTE_TOKENIZERS_REPORT_LATEST,
    response_model=TokenizerReportResponse,
    status_code=status.HTTP_200_OK,
)
async def get_latest_tokenizer_report(tokenizer_name: str) -> TokenizerReportResponse:
    try:
        tokenizer_name = normalize_identifier(
            tokenizer_name,
            "Tokenizer name",
            max_length=160,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    service = TokenizersService()
    report = await asyncio.to_thread(service.get_latest_tokenizer_report, tokenizer_name)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No tokenizer report found for '{tokenizer_name}'.",
        )
    return TokenizerReportResponse(status="success", **report)


###############################################################################
@router.get(
    API_ROUTE_TOKENIZERS_REPORT_BY_ID,
    response_model=TokenizerReportResponse,
    status_code=status.HTTP_200_OK,
)
async def get_tokenizer_report_by_id(report_id: int) -> TokenizerReportResponse:
    service = TokenizersService()
    report = await asyncio.to_thread(service.get_tokenizer_report_by_id, report_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tokenizer report '{report_id}' not found.",
        )
    return TokenizerReportResponse(status="success", **report)


###############################################################################
@router.get(
    API_ROUTE_TOKENIZERS_REPORT_VOCABULARY,
    response_model=TokenizerVocabularyPageResponse,
    status_code=status.HTTP_200_OK,
)
async def get_tokenizer_report_vocabulary(
    report_id: int,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
) -> TokenizerVocabularyPageResponse:
    service = TokenizersService()
    page = await asyncio.to_thread(
        service.get_tokenizer_report_vocabulary,
        report_id,
        offset,
        limit,
    )
    if page is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tokenizer report '{report_id}' not found.",
        )
    return TokenizerVocabularyPageResponse(status="success", **page)


###############################################################################
@router.post(
    API_ROUTE_TOKENIZERS_UPLOAD,
    response_model=TokenizerUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_custom_tokenizer(
    file: Annotated[UploadFile, File(...)],
) -> TokenizerUploadResponse:
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
    if not normalized_filename.lower().endswith(".json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a .json file (tokenizer.json)",
        )

    try:
        safe_stem = normalize_upload_stem(normalized_filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    max_upload_bytes = int(get_server_settings().tokenizers.max_upload_bytes)
    if len(content) > max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds max allowed size ({max_upload_bytes} bytes).",
        )

    try:
        result = await asyncio.to_thread(
            tokenizer_job_service.upload_custom_tokenizer,
            content,
            normalized_filename,
            safe_stem,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return TokenizerUploadResponse(**result)


###############################################################################
@router.delete(
    API_ROUTE_TOKENIZERS_CUSTOM,
    response_model=CustomTokenizersDeleteResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_custom_tokenizers() -> CustomTokenizersDeleteResponse:
    await asyncio.to_thread(tokenizer_job_service.clear_custom_tokenizers)
    return CustomTokenizersDeleteResponse(
        status="success",
        message="Custom tokenizers cleared",
    )
