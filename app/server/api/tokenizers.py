from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from pydantic import ValidationError

from server.contracts.jobs import JobStartResponse
from server.contracts.tokenizers import (
    TokenizerDeleteResponse,
    SupportedTokenizerPipeline,
    TokenizerDownloadRequest,
    TokenizerDiscoveryQuery,
    TokenizerDiscoveryResponse,
    TokenizerDiscoverySort,
    TokenizerListItem,
    TokenizerListResponse,
    TokenizerReportGenerateRequest,
    TokenizerReportResponse,
    TokenizerSettingsResponse,
    TokenizerUploadResponse,
    TokenizerVocabularyPageResponse,
)
from server.configurations import get_server_settings
from server.common.constants import (
    API_ROUTE_TOKENIZERS_DELETE,
    API_ROUTE_TOKENIZERS_DOWNLOAD,
    API_ROUTE_TOKENIZERS_LIST,
    API_ROUTE_TOKENIZERS_REPORT_BY_ID,
    API_ROUTE_TOKENIZERS_REPORT_GENERATE,
    API_ROUTE_TOKENIZERS_REPORT_LATEST,
    API_ROUTE_TOKENIZERS_REPORT_VOCABULARY,
    API_ROUTE_TOKENIZERS_DISCOVER,
    API_ROUTE_TOKENIZERS_SETTINGS,
    API_ROUTE_TOKENIZERS_UPLOAD,
    API_ROUTER_PREFIX_TOKENIZERS,
)
from server.common.utils.logger import logger
from server.common.utils.security import (
    normalize_identifier,
)
from server.api.helpers import (
    ManagedJobHttpAdapter,
    read_upload_limited,
    validate_upload_filename,
)
from server.services.keys import (
    HFAccessKeyService,
    HFAccessKeyValidationError,
)
from server.services.tokenizer_jobs import TokenizerJobService
from server.services.tokenizer_reporting import TokenizerReportingService
from server.services.tokenizers import TokenizersService
from server.services.managed_jobs import (
    ManagedJobSpec,
)


router = APIRouter(prefix=API_ROUTER_PREFIX_TOKENIZERS, tags=["tokenizers"])

###############################################################################
@router.get(
    API_ROUTE_TOKENIZERS_SETTINGS,
    response_model=TokenizerSettingsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_tokenizer_settings() -> TokenizerSettingsResponse:
    return TokenizerSettingsResponse(
        default_discovery_limit=get_server_settings().tokenizers.default_discovery_limit,
        max_discovery_limit=get_server_settings().tokenizers.max_discovery_limit,
        max_discovery_candidates=get_server_settings().tokenizers.max_discovery_candidates,
        metadata_candidate_multiplier=get_server_settings().tokenizers.metadata_candidate_multiplier,
    )

###############################################################################
def _build_tokenizer_discovery_query(
    search: Annotated[str | None, Query(max_length=160)] = None,
    limit: Annotated[int | None, Query(ge=1, le=250)] = None,
    pipeline_tag: Annotated[SupportedTokenizerPipeline | None, Query()] = None,
    author: Annotated[str | None, Query(max_length=160)] = None,
    include_tags: Annotated[list[str] | None, Query()] = None,
    exclude_tags: Annotated[list[str] | None, Query()] = None,
    access: Annotated[Literal["all", "public", "gated"], Query()] = "all",
    sort: Annotated[TokenizerDiscoverySort, Query()] = TokenizerDiscoverySort.DOWNLOADS,
    vocabulary_operator: Annotated[Literal["at_least", "at_most"] | None, Query()] = None,
    vocabulary_size: Annotated[int | None, Query(ge=0)] = None,
    vocabulary_sort: Annotated[Literal["none", "ascending", "descending"], Query()] = "none",
) -> TokenizerDiscoveryQuery:
    settings = get_server_settings().tokenizers
    try:
        return TokenizerDiscoveryQuery(
            search=search,
            limit=settings.default_discovery_limit if limit is None else limit,
            pipeline_tag=pipeline_tag,
            author=author,
            include_tags=include_tags or [],
            exclude_tags=exclude_tags or [],
            access=access,
            sort=sort,
            vocabulary_operator=vocabulary_operator,
            vocabulary_size=vocabulary_size,
            vocabulary_sort=vocabulary_sort,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

###############################################################################
@router.get(
    API_ROUTE_TOKENIZERS_DISCOVER,
    response_model=TokenizerDiscoveryResponse,
    status_code=status.HTTP_200_OK,
)
async def discover_tokenizers(
    query: Annotated[TokenizerDiscoveryQuery, Depends(_build_tokenizer_discovery_query)],
) -> TokenizerDiscoveryResponse:
    logger.info("Discovering HuggingFace tokenizers (limit=%s)", query.limit)

    service = TokenizersService()
    try:
        response = await asyncio.to_thread(service.discover_tokenizers, query)

    except HFAccessKeyValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to discover tokenizers from HuggingFace")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to discover tokenizers from HuggingFace.",
        ) from exc

    return response

###############################################################################
@router.get(
    API_ROUTE_TOKENIZERS_LIST,
    response_model=TokenizerListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_tokenizers(
    search: Annotated[str | None, Query(max_length=160)] = None,
    source: Annotated[Literal["all", "huggingface", "custom"], Query()] = "all",
    vocabulary_size_operator: Annotated[Literal["at_least", "at_most"], Query()] = "at_least",
    vocabulary_size: Annotated[int | None, Query(ge=0)] = None,
) -> TokenizerListResponse:
    service = TokenizersService()
    tokenizers = await asyncio.to_thread(
        service.list_tokenizer_catalog,
        search=search,
        source=source,
        vocabulary_size_operator=vocabulary_size_operator,
        vocabulary_size=vocabulary_size,
    )
    return TokenizerListResponse(
        tokenizers=[TokenizerListItem.model_validate(item) for item in tokenizers],
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
        await asyncio.to_thread(key_service.get_active_key)
    except HFAccessKeyValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return ManagedJobHttpAdapter.start(
        request,
        ManagedJobSpec(
            job_type="tokenizer_download",
            runner=TokenizerJobService().run_download_job,
            kwargs={"request_payload": payload.model_dump()},
            conflict_detail="Tokenizer download is already in progress.",
            initialization_detail="Failed to initialize tokenizer download job.",
            message="Tokenizer download job started.",
        ),
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
    tokenizer_available = await asyncio.to_thread(
        service.has_available_tokenizer,
        tokenizer_name,
    )
    if not tokenizer_available:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Tokenizer '{tokenizer_name}' is not downloaded. "
                "Download it before generating a report."
            ),
        )

    return ManagedJobHttpAdapter.start(
        request,
        ManagedJobSpec(
            job_type="tokenizer_report",
            runner=TokenizerJobService().run_report_job,
            kwargs={"request_payload": payload.model_dump()},
            conflict_detail="Tokenizer report generation is already in progress.",
            initialization_detail="Failed to initialize tokenizer report job.",
            message="Tokenizer report job started.",
        ),
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

    service = TokenizerReportingService()
    report = await asyncio.to_thread(
        service.get_latest_tokenizer_report, tokenizer_name
    )
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
    service = TokenizerReportingService()
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
    service = TokenizerReportingService()
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
    normalized_filename, safe_stem = validate_upload_filename(
        file,
        extension_allowed=lambda extension: extension == ".json",
        unsupported_detail=lambda _extension: (
            "File must be a .json file (tokenizer.json)"
        ),
    )

    max_upload_bytes = int(get_server_settings().tokenizers.max_upload_bytes)
    try:
        content = await read_upload_limited(file, max_upload_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to read uploaded tokenizer file")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file.",
        ) from exc
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        result = await asyncio.to_thread(
            TokenizersService().register_custom_tokenizer_from_upload,
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
    API_ROUTE_TOKENIZERS_DELETE,
    response_model=TokenizerDeleteResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_tokenizer(
    tokenizer_name: Annotated[str, Query(min_length=1, max_length=160)],
) -> TokenizerDeleteResponse:
    try:
        normalized_name = normalize_identifier(
            tokenizer_name,
            "Tokenizer name",
            max_length=160,
        )
        removed = await asyncio.to_thread(
            TokenizersService().remove_downloaded_tokenizer,
            normalized_name,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tokenizer '{normalized_name}' is not downloaded.",
        )
    return TokenizerDeleteResponse(
        status="success",
        tokenizer_name=normalized_name,
        message="Tokenizer removed",
    )
