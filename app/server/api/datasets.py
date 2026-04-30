from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from TKBEN.server.domain.dataset import (
    DatasetAnalysisRequest,
    DatasetAnalysisResponse,
    DatasetDeleteResponse,
    DatasetDownloadRequest,
    DatasetListResponse,
    DatasetMetricCatalogResponse,
)
from TKBEN.server.domain.jobs import JobStartResponse
from TKBEN.server.configurations import get_server_settings
from TKBEN.server.common.utils.logger import logger
from TKBEN.server.common.utils.security import (
    normalize_identifier,
)
from TKBEN.server.common.constants import (
    API_ROUTE_DATASETS_ANALYZE,
    API_ROUTE_DATASETS_DELETE,
    API_ROUTE_DATASETS_DOWNLOAD,
    API_ROUTE_DATASETS_LIST,
    API_ROUTE_DATASETS_METRICS_CATALOG,
    API_ROUTE_DATASETS_REPORT_BY_ID,
    API_ROUTE_DATASETS_REPORT_LATEST,
    API_ROUTE_DATASETS_UPLOAD,
    API_ROUTER_PREFIX_DATASETS,
)
from TKBEN.server.api.helpers import (
    start_managed_job,
    validate_upload_filename,
    validate_upload_size,
)
from TKBEN.server.services.dataset_jobs import DatasetJobService
from TKBEN.server.services.datasets import DatasetService

router = APIRouter(prefix=API_ROUTER_PREFIX_DATASETS, tags=["datasets"])


###############################################################################
@router.get(
    API_ROUTE_DATASETS_LIST,
    response_model=DatasetListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_datasets() -> DatasetListResponse:
    service = DatasetService()
    datasets = await asyncio.to_thread(service.get_dataset_previews)
    return DatasetListResponse(datasets=datasets)


###############################################################################
@router.get(
    API_ROUTE_DATASETS_METRICS_CATALOG,
    response_model=DatasetMetricCatalogResponse,
    status_code=status.HTTP_200_OK,
)
async def get_dataset_metrics_catalog() -> DatasetMetricCatalogResponse:
    service = DatasetService()
    categories = await asyncio.to_thread(service.get_metric_catalog)
    return DatasetMetricCatalogResponse(categories=categories)


###############################################################################
@router.post(
    API_ROUTE_DATASETS_DOWNLOAD,
    response_model=JobStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def download_dataset(
    request: Request,
    payload: DatasetDownloadRequest,
) -> JobStartResponse:
    logger.info(
        "Dataset download requested: corpus=%s, config=%s",
        payload.corpus,
        payload.configs.configuration,
    )

    return start_managed_job(
        request,
        job_type="dataset_download",
        runner=DatasetJobService().run_download_job,
        kwargs={
            "request_payload": payload.model_dump(),
        },
        conflict_detail="Dataset download is already in progress.",
        init_failure_detail="Failed to initialize dataset download job.",
        message="Dataset download job started.",
    )


###############################################################################
@router.post(
    API_ROUTE_DATASETS_UPLOAD,
    response_model=JobStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_custom_dataset(
    request: Request,
    file: UploadFile = File(..., description="CSV or Excel file to upload"),
) -> JobStartResponse:
    allowed_extensions = set(get_server_settings().datasets.allowed_extensions)
    normalized_filename, _ = validate_upload_filename(
        file,
        extension_allowed=allowed_extensions.__contains__,
        unsupported_detail=lambda extension: (
            f"Unsupported file type: {extension}. Use .csv, .xlsx, or .xls"
        ),
        validate_stem_before_extension=True,
    )

    logger.info("Custom dataset upload requested: filename=%s", normalized_filename)

    try:
        file_content = await file.read()
    except Exception as exc:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file.",
        ) from exc
    if not file_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    max_upload_bytes = int(get_server_settings().datasets.max_upload_bytes)
    validate_upload_size(file_content, max_upload_bytes)

    return start_managed_job(
        request,
        job_type="dataset_upload",
        runner=DatasetJobService().run_upload_job,
        kwargs={
            "file_content": file_content,
            "filename": normalized_filename,
        },
        conflict_detail="Dataset upload is already in progress.",
        init_failure_detail="Failed to initialize dataset upload job.",
        message="Custom dataset upload job started.",
    )


###############################################################################
@router.post(
    API_ROUTE_DATASETS_ANALYZE,
    response_model=JobStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def analyze_dataset(
    request: Request,
    payload: DatasetAnalysisRequest,
) -> JobStartResponse:
    logger.info("Dataset validation requested: dataset=%s", payload.dataset_name)

    job_manager = request.app.state.job_manager
    if job_manager.is_job_running("dataset_validation"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Dataset validation is already in progress.",
        )

    service = DatasetService()
    if not service.is_dataset_in_database(payload.dataset_name):
        logger.warning("Dataset not found: %s", payload.dataset_name)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{payload.dataset_name}' not found. Please load it first.",
        )

    return start_managed_job(
        request,
        job_type="dataset_validation",
        runner=DatasetJobService().run_analysis_job,
        kwargs={
            "request_payload": payload.model_dump(),
        },
        conflict_detail="Dataset validation is already in progress.",
        init_failure_detail="Failed to initialize dataset validation job.",
        message="Dataset validation job started.",
        check_conflict=False,
    )


###############################################################################
@router.get(
    API_ROUTE_DATASETS_REPORT_LATEST,
    response_model=DatasetAnalysisResponse,
    status_code=status.HTTP_200_OK,
)
async def get_latest_dataset_report(dataset_name: str) -> DatasetAnalysisResponse:
    try:
        dataset_name = normalize_identifier(
            dataset_name, "Dataset name", max_length=200
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    service = DatasetService()
    report = await asyncio.to_thread(service.get_latest_validation_report, dataset_name)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No validation report found for dataset '{dataset_name}'.",
        )
    return DatasetAnalysisResponse(status="success", **report)


###############################################################################
@router.get(
    API_ROUTE_DATASETS_REPORT_BY_ID,
    response_model=DatasetAnalysisResponse,
    status_code=status.HTTP_200_OK,
)
async def get_dataset_report_by_id(report_id: int) -> DatasetAnalysisResponse:
    service = DatasetService()
    report = await asyncio.to_thread(service.get_validation_report_by_id, report_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset validation report '{report_id}' not found.",
        )
    return DatasetAnalysisResponse(status="success", **report)


###############################################################################
@router.delete(
    API_ROUTE_DATASETS_DELETE,
    response_model=DatasetDeleteResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_dataset(dataset_name: str) -> DatasetDeleteResponse:
    try:
        dataset_name = normalize_identifier(
            dataset_name, "Dataset name", max_length=200
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    service = DatasetService()
    if not service.is_dataset_in_database(dataset_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_name}' not found.",
        )

    await asyncio.to_thread(service.remove_dataset, dataset_name)
    return DatasetDeleteResponse(
        status="success",
        dataset_name=dataset_name,
        message="Dataset removed.",
    )
