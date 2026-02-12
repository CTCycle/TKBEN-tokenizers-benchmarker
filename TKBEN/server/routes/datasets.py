from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from TKBEN.server.entities.dataset import (
    DatasetAnalysisRequest,
    DatasetDownloadRequest,
    DatasetListResponse,
)
from TKBEN.server.entities.jobs import JobStartResponse
from TKBEN.server.configurations import server_settings
from TKBEN.server.common.utils.logger import logger
from TKBEN.server.common.constants import (
    API_ROUTE_DATASETS_ANALYZE,
    API_ROUTE_DATASETS_DELETE,
    API_ROUTE_DATASETS_DOWNLOAD,
    API_ROUTE_DATASETS_LIST,
    API_ROUTE_DATASETS_UPLOAD,
    API_ROUTER_PREFIX_DATASETS,
)
from TKBEN.server.services.jobs import JobProgressReporter, JobStopChecker, job_manager
from TKBEN.server.services.datasets import DatasetService

router = APIRouter(prefix=API_ROUTER_PREFIX_DATASETS, tags=["datasets"])


###############################################################################
class DatasetJobHandler:
    def build_histogram_payload(self, histogram_data: dict[str, Any]) -> dict[str, Any]:
        return {
            "bins": histogram_data.get("bins", []),
            "counts": histogram_data.get("counts", []),
            "bin_edges": histogram_data.get("bin_edges", []),
            "min_length": histogram_data.get("min_length", 0),
            "max_length": histogram_data.get("max_length", 0),
            "mean_length": histogram_data.get("mean_length", 0.0),
            "median_length": histogram_data.get("median_length", 0.0),
        }

    # -------------------------------------------------------------------------
    def build_dataset_mutation_payload(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "success",
            "dataset_name": result.get("dataset_name", ""),
            "text_column": result.get("text_column", ""),
            "document_count": result.get("document_count", 0),
            "saved_count": result.get("saved_count", 0),
            "histogram": self.build_histogram_payload(result.get("histogram", {})),
        }

    # -------------------------------------------------------------------------
    def build_download_payload(self, result: dict[str, Any]) -> dict[str, Any]:
        return self.build_dataset_mutation_payload(result)

    # -------------------------------------------------------------------------
    def build_upload_payload(self, result: dict[str, Any]) -> dict[str, Any]:
        return self.build_dataset_mutation_payload(result)

    # -------------------------------------------------------------------------
    def extract_configuration(self, request_payload: dict[str, Any]) -> str | None:
        configs = request_payload.get("configs")
        if isinstance(configs, dict):
            value = configs.get("configuration")
            if isinstance(value, str) and value.strip():
                return value.strip()
            return None
        return None

    # -------------------------------------------------------------------------
    def build_analysis_payload(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "success",
            "dataset_name": result.get("dataset_name", ""),
            "document_count": result.get("document_count", 0),
            "document_length_histogram": self.build_histogram_payload(
                result.get("document_length_histogram", {})
            ),
            "word_length_histogram": self.build_histogram_payload(
                result.get("word_length_histogram", {})
            ),
            "min_document_length": result.get("min_document_length", 0),
            "max_document_length": result.get("max_document_length", 0),
            "most_common_words": result.get("most_common_words", []),
            "least_common_words": result.get("least_common_words", []),
        }

    # -------------------------------------------------------------------------
    def run_download_job(
        self,
        request_payload: dict[str, Any],
        job_id: str,
    ) -> dict[str, Any]:
        service = DatasetService()
        progress_callback = JobProgressReporter(job_manager, job_id)
        should_stop = JobStopChecker(job_manager, job_id)
        result = service.download_and_persist(
            corpus=request_payload.get("corpus", ""),
            config=self.extract_configuration(request_payload),
            remove_invalid=True,
            progress_callback=progress_callback,
            should_stop=should_stop,
            job_id=job_id,
        )
        if job_manager.should_stop(job_id):
            return {}
        return self.build_download_payload(result)

    # -------------------------------------------------------------------------
    def run_upload_job(
        self,
        file_content: bytes,
        filename: str,
        job_id: str,
    ) -> dict[str, Any]:
        service = DatasetService()
        progress_callback = JobProgressReporter(job_manager, job_id)
        should_stop = JobStopChecker(job_manager, job_id)
        result = service.upload_and_persist(
            file_content=file_content,
            filename=filename,
            remove_invalid=True,
            progress_callback=progress_callback,
            should_stop=should_stop,
        )
        if job_manager.should_stop(job_id):
            return {}
        return self.build_upload_payload(result)

    # -------------------------------------------------------------------------
    def run_analysis_job(
        self,
        dataset_name: str,
        job_id: str,
    ) -> dict[str, Any]:
        service = DatasetService()
        progress_callback = JobProgressReporter(job_manager, job_id)
        should_stop = JobStopChecker(job_manager, job_id)
        result = service.analyze_dataset(
            dataset_name=dataset_name,
            progress_callback=progress_callback,
            should_stop=should_stop,
        )
        if job_manager.should_stop(job_id):
            return {}
        return self.build_analysis_payload(result)


dataset_job_handler = DatasetJobHandler()


###############################################################################
@router.get(
    API_ROUTE_DATASETS_LIST,
    response_model=DatasetListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_datasets() -> DatasetListResponse:
    """
    List all available datasets in the database.

    Returns:
        DatasetListResponse with list of dataset names.
    """
    service = DatasetService()
    datasets = await asyncio.to_thread(service.get_dataset_previews)
    return DatasetListResponse(datasets=datasets)


###############################################################################
@router.post(
    API_ROUTE_DATASETS_DOWNLOAD,
    response_model=JobStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def download_dataset(request: DatasetDownloadRequest) -> JobStartResponse:
    """
    Download a text dataset from HuggingFace and save it to the database.

    This endpoint fetches the specified dataset, removes empty or invalid
    documents, computes document-length statistics, and persists the cleaned
    texts to the database for tokenizer benchmarking.

    Args:
        request: DatasetDownloadRequest containing the corpus, config, and options.

    Returns:
        DatasetDownloadResponse with download statistics and histogram data.
    """
    logger.info(
        "Dataset download requested: corpus=%s, config=%s",
        request.corpus,
        request.configs.configuration,
    )

    if job_manager.is_job_running("dataset_download"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Dataset download is already in progress.",
        )

    request_payload = request.model_dump()
    job_id = job_manager.start_job(
        job_type="dataset_download",
        runner=dataset_job_handler.run_download_job,
        kwargs={
            "request_payload": request_payload,
        },
    )

    job_status = job_manager.get_job_status(job_id)
    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize dataset download job.",
        )

    return JobStartResponse(
        job_id=job_id,
        job_type=job_status["job_type"],
        status=job_status["status"],
        message="Dataset download job started.",
        poll_interval=server_settings.jobs.polling_interval,
    )


###############################################################################
@router.post(
    API_ROUTE_DATASETS_UPLOAD,
    response_model=JobStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_custom_dataset(
    file: UploadFile = File(..., description="CSV or Excel file to upload"),
) -> JobStartResponse:
    """
    Upload a CSV or Excel file and save it to the database as a custom dataset.

    This endpoint reads the uploaded file, extracts text from a suitable column,
    computes document-length statistics, and persists the cleaned texts to the
    database for tokenizer benchmarking.

    Args:
        file: Uploaded CSV (.csv) or Excel (.xlsx, .xls) file.

    Returns:
        CustomDatasetUploadResponse with upload statistics and histogram data.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )
    extension = os.path.splitext(file.filename)[1].lower()
    allowed_extensions = set(server_settings.datasets.allowed_extensions)
    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {extension}. Use .csv, .xlsx, or .xls",
        )

    logger.info("Custom dataset upload requested: filename=%s", file.filename)

    # Read file content
    try:
        file_content = await file.read()
    except Exception as exc:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file.",
        ) from exc

    if job_manager.is_job_running("dataset_upload"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Dataset upload is already in progress.",
        )

    job_id = job_manager.start_job(
        job_type="dataset_upload",
        runner=dataset_job_handler.run_upload_job,
        kwargs={
            "file_content": file_content,
            "filename": file.filename,
        },
    )

    job_status = job_manager.get_job_status(job_id)
    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize dataset upload job.",
        )

    return JobStartResponse(
        job_id=job_id,
        job_type=job_status["job_type"],
        status=job_status["status"],
        message="Custom dataset upload job started.",
        poll_interval=server_settings.jobs.polling_interval,
    )


###############################################################################
@router.post(
    API_ROUTE_DATASETS_ANALYZE,
    response_model=JobStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def analyze_dataset(request: DatasetAnalysisRequest) -> JobStartResponse:
    """
    Validate a loaded dataset and compute document + word-level statistics.
    """
    logger.info("Dataset validation requested: dataset=%s", request.dataset_name)

    if job_manager.is_job_running("dataset_validation"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Dataset validation is already in progress.",
        )

    service = DatasetService()

    # Check if dataset exists
    if not service.is_dataset_in_database(request.dataset_name):
        logger.warning("Dataset not found: %s", request.dataset_name)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{request.dataset_name}' not found. Please load it first.",
        )

    job_id = job_manager.start_job(
        job_type="dataset_validation",
        runner=dataset_job_handler.run_analysis_job,
        kwargs={
            "dataset_name": request.dataset_name,
        },
    )

    job_status = job_manager.get_job_status(job_id)
    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize dataset validation job.",
        )

    return JobStartResponse(
        job_id=job_id,
        job_type=job_status["job_type"],
        status=job_status["status"],
        message="Dataset validation job started.",
        poll_interval=server_settings.jobs.polling_interval,
    )


###############################################################################
@router.delete(
    API_ROUTE_DATASETS_DELETE,
    status_code=status.HTTP_200_OK,
)
async def delete_dataset(dataset_name: str) -> dict[str, Any]:
    service = DatasetService()
    if not service.is_dataset_in_database(dataset_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_name}' not found.",
        )
    await asyncio.to_thread(service.remove_dataset, dataset_name)
    return {
        "status": "success",
        "dataset_name": dataset_name,
        "message": "Dataset removed.",
    }

