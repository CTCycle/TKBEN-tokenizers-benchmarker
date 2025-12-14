from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from TKBEN.server.schemas.dataset import (
    CustomDatasetUploadResponse,
    DatasetAnalysisRequest,
    DatasetAnalysisResponse,
    DatasetDownloadRequest,
    DatasetDownloadResponse,
    DatasetListResponse,
    DatasetStatisticsSummary,
    HistogramData,
)
from TKBEN.server.utils.logger import logger
from TKBEN.server.utils.constants import (
    API_ROUTE_DATASETS_ANALYZE,
    API_ROUTE_DATASETS_DOWNLOAD,
    API_ROUTE_DATASETS_LIST,
    API_ROUTE_DATASETS_UPLOAD,
    API_ROUTER_PREFIX_DATASETS,
)
from TKBEN.server.utils.services.datasets import DatasetService

router = APIRouter(prefix=API_ROUTER_PREFIX_DATASETS, tags=["datasets"])


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
    datasets = await asyncio.to_thread(service.get_available_datasets)
    return DatasetListResponse(datasets=datasets)


###############################################################################
@router.post(
    API_ROUTE_DATASETS_DOWNLOAD,
    response_model=DatasetDownloadResponse,
    status_code=status.HTTP_200_OK,
)
async def download_dataset(request: DatasetDownloadRequest) -> DatasetDownloadResponse:
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
        request.config,
    )

    service = DatasetService(hf_access_token=request.hf_access_token)

    try:
        result = await asyncio.to_thread(
            service.download_and_persist,
            corpus=request.corpus,
            config=request.config,
            remove_invalid=True,
        )
    except ValueError as exc:
        logger.warning("Dataset download validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to download dataset %s/%s", request.corpus, request.config)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download dataset from HuggingFace.",
        ) from exc

    logger.info(
        "Successfully downloaded and saved dataset: %s (%d documents)",
        result["dataset_name"],
        result["saved_count"],
    )

    histogram_data = result.get("histogram", {})
    histogram = HistogramData(
        bins=histogram_data.get("bins", []),
        counts=histogram_data.get("counts", []),
        bin_edges=histogram_data.get("bin_edges", []),
        min_length=histogram_data.get("min_length", 0),
        max_length=histogram_data.get("max_length", 0),
        mean_length=histogram_data.get("mean_length", 0.0),
        median_length=histogram_data.get("median_length", 0.0),
    )

    return DatasetDownloadResponse(
        status="success",
        dataset_name=result["dataset_name"],
        text_column=result["text_column"],
        document_count=result["document_count"],
        saved_count=result["saved_count"],
        histogram=histogram,
    )


###############################################################################
@router.post(
    API_ROUTE_DATASETS_UPLOAD,
    response_model=CustomDatasetUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_custom_dataset(
    file: UploadFile = File(..., description="CSV or Excel file to upload"),
) -> CustomDatasetUploadResponse:
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

    service = DatasetService()

    try:
        result = await asyncio.to_thread(
            service.upload_and_persist,
            file_content=file_content,
            filename=file.filename,
            remove_invalid=True,
        )
    except ValueError as exc:
        logger.warning("Custom dataset upload validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to process uploaded file %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process uploaded file.",
        ) from exc

    logger.info(
        "Successfully processed uploaded file: %s (%d documents)",
        result["dataset_name"],
        result["saved_count"],
    )

    histogram_data = result.get("histogram", {})
    histogram = HistogramData(
        bins=histogram_data.get("bins", []),
        counts=histogram_data.get("counts", []),
        bin_edges=histogram_data.get("bin_edges", []),
        min_length=histogram_data.get("min_length", 0),
        max_length=histogram_data.get("max_length", 0),
        mean_length=histogram_data.get("mean_length", 0.0),
        median_length=histogram_data.get("median_length", 0.0),
    )

    return CustomDatasetUploadResponse(
        status="success",
        dataset_name=result["dataset_name"],
        text_column=result["text_column"],
        document_count=result["document_count"],
        saved_count=result["saved_count"],
        histogram=histogram,
    )


###############################################################################
@router.post(
    API_ROUTE_DATASETS_ANALYZE,
    response_model=DatasetAnalysisResponse,
    status_code=status.HTTP_200_OK,
)
async def analyze_dataset(request: DatasetAnalysisRequest) -> DatasetAnalysisResponse:
    """
    Analyze a loaded dataset, computing per-document word-level statistics.

    This endpoint calculates word count, average word length, and word length
    standard deviation for each document. Results are persisted to the database
    and aggregate statistics are returned.

    Args:
        request: DatasetAnalysisRequest containing the dataset name.

    Returns:
        DatasetAnalysisResponse with analysis statistics.
    """
    logger.info("Dataset analysis requested: dataset=%s", request.dataset_name)

    service = DatasetService()

    # Check if dataset exists
    if not service.is_dataset_in_database(request.dataset_name):
        logger.warning("Dataset not found: %s", request.dataset_name)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{request.dataset_name}' not found. Please load it first.",
        )

    try:
        result = await asyncio.to_thread(
            service.analyze_dataset,
            dataset_name=request.dataset_name,
        )
    except Exception as exc:
        logger.exception("Failed to analyze dataset %s", request.dataset_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze dataset.",
        ) from exc

    logger.info(
        "Successfully analyzed dataset: %s (%d documents)",
        result["dataset_name"],
        result["analyzed_count"],
    )

    stats_data = result.get("statistics", {})
    statistics = DatasetStatisticsSummary(
        total_documents=stats_data.get("total_documents", 0),
        mean_words_count=stats_data.get("mean_words_count", 0.0),
        median_words_count=stats_data.get("median_words_count", 0.0),
        mean_avg_word_length=stats_data.get("mean_avg_word_length", 0.0),
        mean_std_word_length=stats_data.get("mean_std_word_length", 0.0),
    )

    return DatasetAnalysisResponse(
        status="success",
        dataset_name=result["dataset_name"],
        analyzed_count=result["analyzed_count"],
        statistics=statistics,
    )
