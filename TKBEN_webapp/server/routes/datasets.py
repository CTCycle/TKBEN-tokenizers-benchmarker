from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from TKBEN_webapp.server.schemas.dataset import (
    CustomDatasetUploadResponse,
    DatasetDownloadRequest,
    DatasetDownloadResponse,
    HistogramData,
)
from TKBEN_webapp.server.utils.logger import logger
from TKBEN_webapp.server.utils.services.datasets import DatasetService

router = APIRouter(prefix="/datasets", tags=["datasets"])


###############################################################################
@router.post(
    "/download",
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
    "/upload",
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
