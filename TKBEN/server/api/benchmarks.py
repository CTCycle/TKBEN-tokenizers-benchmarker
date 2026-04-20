from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status

from TKBEN.server.domain.benchmarks import (
    BenchmarkMetricCatalogCategory,
    BenchmarkMetricCatalogResponse,
    BenchmarkReportListResponse,
    BenchmarkReportSummary,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
)
from TKBEN.server.domain.jobs import JobStartResponse
from TKBEN.server.configurations import get_server_settings
from TKBEN.server.common.constants import (
    API_ROUTE_BENCHMARKS_METRICS_CATALOG,
    API_ROUTE_BENCHMARKS_REPORT_BY_ID,
    API_ROUTE_BENCHMARKS_REPORTS,
    API_ROUTE_BENCHMARKS_RUN,
    API_ROUTER_PREFIX_BENCHMARKS,
)
from TKBEN.server.common.utils.logger import logger
from TKBEN.server.services.benchmark_jobs import BenchmarkJobService
from TKBEN.server.services.benchmarks import BenchmarkService


router = APIRouter(prefix=API_ROUTER_PREFIX_BENCHMARKS, tags=["benchmarks"])
benchmark_job_service = BenchmarkJobService()


###############################################################################
@router.post(
    API_ROUTE_BENCHMARKS_RUN,
    response_model=JobStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_benchmarks(
    request: Request,
    payload: BenchmarkRunRequest,
) -> JobStartResponse:
    if not payload.tokenizers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one tokenizer must be specified.",
        )

    if not payload.dataset_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset name must be specified.",
        )

    logger.info(
        "Benchmark run requested: dataset=%s, tokenizers=%s, max_docs=%s",
        payload.dataset_name,
        payload.tokenizers,
        payload.config.max_documents,
    )

    service = BenchmarkService(max_documents=payload.config.max_documents)
    custom_tokenizers = service.resolve_custom_tokenizer_selection(
        payload.custom_tokenizer_name
    )

    job_manager = request.app.state.job_manager
    if job_manager.is_job_running("benchmark_run"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Benchmark run is already in progress.",
        )

    doc_count = service.get_dataset_document_count(payload.dataset_name)
    if doc_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset '{payload.dataset_name}' not found or empty",
        )

    try:
        missing_tokenizers = service.get_missing_persisted_tokenizers(payload.tokenizers)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if missing_tokenizers:
        missing_display = ", ".join(missing_tokenizers[:5])
        if len(missing_tokenizers) > 5:
            missing_display = f"{missing_display}, ..."
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Tokenizers must be downloaded before benchmarking. "
                f"Missing: {missing_display}"
            ),
        )

    request_payload = payload.model_dump()
    request_payload["custom_tokenizers"] = custom_tokenizers

    job_id = job_manager.start_job(
        job_type="benchmark_run",
        runner=benchmark_job_service.run_benchmark_job,
        kwargs={
            "request_payload": request_payload,
            "job_manager": job_manager,
        },
    )

    job_status = job_manager.get_job_status(job_id)
    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize benchmark job.",
        )

    return JobStartResponse(
        job_id=job_id,
        job_type=job_status["job_type"],
        status=job_status["status"],
        message="Benchmark job started.",
        poll_interval=get_server_settings().jobs.polling_interval,
    )


###############################################################################
@router.get(
    API_ROUTE_BENCHMARKS_REPORTS,
    response_model=BenchmarkReportListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_benchmark_reports(
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> BenchmarkReportListResponse:
    service = BenchmarkService()
    reports = await asyncio.to_thread(service.list_benchmark_reports, limit)
    report_summaries = [
        BenchmarkReportSummary.model_validate(report) for report in reports
    ]
    return BenchmarkReportListResponse(reports=report_summaries)


###############################################################################
@router.get(
    API_ROUTE_BENCHMARKS_REPORT_BY_ID,
    response_model=BenchmarkRunResponse,
    status_code=status.HTTP_200_OK,
)
async def get_benchmark_report_by_id(report_id: int) -> BenchmarkRunResponse:
    service = BenchmarkService()
    report = await asyncio.to_thread(service.load_benchmark_report_by_id, report_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark report '{report_id}' not found.",
        )
    return BenchmarkRunResponse(**report)


###############################################################################
@router.get(
    API_ROUTE_BENCHMARKS_METRICS_CATALOG,
    response_model=BenchmarkMetricCatalogResponse,
    status_code=status.HTTP_200_OK,
)
async def get_benchmark_metrics_catalog() -> BenchmarkMetricCatalogResponse:
    service = BenchmarkService()
    categories = await asyncio.to_thread(service.get_metric_catalog)
    metric_categories = [
        BenchmarkMetricCatalogCategory.model_validate(item) for item in categories
    ]
    return BenchmarkMetricCatalogResponse(categories=metric_categories)
