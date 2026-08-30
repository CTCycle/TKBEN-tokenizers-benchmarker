from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import ValidationError

from server.contracts.benchmarks import (
    BenchmarkMetricCatalogCategory,
    BenchmarkMetricCatalogResponse,
    BenchmarkReportQuery,
    BenchmarkReportListResponse,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    BenchmarkReportSort,
)
from server.contracts.jobs import JobStartResponse
from server.common.constants import (
    API_ROUTE_BENCHMARKS_METRICS_CATALOG,
    API_ROUTE_BENCHMARKS_REPORT_BY_ID,
    API_ROUTE_BENCHMARKS_REPORTS,
    API_ROUTE_BENCHMARKS_RUN,
    API_ROUTER_PREFIX_BENCHMARKS,
)
from server.common.utils.logger import logger
from server.api.helpers import ManagedJobHttpAdapter
from server.services.benchmark_jobs import BenchmarkJobService
from server.services.benchmark_reports import BenchmarkReportService
from server.services.benchmarks import BenchmarkService
from server.services.managed_jobs import ManagedJobSpec


router = APIRouter(prefix=API_ROUTER_PREFIX_BENCHMARKS, tags=["benchmarks"])


###############################################################################
def _build_benchmark_report_query(
    search: Annotated[str | None, Query(max_length=160)] = None,
    sort: Annotated[BenchmarkReportSort, Query()] = BenchmarkReportSort.NEWEST,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> BenchmarkReportQuery:
    try:
        return BenchmarkReportQuery(
            search=search,
            sort=sort,
            offset=offset,
            limit=limit,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


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
    logger.info(
        "Benchmark run requested: dataset=%s, tokenizers=%s, max_docs=%s",
        payload.dataset_name,
        payload.tokenizers,
        payload.config.max_documents,
    )

    service = BenchmarkService(max_documents=payload.config.max_documents)
    try:
        request_payload = await asyncio.to_thread(service.prepare_run, payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return ManagedJobHttpAdapter.start(
        request,
        ManagedJobSpec(
            job_type="benchmark_run",
            runner=BenchmarkJobService().run_benchmark_job,
            kwargs={"request_payload": request_payload},
            conflict_detail="Benchmark run is already in progress.",
            initialization_detail="Failed to initialize benchmark job.",
            message="Benchmark job started.",
        ),
    )


###############################################################################
@router.get(
    API_ROUTE_BENCHMARKS_REPORTS,
    response_model=BenchmarkReportListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_benchmark_reports(
    query: Annotated[BenchmarkReportQuery, Depends(_build_benchmark_report_query)],
) -> BenchmarkReportListResponse:
    service = BenchmarkReportService()
    return await asyncio.to_thread(service.list_benchmark_reports, query)


###############################################################################
@router.delete(
    API_ROUTE_BENCHMARKS_REPORT_BY_ID,
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_benchmark_report(report_id: int) -> None:
    service = BenchmarkReportService()
    deleted = await asyncio.to_thread(service.delete_benchmark_report, report_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark report '{report_id}' not found.",
        )


###############################################################################
@router.get(
    API_ROUTE_BENCHMARKS_REPORT_BY_ID,
    response_model=BenchmarkRunResponse,
    status_code=status.HTTP_200_OK,
)
async def get_benchmark_report_by_id(report_id: int) -> BenchmarkRunResponse:
    service = BenchmarkReportService()
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
