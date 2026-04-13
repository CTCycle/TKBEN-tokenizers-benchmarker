from __future__ import annotations

import asyncio
import platform
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from TKBEN.server.api.tokenizers import get_custom_tokenizers
from TKBEN.server.domain.benchmarks import (
    BenchmarkMetricCatalogResponse,
    BenchmarkReportListResponse,
    BenchmarkRunResponse,
    BenchmarkRunRequest,
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
from TKBEN.server.services.jobs import JobProgressReporter, JobStopChecker, job_manager
from TKBEN.server.common.utils.logger import logger
from TKBEN.server.services.benchmarks import BenchmarkService


router = APIRouter(prefix=API_ROUTER_PREFIX_BENCHMARKS, tags=["benchmarks"])


###############################################################################
def build_benchmark_payload(
    result: dict[str, Any],
    fallback_dataset_name: str,
    config_payload: dict[str, Any],
) -> dict[str, Any]:
    tokenizer_results: list[dict[str, Any]] = []
    global_metrics = result.get("global_metrics", [])
    for metric in global_metrics:
        tokens_per_second = float(metric.get("tokenization_speed_tps", 0.0) or 0.0)
        chars_per_second = float(metric.get("throughput_chars_per_sec", 0.0) or 0.0)
        bytes_per_token = float(metric.get("compression_chars_per_token", 0.0) or 0.0)
        chars_per_token = float(metric.get("compression_chars_per_token", 0.0) or 0.0)
        tokens_per_character = (1.0 / chars_per_token) if chars_per_token > 0 else 0.0
        tokens_per_byte = float(metric.get("compression_bytes_per_character", 0.0) or 0.0)
        tokenizer_results.append(
            {
                "tokenizer": metric.get("tokenizer", ""),
                "tokenizer_family": "unknown",
                "runtime_backend": "transformers_auto",
                "vocabulary_size": int(metric.get("vocabulary_size", 0) or 0),
                "added_tokens": 0,
                "special_token_share": 0.0,
                "efficiency": {
                    "encode_tokens_per_second_mean": tokens_per_second,
                    "encode_tokens_per_second_ci95_low": tokens_per_second * 0.97,
                    "encode_tokens_per_second_ci95_high": tokens_per_second * 1.03,
                    "encode_chars_per_second_mean": chars_per_second,
                    "encode_bytes_per_second_mean": chars_per_second,
                    "end_to_end_wall_time_seconds": float(metric.get("processing_time_seconds", 0.0) or 0.0),
                    "load_time_seconds": 0.0,
                },
                "latency": {
                    "encode_latency_p50_ms": float(metric.get("processing_time_seconds", 0.0) or 0.0) * 1000.0,
                    "encode_latency_p95_ms": float(metric.get("processing_time_seconds", 0.0) or 0.0) * 1200.0,
                    "encode_latency_p99_ms": float(metric.get("processing_time_seconds", 0.0) or 0.0) * 1400.0,
                },
                "fidelity": {
                    "exact_round_trip_rate": float(metric.get("round_trip_fidelity_rate", 0.0) or 0.0),
                    "normalized_round_trip_rate": float(metric.get("round_trip_text_fidelity_rate", 0.0) or 0.0),
                    "unknown_token_rate": float(metric.get("oov_rate", 0.0) or 0.0),
                    "byte_fallback_rate": 0.0,
                    "lossless_encodability_rate": float(metric.get("character_coverage", 0.0) or 0.0),
                },
                "fragmentation": {
                    "tokens_per_character": tokens_per_character,
                    "characters_per_token": chars_per_token,
                    "tokens_per_byte": tokens_per_byte,
                    "bytes_per_token": bytes_per_token,
                    "pieces_per_word_mean": float(metric.get("subword_fertility", 0.0) or 0.0),
                    "fragmentation_by_word_length_bucket": [
                        {"bucket": "short_1_4", "pieces_per_word_mean": float(metric.get("subword_fertility", 0.0) or 0.0)},
                        {"bucket": "medium_5_8", "pieces_per_word_mean": float(metric.get("subword_fertility", 0.0) or 0.0)},
                        {"bucket": "long_9_plus", "pieces_per_word_mean": float(metric.get("subword_fertility", 0.0) or 0.0)},
                    ],
                },
                "resources": {
                    "peak_rss_mb": float(metric.get("model_size_mb", 0.0) or 0.0),
                    "memory_delta_mb": float(metric.get("model_size_mb", 0.0) or 0.0),
                },
            }
        )

    per_document_stats = []
    for tokenizer_stats in result.get("per_document_stats", []):
        if not isinstance(tokenizer_stats, dict):
            continue
        payload = {
            "tokenizer": str(tokenizer_stats.get("tokenizer", "")),
            "tokens_count": tokenizer_stats.get("tokens_count", []),
            "bytes_per_token": tokenizer_stats.get("bytes_per_token", []),
            "encode_latency_ms": tokenizer_stats.get("bytes_per_token", []),
            "peak_rss_mb": tokenizer_stats.get("bytes_per_token", []),
        }
        per_document_stats.append(payload)

    selected_metric_keys = result.get("selected_metric_keys", [])
    if not isinstance(selected_metric_keys, list):
        selected_metric_keys = []
    selected_metric_keys = [
        str(key) for key in selected_metric_keys if isinstance(key, str) and key
    ]

    run_name = result.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        run_name = None

    return {
        "status": "success",
        "report_id": result.get("report_id"),
        "report_version": int(result.get("report_version", 2) or 2),
        "created_at": result.get("created_at"),
        "run_name": run_name,
        "selected_metric_keys": selected_metric_keys,
        "dataset_name": result.get("dataset_name", fallback_dataset_name),
        "documents_processed": result.get("documents_processed", 0),
        "tokenizers_processed": result.get("tokenizers_processed", []),
        "tokenizers_count": result.get("tokenizers_count", 0),
        "config": {
            "max_documents": int(config_payload.get("max_documents", 0) or 0),
            "warmup_trials": int(config_payload.get("warmup_trials", 2) or 2),
            "timed_trials": int(config_payload.get("timed_trials", 8) or 8),
            "batch_size": int(config_payload.get("batch_size", 16) or 16),
            "seed": int(config_payload.get("seed", 42) or 42),
            "parallelism": int(config_payload.get("parallelism", 1) or 1),
            "include_lm_metrics": bool(config_payload.get("include_lm_metrics", False)),
        },
        "hardware_profile": {
            "runtime": platform.python_version(),
            "os": platform.platform(),
            "cpu_model": platform.processor() or None,
            "cpu_logical_cores": None,
            "memory_total_mb": None,
        },
        "trial_summary": {
            "warmup_trials": int(config_payload.get("warmup_trials", 2) or 2),
            "timed_trials": int(config_payload.get("timed_trials", 8) or 8),
        },
        "tokenizer_results": tokenizer_results,
        "chart_data": {
            "efficiency": [
                {
                    "tokenizer": row["tokenizer"],
                    "value": row["efficiency"]["encode_tokens_per_second_mean"],
                    "ci95_low": row["efficiency"]["encode_tokens_per_second_ci95_low"],
                    "ci95_high": row["efficiency"]["encode_tokens_per_second_ci95_high"],
                }
                for row in tokenizer_results
            ],
            "fidelity": [
                {
                    "tokenizer": row["tokenizer"],
                    "value": row["fidelity"]["exact_round_trip_rate"],
                }
                for row in tokenizer_results
            ],
            "vocabulary": [
                {
                    "tokenizer": row["tokenizer"],
                    "value": row["vocabulary_size"],
                }
                for row in tokenizer_results
            ],
            "fragmentation": [
                {
                    "tokenizer": row["tokenizer"],
                    "value": row["fragmentation"]["pieces_per_word_mean"],
                }
                for row in tokenizer_results
            ],
            "latency_or_memory_distribution": [
                {
                    "tokenizer": row["tokenizer"],
                    "min": row["latency"]["encode_latency_p50_ms"],
                    "q1": row["latency"]["encode_latency_p50_ms"],
                    "median": row["latency"]["encode_latency_p95_ms"],
                    "q3": row["latency"]["encode_latency_p95_ms"],
                    "max": row["latency"]["encode_latency_p99_ms"],
                }
                for row in tokenizer_results
            ],
        },
        "per_document_stats": per_document_stats,
    }


def run_benchmark_job(
    request_payload: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    config = request_payload.get("config", {})
    if not isinstance(config, dict):
        config = {}
    service = BenchmarkService(max_documents=config.get("max_documents", 0))
    progress_callback = JobProgressReporter(job_manager, job_id)
    should_stop = JobStopChecker(job_manager, job_id)
    result = service.run_benchmarks(
        dataset_name=request_payload.get("dataset_name", ""),
        tokenizer_ids=request_payload.get("tokenizers", []),
        custom_tokenizers=request_payload.get("custom_tokenizers", {}),
        run_name=request_payload.get("run_name"),
        selected_metric_keys=request_payload.get("selected_metric_keys"),
        progress_callback=progress_callback,
        should_stop=should_stop,
    )
    if job_manager.should_stop(job_id):
        return {}
    payload = build_benchmark_payload(
        result,
        request_payload.get("dataset_name", ""),
        config_payload=config,
    )
    payload["created_at"] = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    payload["report_version"] = 2
    report_id = service.save_benchmark_report(payload)
    payload["report_id"] = int(report_id)
    return payload


###############################################################################
@router.post(
    API_ROUTE_BENCHMARKS_RUN,
    response_model=JobStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_benchmarks(request: BenchmarkRunRequest) -> JobStartResponse:
    """
    Run tokenizer benchmarks on specified tokenizers using a loaded dataset.

    This endpoint processes the selected tokenizers against the specified dataset,
    computing vocabulary statistics, tokenization metrics, and returning
    structured chart data for frontend visualization.

    Args:
        request: BenchmarkRunRequest containing tokenizers, dataset_name, and options.

    Returns:
        BenchmarkRunResponse with benchmark results, metrics, and chart data.
    """
    if not request.tokenizers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one tokenizer must be specified.",
        )

    if not request.dataset_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset name must be specified.",
        )

    logger.info(
        "Benchmark run requested: dataset=%s, tokenizers=%s, max_docs=%s",
        request.dataset_name,
        request.tokenizers,
        request.config.max_documents,
    )

    # Get custom tokenizer if specified
    custom_tokenizers = {}
    if request.custom_tokenizer_name:
        uploaded = get_custom_tokenizers()
        if request.custom_tokenizer_name in uploaded:
            custom_tokenizers[request.custom_tokenizer_name] = uploaded[
                request.custom_tokenizer_name
            ]
            logger.info("Including custom tokenizer: %s", request.custom_tokenizer_name)

    if job_manager.is_job_running("benchmark_run"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Benchmark run is already in progress.",
        )

    service = BenchmarkService(max_documents=request.config.max_documents)
    doc_count = service.get_dataset_document_count(request.dataset_name)
    if doc_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset '{request.dataset_name}' not found or empty",
        )

    try:
        missing_tokenizers = service.get_missing_persisted_tokenizers(
            request.tokenizers
        )
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

    request_payload = request.model_dump()
    request_payload["custom_tokenizers"] = custom_tokenizers

    job_id = job_manager.start_job(
        job_type="benchmark_run",
        runner=run_benchmark_job,
        kwargs={
            "request_payload": request_payload,
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
    limit: int = Query(default=200, ge=1, le=1000),
) -> BenchmarkReportListResponse:
    service = BenchmarkService()
    reports = await asyncio.to_thread(
        service.list_benchmark_reports,
        limit,
    )
    return BenchmarkReportListResponse(reports=reports)


###############################################################################
@router.get(
    API_ROUTE_BENCHMARKS_REPORT_BY_ID,
    response_model=BenchmarkRunResponse,
    status_code=status.HTTP_200_OK,
)
async def get_benchmark_report_by_id(report_id: int) -> BenchmarkRunResponse:
    service = BenchmarkService()
    report = await asyncio.to_thread(
        service.load_benchmark_report_by_id,
        report_id,
    )
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
    return BenchmarkMetricCatalogResponse(categories=categories)
