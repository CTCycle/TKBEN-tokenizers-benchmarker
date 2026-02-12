from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from TKBEN.server.routes.tokenizers import get_custom_tokenizers
from TKBEN.server.entities.benchmarks import (
    BenchmarkRunRequest,
)
from TKBEN.server.entities.jobs import JobStartResponse
from TKBEN.server.configurations import server_settings
from TKBEN.server.common.constants import (
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
) -> dict[str, Any]:
    global_metrics = []
    for metric in result.get("global_metrics", []):
        global_metrics.append(
            {
                "tokenizer": metric.get("tokenizer", ""),
                "dataset_name": metric.get("dataset_name", ""),
                "tokenization_speed_tps": metric.get("tokenization_speed_tps", 0.0),
                "throughput_chars_per_sec": metric.get("throughput_chars_per_sec", 0.0),
                "vocabulary_size": metric.get("vocabulary_size", 0),
                "avg_sequence_length": metric.get("avg_sequence_length", 0.0),
                "median_sequence_length": metric.get("median_sequence_length", 0.0),
                "subword_fertility": metric.get("subword_fertility", 0.0),
                "oov_rate": metric.get("oov_rate", 0.0),
                "word_recovery_rate": metric.get("word_recovery_rate", 0.0),
                "character_coverage": metric.get("character_coverage", 0.0),
                "determinism_rate": metric.get("determinism_rate", 0.0),
                "boundary_preservation_rate": metric.get(
                    "boundary_preservation_rate", 0.0
                ),
                "round_trip_fidelity_rate": metric.get("round_trip_fidelity_rate", 0.0),
            }
        )

    vocabulary_stats = []
    for vs in result.get("vocabulary_stats", []):
        vocabulary_stats.append(
            {
                "tokenizer": vs.get("tokenizer", ""),
                "vocabulary_size": vs.get("vocabulary_size", 0),
                "subwords_count": vs.get("subwords_count", 0),
                "true_words_count": vs.get("true_words_count", 0),
                "subwords_percentage": vs.get("subwords_percentage", 0.0),
            }
        )

    token_length_distributions = []
    for tld in result.get("token_length_distributions", []):
        bins = []
        for bin_item in tld.get("bins", []):
            bins.append(
                {
                    "bin_start": bin_item.get("bin_start", 0),
                    "bin_end": bin_item.get("bin_end", 0),
                    "count": bin_item.get("count", 0),
                }
            )
        token_length_distributions.append(
            {
                "tokenizer": tld.get("tokenizer", ""),
                "bins": bins,
                "mean": tld.get("mean", 0.0),
                "std": tld.get("std", 0.0),
            }
        )

    speed_metrics = []
    for sm in result.get("speed_metrics", []):
        speed_metrics.append(
            {
                "tokenizer": sm.get("tokenizer", ""),
                "tokens_per_second": sm.get("tokens_per_second", 0.0),
                "chars_per_second": sm.get("chars_per_second", 0.0),
                "processing_time_seconds": sm.get("processing_time_seconds", 0.0),
            }
        )

    return {
        "status": "success",
        "dataset_name": result.get("dataset_name", fallback_dataset_name),
        "documents_processed": result.get("documents_processed", 0),
        "tokenizers_processed": result.get("tokenizers_processed", []),
        "tokenizers_count": result.get("tokenizers_count", 0),
        "global_metrics": global_metrics,
        "chart_data": {
            "vocabulary_stats": vocabulary_stats,
            "token_length_distributions": token_length_distributions,
            "speed_metrics": speed_metrics,
        },
    }


def run_benchmark_job(
    request_payload: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    service = BenchmarkService(max_documents=request_payload.get("max_documents", 0))
    progress_callback = JobProgressReporter(job_manager, job_id)
    should_stop = JobStopChecker(job_manager, job_id)
    result = service.run_benchmarks(
        dataset_name=request_payload.get("dataset_name", ""),
        tokenizer_ids=request_payload.get("tokenizers", []),
        custom_tokenizers=request_payload.get("custom_tokenizers", {}),
        progress_callback=progress_callback,
        should_stop=should_stop,
    )
    if job_manager.should_stop(job_id):
        return {}
    return build_benchmark_payload(result, request_payload.get("dataset_name", ""))


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
        request.max_documents,
    )

    # Get custom tokenizer if specified
    custom_tokenizers = {}
    if request.custom_tokenizer_name:
        uploaded = get_custom_tokenizers()
        if request.custom_tokenizer_name in uploaded:
            custom_tokenizers[request.custom_tokenizer_name] = uploaded[request.custom_tokenizer_name]
            logger.info("Including custom tokenizer: %s", request.custom_tokenizer_name)

    if job_manager.is_job_running("benchmark_run"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Benchmark run is already in progress.",
        )

    service = BenchmarkService(max_documents=request.max_documents)
    doc_count = service.get_dataset_document_count(request.dataset_name)
    if doc_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset '{request.dataset_name}' not found or empty",
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
        poll_interval=server_settings.jobs.polling_interval,
    )

