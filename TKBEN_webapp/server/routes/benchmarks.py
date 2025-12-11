from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, status

from TKBEN_webapp.server.routes.tokenizers import get_custom_tokenizers
from TKBEN_webapp.server.schemas.benchmarks import (
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    ChartData,
    GlobalMetrics,
    SpeedMetric,
    TokenLengthBin,
    TokenLengthDistribution,
    VocabularyStats,
)
from TKBEN_webapp.server.utils.logger import logger
from TKBEN_webapp.server.utils.services.benchmarks import BenchmarkService


router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


###############################################################################
@router.post(
    "/run",
    response_model=BenchmarkRunResponse,
    status_code=status.HTTP_200_OK,
)
async def run_benchmarks(request: BenchmarkRunRequest) -> BenchmarkRunResponse:
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

    service = BenchmarkService(
        max_documents=request.max_documents,
    )

    try:
        result = await asyncio.to_thread(
            service.run_benchmarks,
            dataset_name=request.dataset_name,
            tokenizer_ids=request.tokenizers,
            custom_tokenizers=custom_tokenizers,
        )
    except ValueError as exc:
        logger.warning("Benchmark run validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to run benchmarks")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run tokenizer benchmarks.",
        ) from exc

    logger.info(
        "Benchmark run completed: %d tokenizers, %d documents",
        result.get("tokenizers_count", 0),
        result.get("documents_processed", 0),
    )

    # Convert raw metrics to schema objects
    global_metrics = []
    for m in result.get("global_metrics", []):
        global_metrics.append(
            GlobalMetrics(
                tokenizer=m.get("tokenizer", ""),
                dataset_name=m.get("dataset_name", ""),
                tokenization_speed_tps=m.get("tokenization_speed_tps", 0.0),
                throughput_chars_per_sec=m.get("throughput_chars_per_sec", 0.0),
                vocabulary_size=m.get("vocabulary_size", 0),
                avg_sequence_length=m.get("avg_sequence_length", 0.0),
                median_sequence_length=m.get("median_sequence_length", 0.0),
                subword_fertility=m.get("subword_fertility", 0.0),
                oov_rate=m.get("oov_rate", 0.0),
                word_recovery_rate=m.get("word_recovery_rate", 0.0),
                character_coverage=m.get("character_coverage", 0.0),
                determinism_rate=m.get("determinism_rate", 0.0),
                boundary_preservation_rate=m.get("boundary_preservation_rate", 0.0),
                round_trip_fidelity_rate=m.get("round_trip_fidelity_rate", 0.0),
            )
        )

    # Build chart data from result
    vocabulary_stats = []
    for vs in result.get("vocabulary_stats", []):
        vocabulary_stats.append(
            VocabularyStats(
                tokenizer=vs.get("tokenizer", ""),
                vocabulary_size=vs.get("vocabulary_size", 0),
                subwords_count=vs.get("subwords_count", 0),
                true_words_count=vs.get("true_words_count", 0),
                subwords_percentage=vs.get("subwords_percentage", 0.0),
            )
        )

    token_length_distributions = []
    for tld in result.get("token_length_distributions", []):
        bins = [
            TokenLengthBin(bin_start=b["bin_start"], bin_end=b["bin_end"], count=b["count"])
            for b in tld.get("bins", [])
        ]
        token_length_distributions.append(
            TokenLengthDistribution(
                tokenizer=tld.get("tokenizer", ""),
                bins=bins,
                mean=tld.get("mean", 0.0),
                std=tld.get("std", 0.0),
            )
        )

    speed_metrics = []
    for sm in result.get("speed_metrics", []):
        speed_metrics.append(
            SpeedMetric(
                tokenizer=sm.get("tokenizer", ""),
                tokens_per_second=sm.get("tokens_per_second", 0.0),
                chars_per_second=sm.get("chars_per_second", 0.0),
                processing_time_seconds=sm.get("processing_time_seconds", 0.0),
            )
        )

    chart_data = ChartData(
        vocabulary_stats=vocabulary_stats,
        token_length_distributions=token_length_distributions,
        speed_metrics=speed_metrics,
    )

    return BenchmarkRunResponse(
        status="success",
        dataset_name=result.get("dataset_name", request.dataset_name),
        documents_processed=result.get("documents_processed", 0),
        tokenizers_processed=result.get("tokenizers_processed", []),
        tokenizers_count=result.get("tokenizers_count", 0),
        global_metrics=global_metrics,
        chart_data=chart_data,
    )

