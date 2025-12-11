from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, status

from TKBEN_webapp.server.schemas.benchmarks import (
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    GlobalMetrics,
    PlotData,
)
from TKBEN_webapp.server.utils.logger import logger
from TKBEN_webapp.server.utils.services.benchmark_service import BenchmarkService


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
    computing vocabulary statistics, tokenization metrics, and generating
    visualization plots. Results are persisted to the database.

    Args:
        request: BenchmarkRunRequest containing tokenizers, dataset_name, and options.

    Returns:
        BenchmarkRunResponse with benchmark results, metrics, and plots as base64.
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

    service = BenchmarkService(
        max_documents=request.max_documents,
        include_custom_tokenizer=request.include_custom_tokenizer,
        include_nsl=request.include_nsl,
    )

    try:
        result = await asyncio.to_thread(
            service.run_benchmarks,
            dataset_name=request.dataset_name,
            tokenizer_ids=request.tokenizers,
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
    plots = [
        PlotData(name=p["name"], data=p["data"])
        for p in result.get("plots_generated", [])
    ]

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

    return BenchmarkRunResponse(
        status="success",
        dataset_name=result.get("dataset_name", request.dataset_name),
        documents_processed=result.get("documents_processed", 0),
        tokenizers_processed=result.get("tokenizers_processed", []),
        tokenizers_count=result.get("tokenizers_count", 0),
        plots=plots,
        global_metrics=global_metrics,
    )
