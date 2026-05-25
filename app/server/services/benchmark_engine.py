from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Callable

import numpy as np
import psutil

from server.domain.benchmark_observations import BatchObservation, TokenizerRunConfig
from server.services.tokenizer_adapters import TokenizerAdapter


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile for empty values")
    return float(np.percentile(np.asarray(values, dtype=float), q * 100.0))


def ci95_bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(statistics.mean(values))
    if len(values) < 2:
        return mean, mean
    half_width = 1.96 * (statistics.stdev(values) / (len(values) ** 0.5))
    return max(0.0, mean - half_width), mean + half_width


def run_tokenizer_trials(
    *,
    tokenizer: TokenizerAdapter,
    text_batches_factory: Callable[[], list[list[str]]],
    config: TokenizerRunConfig,
    warmup_trials: int,
    timed_trials: int,
) -> list[BatchObservation]:
    if warmup_trials < 0:
        raise ValueError("warmup_trials must be non-negative")
    if timed_trials <= 0:
        raise ValueError("timed_trials must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    for _ in range(warmup_trials):
        for texts in text_batches_factory():
            tokenizer.encode_batch(
                texts,
                add_special_tokens=config.add_special_tokens,
                padding=config.padding,
                truncation=config.truncation,
                max_length=config.max_length,
            )

    observations: list[BatchObservation] = []
    for trial_index in range(timed_trials):
        gc.collect()
        for batch_index, texts in enumerate(text_batches_factory()):
            input_bytes = sum(len(text.encode("utf-8")) for text in texts)
            process = psutil.Process()
            rss_before = process.memory_info().rss
            start_ns = time.perf_counter_ns()
            encoded = tokenizer.encode_batch(
                texts,
                add_special_tokens=config.add_special_tokens,
                padding=config.padding,
                truncation=config.truncation,
                max_length=config.max_length,
            )
            elapsed_ns = time.perf_counter_ns() - start_ns
            rss_after = process.memory_info().rss
            peak_rss_mb = float(max(rss_before, rss_after) / (1024 * 1024))
            observations.append(
                BatchObservation(
                    tokenizer_id=tokenizer.tokenizer_id,
                    trial_index=trial_index,
                    batch_index=batch_index,
                    documents=len(texts),
                    input_utf8_bytes=input_bytes,
                    token_count=sum(encoded.token_counts),
                    unknown_token_count=(
                        None
                        if any(v is None for v in encoded.unknown_counts)
                        else sum(int(v or 0) for v in encoded.unknown_counts)
                    ),
                    elapsed_ns=int(elapsed_ns),
                    peak_rss_mb=peak_rss_mb,
                )
            )
    return observations


def summarize_observations(observations: list[BatchObservation]) -> dict[str, float | int | None]:
    if not observations:
        raise ValueError("No benchmark observations collected")

    elapsed_seconds = [obs.elapsed_ns / 1_000_000_000 for obs in observations]
    total_elapsed_seconds = float(sum(elapsed_seconds))
    total_docs = int(sum(obs.documents for obs in observations))
    total_tokens = int(sum(obs.token_count for obs in observations))
    total_bytes = int(sum(obs.input_utf8_bytes for obs in observations))
    per_batch_latency_ms = [seconds * 1000 for seconds in elapsed_seconds]
    per_trial_tps: list[float] = []
    for trial_index in sorted({obs.trial_index for obs in observations}):
        trial_rows = [obs for obs in observations if obs.trial_index == trial_index]
        trial_elapsed = sum(obs.elapsed_ns for obs in trial_rows) / 1_000_000_000
        trial_tokens = sum(obs.token_count for obs in trial_rows)
        per_trial_tps.append(float(trial_tokens / trial_elapsed) if trial_elapsed > 0 else 0.0)
    ci_low, ci_high = ci95_bounds(per_trial_tps)
    return {
        "documents_processed": total_docs,
        "total_tokens": total_tokens,
        "input_utf8_bytes": total_bytes,
        "total_time_seconds": total_elapsed_seconds,
        "documents_per_second": total_docs / total_elapsed_seconds if total_elapsed_seconds else 0.0,
        "tokens_per_second": total_tokens / total_elapsed_seconds if total_elapsed_seconds else 0.0,
        "tokens_per_second_ci95_low": ci_low,
        "tokens_per_second_ci95_high": ci_high,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else None,
        "tokens_per_utf8_byte": total_tokens / total_bytes if total_bytes else None,
        "latency_batch_p50_ms": percentile(per_batch_latency_ms, 0.50),
        "latency_batch_p95_ms": percentile(per_batch_latency_ms, 0.95),
        "latency_batch_p99_ms": percentile(per_batch_latency_ms, 0.99),
        "trial_count": len({obs.trial_index for obs in observations}),
        "batch_observation_count": len(observations),
    }
