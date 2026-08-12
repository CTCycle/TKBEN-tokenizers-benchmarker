from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard, cast

import numpy as np
import pandas as pd

from server.domain.benchmarks import (
    BenchmarkDashboardBucketPoint,
    BenchmarkDashboardData,
    BenchmarkDashboardDistribution,
    BenchmarkDashboardHistogramBin,
    BenchmarkDashboardPoint,
    BenchmarkDashboardWidgetData,
    BenchmarkEfficiencyMetrics,
    BenchmarkFidelityMetrics,
    BenchmarkFragmentationBucket,
    BenchmarkFragmentationMetrics,
    BenchmarkLatencyMetrics,
    BenchmarkPerDocumentTokenizerStats,
    BenchmarkResourceMetrics,
    BenchmarkTokenizerResult,
)
from server.services.metrics.benchmark_definitions import BENCHMARK_METRIC_DEFINITIONS

###############################################################################
class BenchmarkResultBuilder:

    # -------------------------------------------------------------------------
    def __init__(self, tools: Any) -> None:
        self.tools = tools

    # -------------------------------------------------------------------------
    def _percentile(self, values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=float), percentile))

    # -------------------------------------------------------------------------
    def _ci95_half_width(self, values: Sequence[float]) -> float:
        if len(values) < 2:
            return 0.0
        sample_std = statistics.stdev(values)
        return float(1.96 * (sample_std / (len(values) ** 0.5)))

    # -------------------------------------------------------------------------
    def _fragmentation_bucket_label(self, word_length: int) -> str:
        if word_length <= 4:
            return "short_1_4"
        if word_length <= 8:
            return "medium_5_8"
        return "long_9_plus"

    # -------------------------------------------------------------------------
    def _extract_vocab_result(
        self,
        tokenizer: Any,
    ) -> Mapping[Any, Any] | Sequence[Any] | None:
        vocab_method = getattr(tokenizer, "get_vocab", None)
        if not callable(vocab_method):
            return None
        try:
            candidate = vocab_method()
        except Exception:
            return None
        if isinstance(candidate, Mapping):
            return candidate
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            return candidate
        return None

    # -------------------------------------------------------------------------
    def _build_tokenizer_result(
        self,
        *,
        tokenizer_name: str,
        status: str = "success",
        error_type: str | None = None,
        error_message: str | None = None,
        trial_tokenization_speeds_tps: list[float],
        throughput_chars_per_sec: float,
        encode_only_wall_time_seconds: float,
        dataset_stream_wall_time_seconds: float,
        postprocess_wall_time_seconds: float,
        total_processing_time_seconds: float,
        observed_latency_ms: list[float],
        latency_sample_count: int,
        vocabulary_size: int,
        oov_rate: float | None,
        character_coverage: float | None,
        round_trip_fidelity_rate: float | None,
        round_trip_text_fidelity_rate: float | None,
        subword_fertility: float | None,
        compression_chars_per_token: float | None,
        compression_bytes_per_character: float | None,
        fragmentation_buckets: list[BenchmarkFragmentationBucket],
        peak_rss_mb: float | None = None,
        memory_delta_mb: float | None = None,
    ) -> BenchmarkTokenizerResult:
        chars_per_token = float(compression_chars_per_token) if compression_chars_per_token is not None else None
        tokens_per_character = (1.0 / chars_per_token) if chars_per_token is not None and chars_per_token > 0 else None
        tokens_per_byte = float(compression_bytes_per_character) if compression_bytes_per_character is not None else None
        bytes_per_token = (1.0 / tokens_per_byte) if tokens_per_byte is not None and tokens_per_byte > 0 else None
        tokenization_speed_tps = (
            float(np.mean(trial_tokenization_speeds_tps))
            if trial_tokenization_speeds_tps
            else 0.0
        )
        ci95_half_width = self._ci95_half_width(trial_tokenization_speeds_tps)
        latency_p50 = self._percentile(observed_latency_ms, 50.0) if observed_latency_ms else None
        latency_p95 = self._percentile(observed_latency_ms, 95.0) if observed_latency_ms else None
        latency_p99 = self._percentile(observed_latency_ms, 99.0) if observed_latency_ms else None

        return BenchmarkTokenizerResult(
            tokenizer=tokenizer_name,
            status=status,
            error_type=error_type,
            error_message=error_message,
            tokenizer_family="unknown",
            runtime_backend="transformers_auto",
            vocabulary_size=int(vocabulary_size),
            added_tokens=0,
            special_token_share=0.0,
            efficiency=BenchmarkEfficiencyMetrics(
                encode_tokens_per_second_mean=float(tokenization_speed_tps),
                encode_tokens_per_second_ci95_low=float(
                    max(0.0, tokenization_speed_tps - ci95_half_width)
                ),
                encode_tokens_per_second_ci95_high=float(
                    tokenization_speed_tps + ci95_half_width
                ),
                encode_chars_per_second_mean=float(throughput_chars_per_sec),
                encode_bytes_per_second_mean=float(throughput_chars_per_sec),
                encode_only_wall_time_seconds=float(encode_only_wall_time_seconds),
                dataset_stream_wall_time_seconds=float(
                    dataset_stream_wall_time_seconds
                ),
                postprocess_wall_time_seconds=float(postprocess_wall_time_seconds),
                end_to_end_wall_time_seconds=float(total_processing_time_seconds),
                load_time_seconds=0.0,
            ),
            latency=BenchmarkLatencyMetrics(
                encode_latency_p50_ms=latency_p50,
                encode_latency_p95_ms=latency_p95,
                encode_latency_p99_ms=latency_p99,
                sample_count=int(latency_sample_count),
            ),
            fidelity=BenchmarkFidelityMetrics(
                exact_round_trip_rate=round_trip_fidelity_rate,
                normalized_round_trip_rate=round_trip_text_fidelity_rate,
                unknown_token_rate=(
                    float(oov_rate) if isinstance(oov_rate, int | float) else None
                ),
                byte_fallback_rate=None,
                lossless_encodability_rate=(
                    float(character_coverage)
                    if isinstance(character_coverage, int | float)
                    else None
                ),
            ),
            fragmentation=BenchmarkFragmentationMetrics(
                tokens_per_character=tokens_per_character,
                characters_per_token=chars_per_token,
                tokens_per_byte=tokens_per_byte,
                bytes_per_token=bytes_per_token,
                pieces_per_word_mean=subword_fertility,
                fragmentation_by_word_length_bucket=fragmentation_buckets,
            ),
            resources=BenchmarkResourceMetrics(
                peak_rss_mb=peak_rss_mb,
                memory_delta_mb=memory_delta_mb,
            ),
        )

    # -------------------------------------------------------------------------
    def _build_per_document_stats(
        self,
        tokenizer_name: str,
        data: pd.DataFrame,
        per_document_latency_ms: list[float | None],
    ) -> BenchmarkPerDocumentTokenizerStats:
        sorted_data = data.copy()
        if "text_id" in sorted_data.columns:
            sorted_data["text_id"] = pd.to_numeric(
                sorted_data["text_id"], errors="coerce"
            )
            sorted_data = sorted_data.sort_values("text_id")

        tokens_count = cast(
            pd.Series,
            pd.to_numeric(sorted_data["tokens_count"], errors="coerce"),
        ).fillna(0)
        bytes_per_token = cast(
            pd.Series,
            pd.to_numeric(sorted_data["bytes_per_token"], errors="coerce"),
        ).fillna(0)
        per_doc_latency_ms = list(per_document_latency_ms)
        per_doc_peak_rss: list[float | None] = [None] * len(sorted_data)
        if "pieces_per_word" in sorted_data.columns:
            pieces_series = cast(
                pd.Series,
                pd.to_numeric(sorted_data["pieces_per_word"], errors="coerce"),
            )
            pieces_per_word: list[float | None] = [
                None if pd.isna(value) else float(value)
                for value in pieces_series.tolist()
            ]
        else:
            pieces_per_word = [None] * len(sorted_data)

        return BenchmarkPerDocumentTokenizerStats(
            tokenizer=tokenizer_name,
            tokens_count=[int(value) for value in tokens_count.tolist()],
            bytes_per_token=[float(value) for value in bytes_per_token.tolist()],
            pieces_per_word=pieces_per_word,
            encode_latency_ms=per_doc_latency_ms,
            peak_rss_mb=per_doc_peak_rss,
        )

    # -------------------------------------------------------------------------
    def build_dashboard_data(
        self,
        tokenizer_results: list[BenchmarkTokenizerResult],
        raw_observations: dict[str, list[dict[str, object]]],
        per_document_stats: list[BenchmarkPerDocumentTokenizerStats] | None = None,
        selected_metric_keys: list[str] | None = None,
    ) -> BenchmarkDashboardData:
        successful = [result for result in tokenizer_results if result.status == "success"]
        selected = set(selected_metric_keys or [])
        per_document_by_tokenizer = {item.tokenizer: item for item in per_document_stats or []}
        widgets: list[BenchmarkDashboardWidgetData] = []
        available_keys: set[str] = set()
        for definition in BENCHMARK_METRIC_DEFINITIONS:
            points: list[BenchmarkDashboardPoint] = []
            distributions: list[BenchmarkDashboardDistribution] = []
            buckets: list[BenchmarkDashboardBucketPoint] = []
            histogram_bins: list[BenchmarkDashboardHistogramBin] = []
            if definition.distribution_source:
                values_by_tokenizer: dict[str, list[float]] = {}
                for result in successful:
                    values = self._dashboard_distribution_values(definition.distribution_source, result.tokenizer, raw_observations, per_document_by_tokenizer)
                    values_by_tokenizer[result.tokenizer] = values
                    summary = self._distribution_summary(values)
                    if summary is not None:
                        distributions.append(BenchmarkDashboardDistribution(tokenizer=result.tokenizer, **summary))
                histogram_bins = self._histogram_bins(values_by_tokenizer)
            elif definition.default_visualization.value == "grouped_bar":
                for result in successful:
                    for bucket in result.fragmentation.fragmentation_by_word_length_bucket:
                        if self._is_number(bucket.pieces_per_word_mean):
                            buckets.append(BenchmarkDashboardBucketPoint(tokenizer=result.tokenizer, bucket=bucket.bucket, value=float(bucket.pieces_per_word_mean)))
            else:
                for result in successful:
                    value = self._extract_path(result, definition.path)
                    if not self._is_number(value):
                        continue
                    low = self._extract_path(result, definition.interval_low_path) if definition.interval_low_path else None
                    high = self._extract_path(result, definition.interval_high_path) if definition.interval_high_path else None
                    points.append(BenchmarkDashboardPoint(tokenizer=result.tokenizer, value=float(value), interval_low=float(low) if self._is_number(low) else None, interval_high=float(high) if self._is_number(high) else None))
            if not points and not distributions and not buckets:
                continue
            tokenizer_count = len({point.tokenizer for point in points} | {point.tokenizer for point in distributions} | {point.tokenizer for point in buckets})
            bucket_count = len({point.bucket for point in buckets})
            data_width = "wide" if definition.width.value == "wide" or tokenizer_count > 4 or bucket_count > 5 else "standard"
            widgets.append(BenchmarkDashboardWidgetData(widget_id=definition.widget_id, metric_keys=list(definition.required_metric_keys), category_key=definition.category_key, category_label=definition.category_label, label=definition.label, description=definition.description, unit=definition.unit, display_format=definition.display_format, default_visualization=definition.default_visualization, compatible_visualizations=list(definition.compatible_visualizations), default_visible=definition.default_visible, width=data_width, points=points, distributions=distributions, buckets=buckets, histogram_bins=histogram_bins))
            available_keys.update(definition.required_metric_keys)
        return BenchmarkDashboardData(widgets=widgets, available_widget_ids=[widget.widget_id for widget in widgets], available_metric_keys=[definition.key for definition in BENCHMARK_METRIC_DEFINITIONS if definition.key in available_keys], unavailable_selected_metric_keys=[key for key in selected if key not in available_keys])

    # -------------------------------------------------------------------------
    def _extract_path(self, value: object, path: str | None) -> object | None:
        current = value
        for part in (path or "").split("."):
            if not part:
                continue
            current = getattr(current, part, None)
        return current

    # -------------------------------------------------------------------------
    def _is_number(self, value: object) -> TypeGuard[int | float]:
        return isinstance(value, int | float) and not isinstance(value, bool) and np.isfinite(float(value))

    # -------------------------------------------------------------------------
    def _distribution_summary(self, values: list[float]) -> dict[str, Any] | None:
        finite = [value for value in values if self._is_number(value)]
        if not finite:
            return None
        array = np.asarray(finite, dtype=float)
        return {"min": float(np.min(array)), "q1": float(np.percentile(array, 25)), "median": float(np.percentile(array, 50)), "q3": float(np.percentile(array, 75)), "max": float(np.max(array)), "sample_count": len(finite)}

    # -------------------------------------------------------------------------
    def _dashboard_distribution_values(self, source: str, tokenizer: str, raw: dict[str, list[dict[str, object]]], stats: dict[str, BenchmarkPerDocumentTokenizerStats]) -> list[float]:
        if source == "raw_latency":
            values: list[float] = []
            for row in raw.get(tokenizer, []):
                elapsed_ns = row.get("elapsed_ns")
                documents = row.get("documents")
                if not (self._is_number(elapsed_ns) and self._is_number(documents)):
                    continue
                values.append(
                    (float(elapsed_ns) / 1_000_000.0)
                    / max(1.0, float(documents))
                )
            return values
        values = getattr(stats.get(tokenizer), source, []) if tokenizer in stats else []
        return [float(value) for value in values if self._is_number(value)]

    # -------------------------------------------------------------------------
    def _histogram_bins(self, values_by_tokenizer: dict[str, list[float]]) -> list[BenchmarkDashboardHistogramBin]:
        finite = [value for values in values_by_tokenizer.values() for value in values if self._is_number(value)]
        if not finite:
            return []
        array = np.asarray(finite, dtype=float)
        low = float(np.min(array))
        high = float(np.max(array))
        if low == high:
            padding = max(abs(low) * 0.05, 0.5)
            edges = np.asarray([low - padding, high + padding], dtype=float)
        else:
            edges = np.asarray(np.histogram_bin_edges(array, bins="auto"), dtype=float)
            edges = np.unique(edges[np.isfinite(edges)])
            if len(edges) < 2:
                edges = np.asarray([low, high], dtype=float)
            if len(edges) - 1 > 24:
                edges = np.linspace(low, high, 25)
        bins: list[BenchmarkDashboardHistogramBin] = []
        for tokenizer, values in values_by_tokenizer.items():
            finite_values = np.asarray([value for value in values if self._is_number(value)], dtype=float)
            if finite_values.size == 0:
                continue
            counts, _ = np.histogram(finite_values, bins=edges)
            total = int(finite_values.size)
            for index, count in enumerate(counts.tolist()):
                bins.append(BenchmarkDashboardHistogramBin(tokenizer=tokenizer, bin_low=float(edges[index]), bin_high=float(edges[index + 1]), count=int(count), proportion=float(count / total)))
        return bins

    # -------------------------------------------------------------------------
