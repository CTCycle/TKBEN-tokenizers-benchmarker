from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from server.common.utils.logger import logger
from server.domain.benchmarks import (
    BenchmarkChartData,
    BenchmarkDistributionPoint,
    BenchmarkEfficiencyMetrics,
    BenchmarkFidelityMetrics,
    BenchmarkFragmentationBucket,
    BenchmarkFragmentationMetrics,
    BenchmarkLatencyMetrics,
    BenchmarkPerDocumentTokenizerStats,
    BenchmarkResourceMetrics,
    BenchmarkSeriesPoint,
    BenchmarkTokenizerResult,
)

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
    def tokenize_document(
        self,
        tokenizer: Any,
        text_value: str,
        uses_tokenize: bool,
        tokenize_method: Callable[[Any], Any] | None,
    ) -> tuple[str, list[str]]:
        tokens_list: list[str] = []
        decoded_text = ""

        if uses_tokenize and tokenize_method is not None:
            try:
                raw_tokens = tokenize_method(text_value)
                tokens_list = self.tools.normalize_token_output(raw_tokens)
            except Exception:
                logger.debug(
                    "Tokenizer %s raised an exception while tokenizing text",
                    getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
                    exc_info=True,
                )
                tokens_list = []

        if not tokens_list:
            decoded_text, tokens_list = self.tools.process_tokens(text_value, tokenizer)
        else:
            decoded_text = " ".join(tokens_list)

        return decoded_text, tokens_list

    # -------------------------------------------------------------------------
    def calculate_morphological_consistency(
        self, tokenizer: Any, base_words: set[str]
    ) -> float:
        if not base_words or not self.tools.is_tokenizer_compatible(tokenizer):
            return 0.0

        selected_words = [w for w in sorted(base_words) if re.match(r"^[A-Za-z]+$", w)][
            :200
        ]
        if not selected_words:
            return 0.0

        scores: list[float] = []
        for word in selected_words:
            base_tokens = self.tools.process_tokens(word, tokenizer)[1]
            if not base_tokens:
                continue

            for variant in (f"{word}s", f"{word}ed", f"{word}ing"):
                variant_tokens = self.tools.process_tokens(variant, tokenizer)[1]
                if not variant_tokens:
                    continue
                score = self.tools.jaccard_similarity(base_tokens, variant_tokens)
                scores.append(score)

        if not scores:
            return 0.0

        return float(np.mean(scores))

    # -------------------------------------------------------------------------
    def calculate_token_id_monotonicity(
        self, vocab_result: Mapping[Any, Any] | Sequence[Any] | None
    ) -> float:
        token_id_pairs: list[tuple[int, str]] = []
        if isinstance(vocab_result, Mapping):
            for token, idx in vocab_result.items():
                try:
                    token_id_pairs.append((int(idx), str(token)))
                except Exception:
                    continue
        elif isinstance(vocab_result, Sequence) and not isinstance(
            vocab_result, (str, bytes)
        ):
            for idx, token in enumerate(vocab_result):
                token_id_pairs.append((idx, str(token)))

        if not token_id_pairs:
            return 0.0

        token_id_pairs.sort(key=lambda pair: pair[0])
        lengths = [len(tok) for _, tok in token_id_pairs]
        if len(lengths) < 2:
            return 1.0

        monotonic_steps = sum(
            1 for i in range(1, len(lengths)) if lengths[i] >= lengths[i - 1]
        )

        return monotonic_steps / (len(lengths) - 1)

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
        round_trip_fidelity_rate: float,
        round_trip_text_fidelity_rate: float,
        subword_fertility: float,
        compression_chars_per_token: float,
        compression_bytes_per_character: float,
        fragmentation_buckets: list[BenchmarkFragmentationBucket],
        peak_rss_mb: float = 0.0,
        memory_delta_mb: float = 0.0,
    ) -> BenchmarkTokenizerResult:
        chars_per_token = float(compression_chars_per_token)
        tokens_per_character = (1.0 / chars_per_token) if chars_per_token > 0 else 0.0
        tokens_per_byte = float(compression_bytes_per_character)
        bytes_per_token = (1.0 / tokens_per_byte) if tokens_per_byte > 0 else 0.0
        tokenization_speed_tps = (
            float(np.mean(trial_tokenization_speeds_tps))
            if trial_tokenization_speeds_tps
            else 0.0
        )
        ci95_half_width = self._ci95_half_width(trial_tokenization_speeds_tps)
        latency_p50 = self._percentile(observed_latency_ms, 50.0)
        latency_p95 = self._percentile(observed_latency_ms, 95.0)
        latency_p99 = self._percentile(observed_latency_ms, 99.0)

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
                exact_round_trip_rate=float(round_trip_fidelity_rate),
                normalized_round_trip_rate=float(round_trip_text_fidelity_rate),
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
                tokens_per_character=float(tokens_per_character),
                characters_per_token=chars_per_token,
                tokens_per_byte=tokens_per_byte,
                bytes_per_token=float(bytes_per_token),
                pieces_per_word_mean=float(subword_fertility),
                fragmentation_by_word_length_bucket=fragmentation_buckets,
            ),
            resources=BenchmarkResourceMetrics(
                peak_rss_mb=float(peak_rss_mb),
                memory_delta_mb=float(memory_delta_mb),
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
    def _build_chart_data(
        self,
        tokenizer_results: list[BenchmarkTokenizerResult],
        raw_observations: dict[str, list[dict[str, object]]],
    ) -> BenchmarkChartData:
        successful_results = [
            result for result in tokenizer_results if result.status == "success"
        ]
        efficiency = [
            BenchmarkSeriesPoint(
                tokenizer=result.tokenizer,
                value=result.efficiency.encode_tokens_per_second_mean,
                ci95_low=result.efficiency.encode_tokens_per_second_ci95_low,
                ci95_high=result.efficiency.encode_tokens_per_second_ci95_high,
            )
            for result in successful_results
        ]
        fidelity = [
            BenchmarkSeriesPoint(
                tokenizer=result.tokenizer,
                value=result.fidelity.exact_round_trip_rate,
            )
            for result in successful_results
        ]
        vocabulary = [
            BenchmarkSeriesPoint(
                tokenizer=result.tokenizer,
                value=float(result.vocabulary_size),
            )
            for result in successful_results
        ]
        fragmentation = [
            BenchmarkSeriesPoint(
                tokenizer=result.tokenizer,
                value=result.fragmentation.pieces_per_word_mean,
            )
            for result in successful_results
        ]
        latency_distribution: list[BenchmarkDistributionPoint] = []
        for result in successful_results:
            rows = raw_observations.get(result.tokenizer, [])
            latencies_ms: list[float] = []
            for row in rows:
                elapsed_ns = row.get("elapsed_ns") if isinstance(row, dict) else None
                documents = row.get("documents") if isinstance(row, dict) else None
                if isinstance(elapsed_ns, int | float) and isinstance(
                    documents, int | float
                ):
                    docs = max(1.0, float(documents))
                    latencies_ms.append((float(elapsed_ns) / 1_000_000.0) / docs)
            if not latencies_ms:
                latencies_ms = [0.0]
            arr = np.asarray(latencies_ms, dtype=float)
            latency_distribution.append(
                BenchmarkDistributionPoint(
                    tokenizer=result.tokenizer,
                    min=float(np.min(arr)),
                    q1=float(np.percentile(arr, 25)),
                    median=float(np.percentile(arr, 50)),
                    q3=float(np.percentile(arr, 75)),
                    max=float(np.max(arr)),
                    sample_count=len(latencies_ms),
                )
            )

        return BenchmarkChartData(
            efficiency=efficiency,
            fidelity=fidelity,
            vocabulary=vocabulary,
            fragmentation=fragmentation,
            latency_or_memory_distribution=latency_distribution,
        )

    # -------------------------------------------------------------------------
