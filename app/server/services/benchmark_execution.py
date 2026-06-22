from __future__ import annotations

import platform
import re
import statistics
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from server.domain.benchmarks import (
    BenchmarkChartDataV2,
    BenchmarkDistributionPoint,
    BenchmarkEfficiencyMetrics,
    BenchmarkFidelityMetrics,
    BenchmarkFragmentationBucket,
    BenchmarkFragmentationMetrics,
    BenchmarkHardwareProfile,
    BenchmarkLatencyMetrics,
    BenchmarkPerDocumentTokenizerStats,
    BenchmarkResourceMetrics,
    BenchmarkRunConfig,
    BenchmarkRunResponse,
    BenchmarkSeriesPoint,
    BenchmarkTokenizerResult,
    BenchmarkTrialSummary,
)
from server.common.utils.logger import logger
from server.domain.benchmark_observations import TokenizerRunConfig
from server.services.benchmark_engine import (
    run_tokenizer_trials,
    summarize_observations,
)
from server.services.benchmark_metric_plan import build_metric_plan
from server.services.benchmark_metadata import collect_runtime_environment
from server.services.benchmark_spool import BenchmarkTextSpool
from server.services.benchmark_streams import iter_limited_rows
from server.services.tokenizer_adapters import UniversalTokenizerAdapter

###############################################################################
class BenchmarkCancelledError(RuntimeError):
    pass

###############################################################################
@dataclass(frozen=True)
class SpooledTextBatchFactory:
    spool: BenchmarkTextSpool
    batch_size: int

    # -------------------------------------------------------------------------
    def __call__(self) -> Any:
        return self.spool.iter_text_batches(self.batch_size)

###############################################################################
class BenchmarkServiceExecutionMixin:
    # Concrete host class provides these members; annotate for static analyzers.
    tools: Any
    max_documents: int
    log_interval: int
    resolve_selected_metric_keys: Callable[[list[str] | None], list[str]]
    get_dataset_document_count: Callable[[str], int]
    load_tokenizers: Callable[[list[str]], dict[str, Any]]
    stream_dataset_rows_from_database: Callable[[str], Any]
    repository: Any

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

        tokens_count = pd.to_numeric(
            sorted_data["tokens_count"], errors="coerce"
        ).fillna(0)
        bytes_per_token = pd.to_numeric(
            sorted_data["bytes_per_token"], errors="coerce"
        ).fillna(0)
        per_doc_latency_ms = list(per_document_latency_ms)
        per_doc_peak_rss: list[float | None] = [None] * len(sorted_data)
        if "pieces_per_word" in sorted_data.columns:
            pieces_series = pd.to_numeric(
                sorted_data["pieces_per_word"], errors="coerce"
            )
            pieces_per_word = [
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
    ) -> BenchmarkChartDataV2:
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

        return BenchmarkChartDataV2(
            efficiency=efficiency,
            fidelity=fidelity,
            vocabulary=vocabulary,
            fragmentation=fragmentation,
            latency_or_memory_distribution=latency_distribution,
        )

    # -------------------------------------------------------------------------
    def run_benchmarks(
        self,
        dataset_name: str,
        tokenizer_ids: list[str],
        custom_tokenizers: dict[str, Any] | None = None,
        run_name: str | None = None,
        selected_metric_keys: list[str] | None = None,
        benchmark_config: dict[str, Any] | None = None,
        progress_callback: Callable[[float], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> BenchmarkRunResponse | dict[str, Any]:
        logger.info("Starting benchmark run for dataset: %s", dataset_name)
        logger.info("Tokenizers to benchmark: %s", tokenizer_ids)
        self._raw_observations = {}
        resolved_metric_keys = self.resolve_selected_metric_keys(selected_metric_keys)
        normalized_run_name = run_name.strip() if isinstance(run_name, str) else ""

        doc_count = self.get_dataset_document_count(dataset_name)
        if doc_count == 0:
            raise ValueError(f"Dataset '{dataset_name}' not found or empty")
        if progress_callback:
            progress_callback(5.0)

        tokenizers = self.load_tokenizers(tokenizer_ids)
        if custom_tokenizers:
            for name, tok in custom_tokenizers.items():
                if self.tools.is_tokenizer_compatible(tok):
                    tokenizers[name] = tok
                    logger.info("Added custom tokenizer: %s", name)

        if not tokenizers:
            raise ValueError("No valid tokenizers could be loaded")

        logger.info("Loaded %d tokenizers", len(tokenizers))
        if progress_callback:
            progress_callback(10.0)

        logger.info("Preparing streamed dataset batches from database...")
        run_started_at = time.perf_counter()
        stream_started_at = time.perf_counter()
        config_payload = benchmark_config or {}
        warmup_trials = int(config_payload.get("warmup_trials", 2))
        timed_trials = int(config_payload.get("timed_trials", 8))
        batch_size = int(config_payload.get("batch_size", 16))
        add_special_tokens = bool(config_payload.get("add_special_tokens", False))
        padding = bool(config_payload.get("padding", False))
        truncation = bool(config_payload.get("truncation", False))
        max_length_value = config_payload.get("max_length")
        max_length = (
            int(max_length_value)
            if isinstance(max_length_value, int) and max_length_value > 0
            else None
        )
        store_per_document_stats = bool(
            config_payload.get("store_per_document_stats", True)
        )
        per_document_sample_size = int(
            config_payload.get("per_document_sample_size", 500)
        )
        if warmup_trials < 0:
            raise ValueError("warmup_trials must be non-negative")
        if timed_trials <= 0:
            raise ValueError("timed_trials must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        max_documents_limit = (
            int(self.max_documents) if int(self.max_documents) > 0 else None
        )
        limited_rows = iter_limited_rows(
            self.stream_dataset_rows_from_database(dataset_name),
            max_documents_limit,
        )
        spool = BenchmarkTextSpool()
        num_docs = 0
        dataset_total_chars = 0
        dataset_total_utf8_bytes = 0
        for row_id, text in limited_rows:
            if should_stop and should_stop():
                spool.cleanup()
                return BenchmarkRunResponse(
                    status="cancelled",
                    schema_version=1,
                    methodology_version="v1_observed_trials",
                    run_name=normalized_run_name or None,
                    selected_metric_keys=resolved_metric_keys,
                    dataset_name=dataset_name,
                    documents_processed=num_docs,
                    tokenizers_processed=[],
                    tokenizers_count=0,
                    config=BenchmarkRunConfig(),
                    hardware_profile=BenchmarkHardwareProfile(),
                    trial_summary=BenchmarkTrialSummary(),
                    tokenizer_results=[],
                    chart_data=BenchmarkChartDataV2(),
                    per_document_stats=[],
                    runtime_metadata={},
                    raw_observations={},
                )
            spool.append(row_id, text)
            num_docs += 1
            dataset_total_chars += len(text)
            dataset_total_utf8_bytes += len(text.encode("utf-8"))
        spool.finalize()
        dataset_stream_wall_time_seconds = max(
            0.0, time.perf_counter() - stream_started_at
        )
        metric_plan = build_metric_plan(
            resolved_metric_keys,
            store_per_document_stats=store_per_document_stats,
        )
        if progress_callback:
            progress_callback(20.0)

        tokenizer_results: list[BenchmarkTokenizerResult] = []
        per_document_stats: list[BenchmarkPerDocumentTokenizerStats] = []
        cancelled = False
        total_tokenizers = len(tokenizers)
        progress_base = 20.0
        progress_span = 80.0
        per_tokenizer_span = (
            progress_span / total_tokenizers if total_tokenizers > 0 else progress_span
        )

        for index, (name, tokenizer) in enumerate(tokenizers.items()):
            if should_stop and should_stop():
                cancelled = True
                break

            logger.info("Processing tokenizer: %s", name)
            try:
                tokenizer_started_at = time.perf_counter()
                adapter = UniversalTokenizerAdapter(
                    tokenizer_id=name, tokenizer=tokenizer
                )
                observations = run_tokenizer_trials(
                    tokenizer=adapter,
                    text_batches_factory=SpooledTextBatchFactory(spool, batch_size),
                    config=TokenizerRunConfig(
                        add_special_tokens=add_special_tokens,
                        padding=padding,
                        truncation=truncation,
                        max_length=max_length,
                        batch_size=batch_size,
                    ),
                    warmup_trials=warmup_trials,
                    timed_trials=timed_trials,
                    should_stop=should_stop,
                )
                if not observations:
                    if should_stop and should_stop():
                        raise BenchmarkCancelledError("Benchmark run cancelled.")
                    raise RuntimeError("No observations collected for tokenizer run.")
                observation_summary = summarize_observations(observations)
                trial_tokenization_speeds_tps: list[float] = []
                for trial_index in sorted({obs.trial_index for obs in observations}):
                    trial_obs = [
                        obs for obs in observations if obs.trial_index == trial_index
                    ]
                    trial_elapsed_ns = sum(obs.elapsed_ns for obs in trial_obs)
                    trial_tokens = sum(obs.token_count for obs in trial_obs)
                    trial_tokenization_speeds_tps.append(
                        float(trial_tokens / (trial_elapsed_ns / 1_000_000_000.0))
                        if trial_elapsed_ns > 0
                        else 0.0
                    )
                postprocess_started_at = time.perf_counter()
                sample_rows: list[dict[str, Any]] = []
                unknown_token_count_total = 0
                unknown_token_count_measurable = False
                total_tokens = 0
                total_chars = 0
                total_bytes = 0
                fragmentation_token_total = 0
                fragmentation_word_piece_counts: list[int] = []
                fragmentation_bucket_values: dict[str, list[float]] = {
                    "short_1_4": [],
                    "medium_5_8": [],
                    "long_9_plus": [],
                }
                round_trip_token_fidelity: list[float] = []
                round_trip_text_fidelity: list[float] = []
                for row_id, text_value in spool.iter_rows():
                    if should_stop and should_stop():
                        raise BenchmarkCancelledError("Benchmark run cancelled.")
                    encoded_batch = adapter.encode_batch(
                        [text_value],
                        add_special_tokens=add_special_tokens,
                        padding=padding,
                        truncation=truncation,
                        max_length=max_length,
                    )
                    encoded_ids = (
                        encoded_batch.input_ids_by_doc[0]
                        if encoded_batch.input_ids_by_doc
                        else []
                    )
                    token_count = len(encoded_ids)
                    total_tokens += token_count
                    total_chars += len(text_value)
                    total_bytes += len(text_value.encode("utf-8"))
                    if (
                        encoded_batch.unknown_counts
                        and encoded_batch.unknown_counts[0] is not None
                    ):
                        unknown_token_count_measurable = True
                        unknown_token_count_total += int(
                            encoded_batch.unknown_counts[0] or 0
                        )
                    pieces_per_word: float | None = None
                    if metric_plan.needs_fragmentation:
                        fragmentation_encoded = adapter.encode_batch(
                            [text_value],
                            add_special_tokens=False,
                            padding=False,
                            truncation=False,
                            max_length=None,
                        )
                        fragmentation_token_total += (
                            fragmentation_encoded.token_counts[0]
                            if fragmentation_encoded.token_counts
                            else 0
                        )
                        words = re.findall(r"\b\w+\b", text_value, flags=re.UNICODE)
                        if words:
                            encoded_words = adapter.encode_batch(
                                words,
                                add_special_tokens=False,
                                padding=False,
                                truncation=False,
                                max_length=None,
                            )
                            word_piece_counts = [
                                int(value)
                                for value in encoded_words.token_counts[: len(words)]
                            ]
                            fragmentation_word_piece_counts.extend(word_piece_counts)
                            for word, piece_count in zip(
                                words, word_piece_counts, strict=False
                            ):
                                fragmentation_bucket_values[
                                    self._fragmentation_bucket_label(len(word))
                                ].append(float(piece_count))
                            pieces_per_word = float(np.mean(word_piece_counts))
                    decoded_text = self.tools.safe_decode(tokenizer, encoded_ids)
                    if metric_plan.needs_round_trip:
                        rt_token_ids = self.tools.extract_token_ids(
                            tokenizer.encode(decoded_text)
                        )
                        round_trip_token_fidelity.append(
                            float(rt_token_ids == encoded_ids)
                        )
                        round_trip_text_fidelity.append(
                            float(
                                unicodedata.normalize("NFC", decoded_text)
                                == unicodedata.normalize("NFC", text_value)
                            )
                        )
                    if (
                        metric_plan.needs_per_document_stats
                        and len(sample_rows) < per_document_sample_size
                    ):
                        sample_rows.append(
                            {
                                "text_id": row_id,
                                "tokens_count": token_count,
                                "bytes_per_token": (
                                    len(text_value.encode("utf-8")) / token_count
                                )
                                if token_count > 0
                                else 0.0,
                                "pieces_per_word": pieces_per_word,
                            }
                        )

                elapsed = max(float(observation_summary["total_time_seconds"]), 1e-9)
                throughput_chars_per_sec = float(
                    observation_summary["documents_per_second"]
                ) * ((total_chars / num_docs) if num_docs else 0.0)
                throughput_bytes_per_sec = (
                    (float(total_bytes) / elapsed) if elapsed > 0 else 0.0
                )

                vocab_result = self._extract_vocab_result(tokenizer)
                if isinstance(vocab_result, (Mapping, Sequence)):
                    vocabulary_size = int(len(vocab_result))
                else:
                    vocabulary_size = 0

                fragmentation_buckets: list[BenchmarkFragmentationBucket] = []
                if metric_plan.needs_fragmentation:
                    subword_fertility = (
                        float(np.mean(fragmentation_word_piece_counts))
                        if fragmentation_word_piece_counts
                        else 0.0
                    )
                    fragmentation_buckets = [
                        BenchmarkFragmentationBucket(
                            bucket=bucket,
                            pieces_per_word_mean=float(np.mean(values)),
                        )
                        for bucket, values in fragmentation_bucket_values.items()
                        if values
                    ]
                else:
                    subword_fertility = 0.0

                oov_rate: float | None = None
                if metric_plan.needs_unknown_rate and total_tokens > 0:
                    if unknown_token_count_measurable:
                        oov_rate = unknown_token_count_total / total_tokens

                if metric_plan.needs_character_coverage:
                    vocab_tokens: set[str] = set()
                    if isinstance(vocab_result, Mapping):
                        vocab_tokens = {str(tok) for tok in vocab_result.keys()}
                    elif isinstance(vocab_result, Sequence):
                        vocab_tokens = {str(tok) for tok in vocab_result}

                    normalized_vocab_tokens = {
                        str(token).replace("##", "").lstrip("?").lstrip("G")
                        for token in vocab_tokens
                    }
                    dataset_chars = set()
                    for _, text_value in spool.iter_rows():
                        dataset_chars.update(text_value)
                    vocab_chars: set[str] = set()
                    for token_item in normalized_vocab_tokens:
                        for ch in token_item:
                            vocab_chars.add(ch)
                    intersection = dataset_chars.intersection(vocab_chars)
                    character_coverage: float | None = (
                        (len(intersection) / max(1, len(dataset_chars)) * 100.0)
                        if dataset_chars
                        else 0.0
                    )
                else:
                    character_coverage = None

                compression_chars_per_token = (
                    float(total_chars / fragmentation_token_total)
                    if fragmentation_token_total > 0
                    else 0.0
                )
                compression_bytes_per_character = (
                    float(fragmentation_token_total / total_bytes)
                    if total_bytes > 0
                    else 0.0
                )
                round_trip_fidelity_rate = (
                    float(np.mean(round_trip_token_fidelity))
                    if metric_plan.needs_round_trip and round_trip_token_fidelity
                    else 0.0
                )
                round_trip_text_rate = (
                    float(np.mean(round_trip_text_fidelity))
                    if metric_plan.needs_round_trip and round_trip_text_fidelity
                    else 0.0
                )
                postprocess_wall_time_seconds = max(
                    0.0, time.perf_counter() - postprocess_started_at
                )
                encode_only_wall_time_seconds = float(
                    observation_summary.get("total_time_seconds", 0.0) or 0.0
                )
                tokenizer_wall_time_seconds = max(
                    0.0, time.perf_counter() - tokenizer_started_at
                )
                tokenizer_results.append(
                    self._build_tokenizer_result(
                        tokenizer_name=name,
                        status="success",
                        trial_tokenization_speeds_tps=trial_tokenization_speeds_tps,
                        throughput_chars_per_sec=float(throughput_chars_per_sec),
                        encode_only_wall_time_seconds=float(
                            encode_only_wall_time_seconds
                        ),
                        dataset_stream_wall_time_seconds=float(
                            dataset_stream_wall_time_seconds
                        ),
                        postprocess_wall_time_seconds=float(
                            postprocess_wall_time_seconds
                        ),
                        total_processing_time_seconds=float(
                            tokenizer_wall_time_seconds
                        ),
                        observed_latency_ms=[
                            (obs.elapsed_ns / 1_000_000.0) / max(1, obs.documents)
                            for obs in observations
                        ]
                        if metric_plan.needs_latency
                        else [],
                        latency_sample_count=(
                            len(observations) if metric_plan.needs_latency else 0
                        ),
                        vocabulary_size=int(vocabulary_size),
                        oov_rate=oov_rate,
                        character_coverage=character_coverage,
                        round_trip_fidelity_rate=float(round_trip_fidelity_rate),
                        round_trip_text_fidelity_rate=float(round_trip_text_rate),
                        subword_fertility=float(subword_fertility),
                        compression_chars_per_token=float(compression_chars_per_token),
                        compression_bytes_per_character=float(
                            compression_bytes_per_character
                        ),
                        fragmentation_buckets=fragmentation_buckets,
                        peak_rss_mb=max(
                            (float(obs.peak_rss_mb or 0.0) for obs in observations),
                            default=0.0,
                        ),
                        memory_delta_mb=max(
                            0.0,
                            max(
                                (float(obs.peak_rss_mb or 0.0) for obs in observations),
                                default=0.0,
                            )
                            - min(
                                (float(obs.peak_rss_mb or 0.0) for obs in observations),
                                default=0.0,
                            ),
                        ),
                    )
                )
                tokenizer_results[-1].efficiency.encode_bytes_per_second_mean = float(
                    throughput_bytes_per_sec
                )

                if metric_plan.needs_per_document_stats:
                    sampled_data = pd.DataFrame(sample_rows)
                    per_document_stats.append(
                        self._build_per_document_stats(
                            tokenizer_name=name,
                            data=sampled_data,
                            per_document_latency_ms=[
                                None for _ in range(len(sampled_data))
                            ],
                        )
                    )
                self._raw_observations[name] = [
                    {
                        "trial_index": int(obs.trial_index),
                        "batch_index": int(obs.batch_index),
                        "documents": int(obs.documents),
                        "elapsed_ns": int(obs.elapsed_ns),
                        "token_count": int(obs.token_count),
                        "input_utf8_bytes": int(obs.input_utf8_bytes),
                    }
                    for obs in observations
                ]
            except Exception as exc:
                if isinstance(exc, BenchmarkCancelledError):
                    cancelled = True
                    break
                logger.exception("Benchmark failed for tokenizer %s", name)
                tokenizer_results.append(
                    self._build_tokenizer_result(
                        tokenizer_name=name,
                        status="failed",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        trial_tokenization_speeds_tps=[],
                        throughput_chars_per_sec=0.0,
                        encode_only_wall_time_seconds=0.0,
                        dataset_stream_wall_time_seconds=float(
                            dataset_stream_wall_time_seconds
                        ),
                        postprocess_wall_time_seconds=0.0,
                        total_processing_time_seconds=0.0,
                        observed_latency_ms=[],
                        latency_sample_count=0,
                        vocabulary_size=0,
                        oov_rate=None,
                        character_coverage=None,
                        round_trip_fidelity_rate=0.0,
                        round_trip_text_fidelity_rate=0.0,
                        subword_fertility=0.0,
                        compression_chars_per_token=0.0,
                        compression_bytes_per_character=0.0,
                        fragmentation_buckets=[],
                        peak_rss_mb=0.0,
                        memory_delta_mb=0.0,
                    )
                )
                self._raw_observations[name] = [
                    {
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }
                ]

            if progress_callback:
                progress_callback(progress_base + ((index + 1) * per_tokenizer_span))

        config = BenchmarkRunConfig.model_validate(
            {
                "max_documents": int(self.max_documents),
                "warmup_trials": int((benchmark_config or {}).get("warmup_trials", 2)),
                "timed_trials": int((benchmark_config or {}).get("timed_trials", 8)),
                "batch_size": int((benchmark_config or {}).get("batch_size", 16)),
                "seed": int((benchmark_config or {}).get("seed", 42)),
                "parallelism": int((benchmark_config or {}).get("parallelism", 1)),
                "include_lm_metrics": bool(
                    (benchmark_config or {}).get("include_lm_metrics", False)
                ),
                "add_special_tokens": add_special_tokens,
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
                "store_per_document_stats": store_per_document_stats,
                "per_document_sample_size": per_document_sample_size,
            }
        )

        hardware_profile = BenchmarkHardwareProfile(
            runtime=platform.python_version(),
            os=platform.platform(),
            cpu_model=platform.processor() or None,
            cpu_logical_cores=None,
            memory_total_mb=None,
        )

        chart_data = self._build_chart_data(
            tokenizer_results,
            getattr(self, "_raw_observations", {}),
        )

        runtime_metadata = collect_runtime_environment()
        runtime_metadata["dataset_total_documents_available"] = int(doc_count)
        runtime_metadata["dataset_documents_benchmarked"] = int(num_docs)
        runtime_metadata["dataset_total_chars"] = int(dataset_total_chars)
        runtime_metadata["dataset_total_utf8_bytes"] = int(dataset_total_utf8_bytes)
        runtime_metadata["benchmark_config"] = {
            "warmup_trials": warmup_trials,
            "timed_trials": timed_trials,
            "batch_size": batch_size,
            "max_documents": max_documents_limit,
            "add_special_tokens": add_special_tokens,
            "padding": padding,
            "truncation": truncation,
            "max_length": max_length,
            "store_per_document_stats": store_per_document_stats,
            "per_document_sample_size": per_document_sample_size,
        }
        runtime_metadata["tokenizer_metadata"] = [
            {
                "tokenizer_name": tokenizer_name,
                "tokenizer_class": type(tokenizer_obj).__name__,
                "is_fast": bool(getattr(tokenizer_obj, "is_fast", False)),
                "name_or_path": getattr(tokenizer_obj, "name_or_path", None),
                "model_max_length": getattr(tokenizer_obj, "model_max_length", None),
                "special_tokens_map": getattr(tokenizer_obj, "special_tokens_map", {})
                or {},
            }
            for tokenizer_name, tokenizer_obj in tokenizers.items()
        ]
        runtime_metadata["benchmark_timing_boundaries"] = {
            "encode_only_definition": "Timed tokenizer encode trials only; warmup excluded.",
            "dataset_stream_definition": "Time spent streaming and spooling dataset rows before tokenizer trials.",
            "postprocess_definition": "Time spent computing post-trial metrics from replayed rows.",
            "latency_summary_definition": "Latency percentiles and distributions are computed from all timed batch observations, normalized by documents per batch.",
        }
        successful_results = [
            result for result in tokenizer_results if result.status == "success"
        ]
        resource_metrics_available = any(
            (
                result.resources.peak_rss_mb > 0.0
                or result.resources.memory_delta_mb > 0.0
            )
            for result in successful_results
        )
        runtime_metadata["metric_availability"] = {
            "resource_metrics": resource_metrics_available,
            "latency_distribution": bool(
                metric_plan.needs_latency and len(successful_results) > 0
            ),
            "byte_fallback_rate": False,
            "unknown_token_rate": any(
                result.fidelity.unknown_token_rate is not None
                for result in successful_results
            ),
            "vocab_character_overlap": any(
                result.fidelity.lossless_encodability_rate is not None
                for result in successful_results
            ),
            "fragmentation_word_length_bucket": any(
                bool(result.fragmentation.fragmentation_by_word_length_bucket)
                for result in successful_results
            ),
            "per_document_stats": bool(
                metric_plan.needs_per_document_stats and len(per_document_stats) > 0
            ),
        }
        runtime_metadata["end_to_end_benchmark_seconds"] = max(
            0.0, time.perf_counter() - run_started_at
        )
        spool.cleanup()

        return BenchmarkRunResponse(
            status="cancelled" if cancelled else "success",
            schema_version=1,
            methodology_version="v2_semantic_honesty",
            run_name=normalized_run_name or None,
            selected_metric_keys=resolved_metric_keys,
            dataset_name=dataset_name,
            documents_processed=num_docs,
            tokenizers_processed=list(tokenizers.keys()),
            tokenizers_count=len(tokenizers),
            config=config,
            hardware_profile=hardware_profile,
            trial_summary=BenchmarkTrialSummary(
                warmup_trials=config.warmup_trials,
                timed_trials=config.timed_trials,
            ),
            tokenizer_results=tokenizer_results,
            chart_data=chart_data,
            per_document_stats=per_document_stats,
            runtime_metadata=runtime_metadata,
            raw_observations=getattr(self, "_raw_observations", {}),
        )

    # -------------------------------------------------------------------------
    def get_dataset_id(self, dataset_name: str) -> int | None:
        return self.repository.get_dataset_id(dataset_name)

    # -------------------------------------------------------------------------
    def ensure_tokenizer_ids(self, tokenizer_names: list[str]) -> dict[str, int]:
        return self.repository.ensure_tokenizer_ids(tokenizer_names)
