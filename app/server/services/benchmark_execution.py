from __future__ import annotations

import platform
import re
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
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
from server.services.benchmark_engine import run_tokenizer_trials, summarize_observations
from server.services.benchmark_metadata import collect_runtime_environment
from server.services.tokenizer_adapters import UniversalTokenizerAdapter


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
        trial_tokenization_speeds_tps: list[float],
        throughput_chars_per_sec: float,
        total_processing_time_seconds: float,
        observed_latency_ms: list[float],
        vocabulary_size: int,
        oov_rate: float,
        character_coverage: float,
        round_trip_fidelity_rate: float,
        round_trip_text_fidelity_rate: float,
        subword_fertility: float,
        compression_chars_per_token: float,
        compression_bytes_per_character: float,
    ) -> BenchmarkTokenizerResult:
        chars_per_token = float(compression_chars_per_token)
        tokens_per_character = (1.0 / chars_per_token) if chars_per_token > 0 else 0.0
        tokens_per_byte = float(compression_bytes_per_character)
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
                end_to_end_wall_time_seconds=float(total_processing_time_seconds),
                load_time_seconds=0.0,
            ),
            latency=BenchmarkLatencyMetrics(
                encode_latency_p50_ms=latency_p50,
                encode_latency_p95_ms=latency_p95,
                encode_latency_p99_ms=latency_p99,
            ),
            fidelity=BenchmarkFidelityMetrics(
                exact_round_trip_rate=float(round_trip_fidelity_rate),
                normalized_round_trip_rate=float(round_trip_text_fidelity_rate),
                unknown_token_rate=float(oov_rate),
                byte_fallback_rate=0.0,
                lossless_encodability_rate=float(character_coverage),
            ),
            fragmentation=BenchmarkFragmentationMetrics(
                tokens_per_character=float(tokens_per_character),
                characters_per_token=chars_per_token,
                tokens_per_byte=tokens_per_byte,
                bytes_per_token=chars_per_token,
                pieces_per_word_mean=float(subword_fertility),
                fragmentation_by_word_length_bucket=[
                    BenchmarkFragmentationBucket(
                        bucket="short_1_4",
                        pieces_per_word_mean=float(subword_fertility),
                    ),
                    BenchmarkFragmentationBucket(
                        bucket="medium_5_8",
                        pieces_per_word_mean=float(subword_fertility),
                    ),
                    BenchmarkFragmentationBucket(
                        bucket="long_9_plus",
                        pieces_per_word_mean=float(subword_fertility),
                    ),
                ],
            ),
            resources=BenchmarkResourceMetrics(
                peak_rss_mb=0.0,
                memory_delta_mb=0.0,
            ),
        )

    # -------------------------------------------------------------------------
    def _build_per_document_stats(
        self,
        tokenizer_name: str,
        data: pd.DataFrame,
        per_document_latency_ms: list[float],
    ) -> BenchmarkPerDocumentTokenizerStats:
        sorted_data = data.copy()
        if "text_id" in sorted_data.columns:
            sorted_data["text_id"] = pd.to_numeric(sorted_data["text_id"], errors="coerce")
            sorted_data = sorted_data.sort_values("text_id")

        tokens_count = pd.to_numeric(sorted_data["tokens_count"], errors="coerce").fillna(0)
        bytes_per_token = pd.to_numeric(sorted_data["bytes_per_token"], errors="coerce").fillna(0)
        per_doc_latency_ms = list(per_document_latency_ms)
        per_doc_peak_rss = [0.0] * len(sorted_data)

        return BenchmarkPerDocumentTokenizerStats(
            tokenizer=tokenizer_name,
            tokens_count=[int(value) for value in tokens_count.tolist()],
            bytes_per_token=[float(value) for value in bytes_per_token.tolist()],
            encode_latency_ms=per_doc_latency_ms,
            peak_rss_mb=per_doc_peak_rss,
        )

    # -------------------------------------------------------------------------
    def _build_chart_data(
        self,
        tokenizer_results: list[BenchmarkTokenizerResult],
    ) -> BenchmarkChartDataV2:
        efficiency = [
            BenchmarkSeriesPoint(
                tokenizer=result.tokenizer,
                value=result.efficiency.encode_tokens_per_second_mean,
                ci95_low=result.efficiency.encode_tokens_per_second_ci95_low,
                ci95_high=result.efficiency.encode_tokens_per_second_ci95_high,
            )
            for result in tokenizer_results
        ]
        fidelity = [
            BenchmarkSeriesPoint(
                tokenizer=result.tokenizer,
                value=result.fidelity.exact_round_trip_rate,
            )
            for result in tokenizer_results
        ]
        vocabulary = [
            BenchmarkSeriesPoint(
                tokenizer=result.tokenizer,
                value=float(result.vocabulary_size),
            )
            for result in tokenizer_results
        ]
        fragmentation = [
            BenchmarkSeriesPoint(
                tokenizer=result.tokenizer,
                value=result.fragmentation.pieces_per_word_mean,
            )
            for result in tokenizer_results
        ]
        latency_distribution = [
            BenchmarkDistributionPoint(
                tokenizer=result.tokenizer,
                min=result.latency.encode_latency_p50_ms,
                q1=result.latency.encode_latency_p50_ms,
                median=result.latency.encode_latency_p95_ms,
                q3=result.latency.encode_latency_p95_ms,
                max=result.latency.encode_latency_p99_ms,
            )
            for result in tokenizer_results
        ]

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
        config_payload = benchmark_config or {}
        warmup_trials = int(config_payload.get("warmup_trials", 2))
        timed_trials = int(config_payload.get("timed_trials", 8))
        batch_size = int(config_payload.get("batch_size", 16))
        if warmup_trials < 0:
            raise ValueError("warmup_trials must be non-negative")
        if timed_trials <= 0:
            raise ValueError("timed_trials must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        text_batches_for_first_trial: list[list[tuple[int, str]]] = []
        total_docs = 0
        current_batch: list[tuple[int, str]] = []
        for row_id, text in self.stream_dataset_rows_from_database(dataset_name):
            if should_stop and should_stop():
                return {}
            current_batch.append((int(row_id), str(text)))
            total_docs += 1
            if len(current_batch) >= batch_size:
                text_batches_for_first_trial.append(current_batch)
                current_batch = []
        if current_batch:
            text_batches_for_first_trial.append(current_batch)
        num_docs = total_docs
        if progress_callback:
            progress_callback(20.0)

        tokenizer_results: list[BenchmarkTokenizerResult] = []
        per_document_stats: list[BenchmarkPerDocumentTokenizerStats] = []
        total_tokenizers = len(tokenizers)
        progress_base = 20.0
        progress_span = 80.0
        per_tokenizer_span = (
            progress_span / total_tokenizers if total_tokenizers > 0 else progress_span
        )

        for index, (name, tokenizer) in enumerate(tokenizers.items()):
            if should_stop and should_stop():
                return {}

            logger.info("Processing tokenizer: %s", name)
            trial_rows = [row for batch in text_batches_for_first_trial for row in batch]
            text_ids = [row[0] for row in trial_rows]
            texts = [row[1] for row in trial_rows]
            data = pd.DataFrame({"tokenizer": [name] * len(texts), "name": [dataset_name] * len(texts), "text_id": text_ids, "text": texts})
            data["num_characters"] = pd.Series(texts).str.len()
            data["words_split"] = pd.Series(texts).str.split()
            data["words_count"] = data["words_split"].apply(len)

            adapter = UniversalTokenizerAdapter(tokenizer_id=name, tokenizer=tokenizer)

            def _text_batches_factory() -> list[list[str]]:
                return [[text for _, text in batch] for batch in text_batches_for_first_trial]

            observations = run_tokenizer_trials(
                tokenizer=adapter,
                text_batches_factory=_text_batches_factory,
                config=TokenizerRunConfig(
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    max_length=None,
                    batch_size=batch_size,
                ),
                warmup_trials=warmup_trials,
                timed_trials=timed_trials,
            )
            observation_summary = summarize_observations(observations)
            first_trial_batches = [obs for obs in observations if obs.trial_index == 0]
            encoded_once = []
            for text_value in texts:
                encoded_once.append(self.tools.process_tokens(text_value, tokenizer)[1])
            data["tokens"] = [" ".join(tokens) for tokens in encoded_once]
            data["tokens_split"] = encoded_once

            data["tokens_count"] = [
                len(toks) if isinstance(toks, (list, tuple)) else 0
                for toks in data["tokens_split"]
            ]
            data["tokens_to_words_ratio"] = np.where(
                data["words_count"] > 0,
                data["tokens_count"] / data["words_count"],
                0,
            )
            data["bytes_per_token"] = np.where(
                data["tokens_count"] > 0,
                data["num_characters"] / data["tokens_count"],
                0,
            )

            boundary_preservation: list[float] = []
            round_trip_token_fidelity: list[float] = []
            round_trip_text_fidelity: list[float] = []
            determinism_flags: list[float] = []
            bytes_per_character: list[float] = []
            total_bytes = 0

            for text_value, decoded_text, tokens_list in zip(
                texts, data["tokens"].tolist(), data["tokens_split"].tolist()
            ):
                if should_stop and should_stop():
                    return {}

                total_bytes += len(str(text_value).encode("utf-8"))
                boundary_preservation.append(
                    self.tools.boundary_preservation_score(text_value, decoded_text)
                )

                determinism_tokens = self.tools.process_tokens(text_value, tokenizer)[1]
                determinism_flags.append(float(determinism_tokens == tokens_list))

                rt_decoded, rt_tokens = self.tools.process_tokens(decoded_text, tokenizer)
                round_trip_token_fidelity.append(float(rt_tokens == tokens_list))
                round_trip_text_fidelity.append(float(rt_decoded == text_value))

                text_len = len(text_value)
                if text_len > 0:
                    bytes_per_character.append(
                        float(len(text_value.encode("utf-8")) / text_len)
                    )
                else:
                    bytes_per_character.append(0.0)

            data["boundary_preservation_rate"] = boundary_preservation
            data["round_trip_token_fidelity"] = round_trip_token_fidelity
            data["round_trip_text_fidelity"] = round_trip_text_fidelity
            data["determinism_stability"] = determinism_flags
            data["bytes_per_character"] = bytes_per_character

            elapsed = max(float(observation_summary["total_time_seconds"]), 1e-9)
            total_tokens = int(data["tokens_count"].sum())
            total_chars = int(data["num_characters"].sum())
            throughput_chars_per_sec = float(observation_summary["documents_per_second"]) * (
                (total_chars / len(texts)) if texts else 0.0
            )

            vocab_result = self._extract_vocab_result(tokenizer)
            if isinstance(vocab_result, (Mapping, Sequence)):
                vocabulary_size = int(len(vocab_result))
            else:
                vocabulary_size = 0

            words_per_doc = data["words_count"].replace(0, np.nan)
            tokens_per_doc = data["tokens_count"]
            fertility_series = (
                (tokens_per_doc / words_per_doc)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )
            subword_fertility = float(fertility_series.mean())

            all_words = [w for lst in data["words_split"] for w in lst]
            unique_words = set(all_words)

            vocab_tokens: set[str] = set()
            if isinstance(vocab_result, Mapping):
                vocab_tokens = {str(tok) for tok in vocab_result.keys()}
            elif isinstance(vocab_result, Sequence):
                vocab_tokens = {str(tok) for tok in vocab_result}

            normalized_vocab_tokens = {
                str(token).replace("##", "").lstrip("?").lstrip("G")
                for token in vocab_tokens
            }
            oov_words = {word for word in unique_words if word not in vocab_tokens}
            oov_rate = (
                (len(oov_words) / len(unique_words) * 100.0) if unique_words else 0.0
            )

            dataset_chars = set("".join(texts))
            vocab_chars: set[str] = set()
            for token_item in normalized_vocab_tokens:
                for ch in token_item:
                    vocab_chars.add(ch)
            intersection = dataset_chars.intersection(vocab_chars)
            character_coverage = (
                (len(intersection) / max(1, len(dataset_chars)) * 100.0)
                if dataset_chars
                else 0.0
            )

            compression_chars_per_token = (
                float(total_chars / total_tokens) if total_tokens > 0 else 0.0
            )
            compression_bytes_per_character = (
                float(total_tokens / total_bytes) if total_bytes > 0 else 0.0
            )
            round_trip_fidelity_rate = (
                float(np.mean(round_trip_token_fidelity)) if round_trip_token_fidelity else 0.0
            )
            round_trip_text_rate = (
                float(np.mean(round_trip_text_fidelity)) if round_trip_text_fidelity else 0.0
            )
            tokenizer_results.append(
                self._build_tokenizer_result(
                    tokenizer_name=name,
                    trial_tokenization_speeds_tps=[
                        float(observation_summary["tokens_per_second"])
                    ],
                    throughput_chars_per_sec=float(throughput_chars_per_sec),
                    total_processing_time_seconds=float(elapsed),
                    observed_latency_ms=[
                        (obs.elapsed_ns / 1_000_000.0) / max(1, obs.documents)
                        for obs in first_trial_batches
                    ],
                    vocabulary_size=int(vocabulary_size),
                    oov_rate=float(oov_rate),
                    character_coverage=float(character_coverage),
                    round_trip_fidelity_rate=float(round_trip_fidelity_rate),
                    round_trip_text_fidelity_rate=float(round_trip_text_rate),
                    subword_fertility=float(subword_fertility),
                    compression_chars_per_token=float(compression_chars_per_token),
                    compression_bytes_per_character=float(compression_bytes_per_character),
                )
            )

            per_document_stats.append(
                self._build_per_document_stats(
                    tokenizer_name=name,
                    data=data,
                    per_document_latency_ms=[0.0 for _ in texts],
                )
            )
            if not hasattr(self, "_raw_observations"):
                self._raw_observations = {}
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
                "include_lm_metrics": bool((benchmark_config or {}).get("include_lm_metrics", False)),
            }
        )

        hardware_profile = BenchmarkHardwareProfile(
            runtime=platform.python_version(),
            os=platform.platform(),
            cpu_model=platform.processor() or None,
            cpu_logical_cores=None,
            memory_total_mb=None,
        )

        chart_data = self._build_chart_data(tokenizer_results)

        return BenchmarkRunResponse(
            status="success",
            schema_version=1,
            methodology_version="v1_observed_trials",
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
            runtime_metadata=collect_runtime_environment(),
            raw_observations=getattr(self, "_raw_observations", {}),
        )

    # -------------------------------------------------------------------------
    def get_dataset_id(self, dataset_name: str) -> int | None:
        return self.repository.get_dataset_id(dataset_name)

    # -------------------------------------------------------------------------
    def ensure_tokenizer_ids(self, tokenizer_names: list[str]) -> dict[str, int]:
        return self.repository.ensure_tokenizer_ids(tokenizer_names)
