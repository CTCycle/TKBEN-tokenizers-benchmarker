from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from TKBEN.server.repositories.database.backend import database
from TKBEN.server.repositories.schemas.models import (
    Dataset,
    TokenizationDatasetStats,
    TokenizationDatasetStatsDetail,
    TokenizationDocumentStats,
    Tokenizer,
    TokenizerVocabulary,
    TokenizerVocabularyStatistics,
)
from TKBEN.server.common.utils.logger import logger


###############################################################################
class BenchmarkServiceExecutionMixin:
    # Concrete host class provides these members; annotate for static analyzers.
    tools: Any
    log_interval: int
    resolve_selected_metric_keys: Callable[[list[str] | None], list[str]]
    get_dataset_document_count: Callable[[str], int]
    load_tokenizers: Callable[[list[str]], dict[str, Any]]
    stream_dataset_rows_from_database: Callable[[str], Any]
    build_per_document_stats: Callable[[pd.DataFrame], list[dict[str, Any]]]

    # -------------------------------------------------------------------------
    def _session(self) -> Session:
        return Session(bind=database.backend.engine)

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
    def calculate_vocabulary_statistics(
        self, tokenizers: dict[str, Any]
    ) -> tuple[list[Any], pd.DataFrame]:
        vocabulary_stats: list[dict[str, Any]] = []
        vocabularies: list[pd.DataFrame] = []

        for name, tokenizer in tokenizers.items():
            if not self.tools.is_tokenizer_compatible(tokenizer):
                logger.warning("Skipping tokenizer %s for vocab stats", name)
                continue

            tokenizer_label = getattr(tokenizer, "name_or_path", name)
            try:
                vocab_func = getattr(tokenizer, "get_vocab", None)
                if not callable(vocab_func):
                    logger.warning(
                        "Tokenizer %s does not expose get_vocab", tokenizer_label
                    )
                    continue

                raw_vocab = vocab_func()
                if not isinstance(raw_vocab, dict):
                    continue

                vocab_words = [str(word) for word in raw_vocab.keys()]
                try:
                    vocab_indices = [int(idx) for idx in raw_vocab.values()]
                except Exception:
                    continue

                subwords = [word for word in vocab_words if "##" in word]
                true_words = [word for word in vocab_words if word not in subwords]
                decoded_words = self.tools.safe_decode(tokenizer, vocab_indices).split()
                shared = set(vocab_words).intersection(decoded_words)
                unshared = set(vocab_words).symmetric_difference(decoded_words)
                total_tokens = len(true_words) + len(subwords)
                subwords_perc = (
                    (len(subwords) / total_tokens * 100.0) if total_tokens else 0.0
                )
                words_perc = (
                    (len(true_words) / total_tokens * 100.0) if total_tokens else 0.0
                )

                vocabulary_stats.append(
                    {
                        "tokenizer": name,
                        "vocabulary_size": len(vocab_words),
                        "decoded_tokens": len(decoded_words),
                        "number_shared_tokens": len(shared),
                        "number_unshared_tokens": len(unshared),
                        "percentage_subwords": subwords_perc,
                        "percentage_true_words": words_perc,
                    }
                )

                decoded_per_id = [
                    self.tools.safe_decode(tokenizer, [idx]) for idx in vocab_indices
                ]
                vocabulary = pd.DataFrame(
                    {
                        "tokenizer": [name] * len(vocab_words),
                        "token_id": vocab_indices,
                        "vocabulary_tokens": vocab_words,
                        "decoded_tokens": decoded_per_id,
                    }
                )
                vocabularies.append(vocabulary)

            except Exception:
                logger.warning("Could not process tokenizer %s for vocab stats", name)
                continue

        vocabulary_stats_df = pd.DataFrame(vocabulary_stats)
        return vocabularies, vocabulary_stats_df

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
    def run_benchmarks(
        self,
        dataset_name: str,
        tokenizer_ids: list[str],
        custom_tokenizers: dict[str, Any] | None = None,
        run_name: str | None = None,
        selected_metric_keys: list[str] | None = None,
        progress_callback: Callable[[float], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        logger.info("Starting benchmark run for dataset: %s", dataset_name)
        logger.info("Tokenizers to benchmark: %s", tokenizer_ids)
        resolved_metric_keys = self.resolve_selected_metric_keys(selected_metric_keys)
        normalized_run_name = run_name.strip() if isinstance(run_name, str) else ""

        # Check dataset exists
        doc_count = self.get_dataset_document_count(dataset_name)
        if doc_count == 0:
            raise ValueError(f"Dataset '{dataset_name}' not found or empty")
        if progress_callback:
            progress_callback(5.0)

        # Load tokenizers from HuggingFace
        tokenizers = self.load_tokenizers(tokenizer_ids)

        # Merge in custom tokenizers if provided
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

        # Calculate vocabulary statistics
        vocabularies, vocabulary_stats = self.calculate_vocabulary_statistics(
            tokenizers
        )
        if progress_callback:
            progress_callback(15.0)

        # Collect texts into memory for processing (respecting max_documents)
        logger.info("Loading texts from database...")
        dataset_rows: list[tuple[int, str]] = []
        for row_id, text in self.stream_dataset_rows_from_database(dataset_name):
            if should_stop and should_stop():
                return {}
            dataset_rows.append((row_id, text))
            if len(dataset_rows) % self.log_interval == 0:
                logger.info("Loaded %d texts...", len(dataset_rows))

        text_ids = [row[0] for row in dataset_rows]
        texts = [row[1] for row in dataset_rows]

        num_docs = len(texts)
        logger.info("Loaded %d documents for benchmarking", num_docs)
        if progress_callback:
            progress_callback(20.0)

        local_stats_frames: list[pd.DataFrame] = []
        global_metrics_rows: list[dict[str, Any]] = []
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

            data = pd.DataFrame(
                {
                    "tokenizer": [name] * num_docs,
                    "name": [dataset_name] * num_docs,
                    "text_id": text_ids,
                    "text": texts,
                }
            )

            data["num_characters"] = pd.Series(texts).str.len()
            data["words_split"] = pd.Series(texts).str.split()
            data["words_count"] = data["words_split"].apply(len)

            t0 = time.perf_counter()
            try:
                tokenize_method = getattr(tokenizer, "tokenize", None)
                uses_tokenize = callable(tokenize_method)
                # Custom tokenizers from file uploads use process_tokens path
                if "CUSTOM" in name:
                    uses_tokenize = False
                    tokenize_method = None

                decoded_tokens = []
                split_tokens = []
                progress_interval = max(1, num_docs // 20)
                processed_docs = 0
                tokenizer_progress_base = progress_base + (index * per_tokenizer_span)
                for text_value in texts:
                    if should_stop and should_stop():
                        return {}
                    decoded, tokens_list = self.tokenize_document(
                        tokenizer, text_value, uses_tokenize, tokenize_method
                    )
                    decoded_tokens.append(decoded)
                    split_tokens.append(tokens_list)
                    processed_docs += 1
                    if progress_callback and processed_docs % progress_interval == 0:
                        progress_value = (
                            tokenizer_progress_base
                            + (processed_docs / max(num_docs, 1)) * per_tokenizer_span
                        )
                        progress_callback(progress_value)

                data["tokens"] = decoded_tokens
                data["tokens_split"] = split_tokens
            except Exception:
                logger.warning("Failed to tokenize documents with %s", name)
                continue

            t1 = time.perf_counter()

            data["tokens_count"] = [
                len(toks) if isinstance(toks, (list, tuple)) else 0
                for toks in data["tokens_split"]
            ]
            data["tokens_to_words_ratio"] = np.where(
                data["words_count"] > 0, data["tokens_count"] / data["words_count"], 0
            )
            data["bytes_per_token"] = np.where(
                data["tokens_count"] > 0,
                data["num_characters"] / data["tokens_count"],
                0,
            )

            # Compute additional metrics
            token_frequency: dict[str, int] = {}
            boundary_preservation: list[float] = []
            round_trip_token_fidelity: list[float] = []
            round_trip_text_fidelity: list[float] = []
            determinism_flags: list[float] = []
            bytes_per_character: list[float] = []
            total_bytes = 0

            for text_value, decoded_text, tokens_list in zip(
                texts, decoded_tokens, split_tokens
            ):
                if should_stop and should_stop():
                    return {}
                total_bytes += len(str(text_value).encode("utf-8"))

                boundary_preservation.append(
                    self.tools.boundary_preservation_score(text_value, decoded_text)
                )

                determinism_tokens = self.tokenize_document(
                    tokenizer, text_value, uses_tokenize, tokenize_method
                )[1]
                determinism_flags.append(float(determinism_tokens == tokens_list))

                rt_decoded, rt_tokens = self.tools.process_tokens(
                    decoded_text, tokenizer
                )
                round_trip_token_fidelity.append(float(rt_tokens == tokens_list))
                round_trip_text_fidelity.append(float(rt_decoded == text_value))

                text_len = len(text_value)
                if text_len > 0:
                    bytes_per_character.append(
                        float(len(text_value.encode("utf-8")) / text_len)
                    )
                else:
                    bytes_per_character.append(0.0)

                for tok in tokens_list:
                    token_frequency[tok] = token_frequency.get(tok, 0) + 1

            data["boundary_preservation_rate"] = boundary_preservation
            data["round_trip_token_fidelity"] = round_trip_token_fidelity
            data["round_trip_text_fidelity"] = round_trip_text_fidelity
            data["determinism_stability"] = determinism_flags
            data["bytes_per_character"] = bytes_per_character

            elapsed = max(t1 - t0, 1e-9)
            total_tokens = int(data["tokens_count"].sum())
            total_chars = int(data["num_characters"].sum())
            tokenization_speed_tps = total_tokens / elapsed
            throughput_chars_per_sec = total_chars / elapsed

            vocab_method = getattr(tokenizer, "get_vocab", None)
            vocab_result: Mapping[Any, Any] | Sequence[Any] | None = None
            if callable(vocab_method):
                try:
                    candidate = vocab_method()
                    if isinstance(candidate, Mapping):
                        vocab_result = candidate
                    elif isinstance(candidate, Sequence) and not isinstance(
                        candidate, (str, bytes)
                    ):
                        vocab_result = candidate
                except Exception:
                    pass

            if isinstance(vocab_result, Mapping):
                vocabulary_size = int(len(vocab_result))
            elif isinstance(vocab_result, Sequence):
                vocabulary_size = int(len(vocab_result))
            else:
                vocabulary_size = 0

            seq_lengths = data["tokens_count"].to_numpy(dtype=float)
            avg_sequence_length = (
                float(np.mean(seq_lengths)) if len(seq_lengths) else 0.0
            )
            median_sequence_length = (
                float(np.median(seq_lengths)) if len(seq_lengths) else 0.0
            )

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
                str(t).replace("##", "").lstrip("?").lstrip("G") for t in vocab_tokens
            }
            oov_words = {w for w in unique_words if w not in vocab_tokens}
            oov_rate = (
                (len(oov_words) / len(unique_words) * 100.0) if unique_words else 0.0
            )

            recovery_count = 0
            sample_words = list(unique_words)
            max_eval = min(5000, len(sample_words))
            sample_words = sample_words[:max_eval]
            for w in sample_words:
                try:
                    if hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
                        enc = tokenizer.encode(w)
                        if hasattr(enc, "ids"):
                            dec = tokenizer.decode(enc.ids)
                        else:
                            dec = tokenizer.decode(enc)
                        if dec == w:
                            recovery_count += 1
                except Exception:
                    continue
            word_recovery_rate = (recovery_count / max(1, len(sample_words))) * 100.0

            dataset_chars = set("".join(texts))
            vocab_chars = set()
            for token_item in normalized_vocab_tokens:
                for ch in token_item:
                    vocab_chars.add(ch)
            intersection = dataset_chars.intersection(vocab_chars)
            character_coverage = (
                (len(intersection) / len(dataset_chars) * 100.0)
                if dataset_chars
                else 0.0
            )

            determinism_rate = (
                float(np.mean(determinism_flags)) if determinism_flags else 0.0
            )
            boundary_preservation_rate = (
                float(np.mean(boundary_preservation)) if boundary_preservation else 0.0
            )
            round_trip_fidelity_rate = (
                float(np.mean(round_trip_token_fidelity))
                if round_trip_token_fidelity
                else 0.0
            )
            round_trip_text_rate = (
                float(np.mean(round_trip_text_fidelity))
                if round_trip_text_fidelity
                else 0.0
            )
            compression_chars_per_token = (
                total_chars / total_tokens if total_tokens else 0.0
            )
            compression_bytes_per_character = (
                total_bytes / total_chars if total_chars else 0.0
            )
            token_entropy = self.tools.token_entropy(token_frequency)
            rare_token_once = int(
                sum(1 for count in token_frequency.values() if count == 1)
            )
            rare_token_twice = int(
                sum(1 for count in token_frequency.values() if count == 2)
            )
            token_unigram_coverage = (
                len(token_frequency) / vocabulary_size if vocabulary_size else 0.0
            )
            segmentation_consistency = self.calculate_morphological_consistency(
                tokenizer, unique_words
            )
            token_id_monotonicity = self.calculate_token_id_monotonicity(vocab_result)

            global_metrics_rows.append(
                {
                    "tokenizer": name,
                    "dataset_name": dataset_name,
                    "tokenization_speed_tps": float(tokenization_speed_tps),
                    "throughput_chars_per_sec": float(throughput_chars_per_sec),
                    "processing_time_seconds": float(elapsed),
                    "model_size_mb": 0.0,  # Not computed for webapp
                    "vocabulary_size": int(vocabulary_size),
                    "avg_sequence_length": float(avg_sequence_length),
                    "median_sequence_length": float(median_sequence_length),
                    "subword_fertility": float(subword_fertility),
                    "oov_rate": float(oov_rate),
                    "word_recovery_rate": float(word_recovery_rate),
                    "character_coverage": float(character_coverage),
                    "segmentation_consistency": float(segmentation_consistency),
                    "determinism_rate": float(determinism_rate),
                    "token_distribution_entropy": float(token_entropy),
                    "rare_token_tail_1": int(rare_token_once),
                    "rare_token_tail_2": int(rare_token_twice),
                    "boundary_preservation_rate": float(boundary_preservation_rate),
                    "compression_chars_per_token": float(compression_chars_per_token),
                    "compression_bytes_per_character": float(
                        compression_bytes_per_character
                    ),
                    "round_trip_fidelity_rate": float(round_trip_fidelity_rate),
                    "round_trip_text_fidelity_rate": float(round_trip_text_rate),
                    "token_id_ordering_monotonicity": float(token_id_monotonicity),
                    "token_unigram_coverage": float(token_unigram_coverage),
                }
            )

            local_stats_frames.append(
                data[
                    [
                        "tokenizer",
                        "name",
                        "text_id",
                        "tokens_count",
                        "tokens_to_words_ratio",
                        "bytes_per_token",
                        "boundary_preservation_rate",
                        "round_trip_token_fidelity",
                        "round_trip_text_fidelity",
                        "determinism_stability",
                        "bytes_per_character",
                    ]
                ].copy()
            )

            logger.info("Completed processing tokenizer: %s", name)
            if progress_callback:
                progress_callback(tokenizer_progress_base + per_tokenizer_span)

        local_stats = (
            pd.concat(local_stats_frames, ignore_index=True)
            if local_stats_frames
            else pd.DataFrame()
        )

        global_metrics = pd.DataFrame(global_metrics_rows)
        per_document_stats = self.build_per_document_stats(local_stats)

        # Persist results to database
        logger.info("Persisting benchmark results to database...")
        self.persist_results(
            vocabularies=vocabularies,
            vocabulary_stats=vocabulary_stats,
            local_stats=local_stats,
            global_metrics=global_metrics,
        )

        # Generate chart data for frontend visualization
        logger.info("Generating chart data...")
        chart_data = self.generate_chart_data(
            vocabularies=vocabularies,
            vocabulary_stats_df=vocabulary_stats,
            global_metrics_df=global_metrics,
            tokenizers=tokenizers,
        )

        return {
            "status": "success",
            "run_name": normalized_run_name or None,
            "selected_metric_keys": resolved_metric_keys,
            "dataset_name": dataset_name,
            "documents_processed": num_docs,
            "tokenizers_processed": list(tokenizers.keys()),
            "tokenizers_count": len(tokenizers),
            "global_metrics": global_metrics.to_dict(orient="records"),
            "per_document_stats": per_document_stats,
            **chart_data,
        }

    # -------------------------------------------------------------------------
    def generate_chart_data(
        self,
        vocabularies: list[pd.DataFrame],
        vocabulary_stats_df: pd.DataFrame,
        global_metrics_df: pd.DataFrame,
        tokenizers: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate structured chart data for frontend visualization."""

        # Vocabulary stats for bar chart
        vocabulary_stats = []
        if not vocabulary_stats_df.empty:
            for _, row in vocabulary_stats_df.iterrows():
                vocab_size = int(row.get("vocabulary_size", 0))
                subwords_pct = float(row.get("percentage_subwords", 0.0))
                words_pct = float(row.get("percentage_true_words", 0.0))
                subwords_count = int(vocab_size * subwords_pct / 100.0)
                true_words_count = int(vocab_size * words_pct / 100.0)
                vocabulary_stats.append(
                    {
                        "tokenizer": str(row.get("tokenizer", "")),
                        "vocabulary_size": vocab_size,
                        "subwords_count": subwords_count,
                        "true_words_count": true_words_count,
                        "subwords_percentage": subwords_pct,
                    }
                )

        # Speed metrics from global metrics
        speed_metrics = []
        if not global_metrics_df.empty:
            for _, row in global_metrics_df.iterrows():
                speed_metrics.append(
                    {
                        "tokenizer": str(row.get("tokenizer", "")),
                        "tokens_per_second": float(
                            row.get("tokenization_speed_tps", 0.0)
                        ),
                        "chars_per_second": float(
                            row.get("throughput_chars_per_sec", 0.0)
                        ),
                        "processing_time_seconds": float(
                            row.get("processing_time_seconds", 0.0)
                        ),
                    }
                )

        # Token length distributions (histogram bins)
        token_length_distributions = []
        # Build from vocabularies - count token lengths
        for vocab_df in vocabularies:
            if vocab_df.empty:
                continue
            tokenizer_name = vocab_df["tokenizer"].iloc[0]
            tokens = vocab_df["vocabulary_tokens"].astype(str)
            lengths = tokens.str.len().dropna().astype(int)

            if len(lengths) == 0:
                continue

            # Create histogram bins
            max_len = min(int(lengths.max()), 30)  # Cap at 30 chars
            bins_edges = list(range(0, max_len + 2, 2))
            if len(bins_edges) < 2:
                bins_edges = [0, max_len + 1]

            counts, edges = np.histogram(lengths, bins=bins_edges)
            bins = []
            for i in range(len(counts)):
                bins.append(
                    {
                        "bin_start": int(edges[i]),
                        "bin_end": int(edges[i + 1]),
                        "count": int(counts[i]),
                    }
                )

            token_length_distributions.append(
                {
                    "tokenizer": tokenizer_name,
                    "bins": bins,
                    "mean": float(lengths.mean()) if len(lengths) > 0 else 0.0,
                    "std": float(lengths.std()) if len(lengths) > 1 else 0.0,
                }
            )

        return {
            "vocabulary_stats": vocabulary_stats,
            "speed_metrics": speed_metrics,
            "token_length_distributions": token_length_distributions,
        }

    # -------------------------------------------------------------------------
    def persist_results(
        self,
        vocabularies: list[pd.DataFrame],
        vocabulary_stats: pd.DataFrame,
        local_stats: pd.DataFrame,
        global_metrics: pd.DataFrame,
    ) -> None:
        tokenizer_names: set[str] = set()
        dataset_names: list[str] = []

        if not local_stats.empty:
            tokenizer_names.update(
                local_stats["tokenizer"].dropna().astype(str).tolist()
            )
            dataset_names.extend(local_stats["name"].dropna().astype(str).tolist())
        if not global_metrics.empty:
            tokenizer_names.update(
                global_metrics["tokenizer"].dropna().astype(str).tolist()
            )
            dataset_names.extend(
                global_metrics["dataset_name"].dropna().astype(str).tolist()
            )
        if not vocabulary_stats.empty:
            tokenizer_names.update(
                vocabulary_stats["tokenizer"].dropna().astype(str).tolist()
            )
        for vocab_df in vocabularies:
            if vocab_df.empty or "tokenizer" not in vocab_df.columns:
                continue
            tokenizer_names.update(vocab_df["tokenizer"].dropna().astype(str).tolist())

        tokenizer_ids = self.ensure_tokenizer_ids(sorted(tokenizer_names))
        dataset_id: int | None = None
        if dataset_names:
            dataset_id = self.get_dataset_id(dataset_names[0])
        if dataset_id is None and (not local_stats.empty or not global_metrics.empty):
            raise ValueError("Dataset id not found while persisting benchmark results.")
        dataset_id_value = int(dataset_id) if dataset_id is not None else None

        # Persist local tokenizer statistics with (tokenizer_id, document_id) semantics.
        if not local_stats.empty:
            local_columns = [
                "tokenizer",
                "text_id",
                "tokens_count",
                "tokens_to_words_ratio",
                "bytes_per_token",
                "boundary_preservation_rate",
                "round_trip_token_fidelity",
                "round_trip_text_fidelity",
                "determinism_stability",
                "bytes_per_character",
            ]
            local_storage = local_stats[
                [col for col in local_columns if col in local_stats.columns]
            ].copy()
            if not local_storage.empty:
                local_storage["tokenizer_id"] = local_storage["tokenizer"].map(
                    tokenizer_ids
                )
                local_storage["document_id"] = pd.to_numeric(
                    local_storage["text_id"], errors="coerce"
                )
                local_storage = local_storage.dropna(
                    subset=["tokenizer_id", "document_id"]
                )
                local_storage["tokenizer_id"] = local_storage["tokenizer_id"].astype(
                    int
                )
                local_storage["document_id"] = local_storage["document_id"].astype(int)
                local_storage = local_storage[
                    [
                        "tokenizer_id",
                        "document_id",
                        "tokens_count",
                        "tokens_to_words_ratio",
                        "bytes_per_token",
                        "boundary_preservation_rate",
                        "round_trip_token_fidelity",
                        "round_trip_text_fidelity",
                        "determinism_stability",
                        "bytes_per_character",
                    ]
                ]
                database.upsert_into_database(
                    local_storage, TokenizationDocumentStats.__tablename__
                )
                logger.info("Saved %d local stats records", len(local_storage))

        # Persist global metrics (core + detail split)
        if not global_metrics.empty:
            global_storage = global_metrics.copy()
            global_storage["tokenizer_id"] = global_storage["tokenizer"].map(
                tokenizer_ids
            )
            global_storage["dataset_id"] = dataset_id_value
            global_storage = global_storage.dropna(
                subset=["tokenizer_id", "dataset_id"]
            )
            global_storage["tokenizer_id"] = global_storage["tokenizer_id"].astype(int)
            global_storage["dataset_id"] = global_storage["dataset_id"].astype(int)

            core_columns = [
                "tokenizer_id",
                "dataset_id",
                "tokenization_speed_tps",
                "throughput_chars_per_sec",
                "model_size_mb",
                "vocabulary_size",
                "subword_fertility",
                "oov_rate",
                "word_recovery_rate",
            ]
            detail_columns = [
                "character_coverage",
                "segmentation_consistency",
                "determinism_rate",
                "token_distribution_entropy",
                "rare_token_tail_1",
                "rare_token_tail_2",
                "boundary_preservation_rate",
                "compression_chars_per_token",
                "compression_bytes_per_character",
                "round_trip_fidelity_rate",
                "round_trip_text_fidelity_rate",
                "token_id_ordering_monotonicity",
                "token_unigram_coverage",
            ]
            core_storage = global_storage[
                [col for col in core_columns if col in global_storage.columns]
            ].copy()
            if not core_storage.empty:
                database.upsert_into_database(
                    core_storage, TokenizationDatasetStats.__tablename__
                )
                logger.info("Saved %d global metrics records", len(core_storage))

            detail_storage = global_storage[
                [col for col in detail_columns if col in global_storage.columns]
            ].copy()
            if not detail_storage.empty:
                if dataset_id_value is None:
                    raise ValueError(
                        "Dataset id not found while persisting benchmark detail results."
                    )
                stmt = select(
                    TokenizationDatasetStats.id,
                    TokenizationDatasetStats.tokenizer_id,
                ).where(TokenizationDatasetStats.dataset_id == dataset_id_value)
                with self._session() as session:
                    rows = session.execute(stmt).all()
                tokenizer_to_global: dict[int, int] = {}
                for global_stats_id, tokenizer_id in rows:
                    tokenizer_to_global[int(tokenizer_id)] = int(global_stats_id)

                detail_storage["tokenizer_id"] = global_storage["tokenizer_id"]
                detail_storage["global_stats_id"] = detail_storage["tokenizer_id"].map(
                    tokenizer_to_global
                )
                detail_storage = detail_storage.dropna(subset=["global_stats_id"])
                detail_storage["global_stats_id"] = detail_storage[
                    "global_stats_id"
                ].astype(int)
                detail_storage = detail_storage.drop(columns=["tokenizer_id"])
                database.upsert_into_database(
                    detail_storage, TokenizationDatasetStatsDetail.__tablename__
                )

        # Persist vocabulary statistics
        if not vocabulary_stats.empty:
            vocab_stats_columns = [
                "tokenizer",
                "vocabulary_size",
                "decoded_tokens",
                "number_shared_tokens",
                "number_unshared_tokens",
                "percentage_subwords",
                "percentage_true_words",
            ]
            vocab_stats_storage = vocabulary_stats[
                [col for col in vocab_stats_columns if col in vocabulary_stats.columns]
            ].copy()
            if not vocab_stats_storage.empty:
                vocab_stats_storage["tokenizer_id"] = vocab_stats_storage[
                    "tokenizer"
                ].map(tokenizer_ids)
                vocab_stats_storage = vocab_stats_storage.dropna(
                    subset=["tokenizer_id"]
                )
                vocab_stats_storage["tokenizer_id"] = vocab_stats_storage[
                    "tokenizer_id"
                ].astype(int)
                vocab_stats_storage = vocab_stats_storage[
                    [
                        "tokenizer_id",
                        "vocabulary_size",
                        "decoded_tokens",
                        "number_shared_tokens",
                        "number_unshared_tokens",
                        "percentage_subwords",
                        "percentage_true_words",
                    ]
                ]
                database.upsert_into_database(
                    vocab_stats_storage, TokenizerVocabularyStatistics.__tablename__
                )
                logger.info(
                    "Saved %d vocabulary stats records", len(vocab_stats_storage)
                )

        # Persist vocabularies
        for vocab_df in vocabularies:
            if vocab_df.empty:
                continue
            vocab_columns = [
                "tokenizer",
                "token_id",
                "vocabulary_tokens",
                "decoded_tokens",
            ]
            vocab_storage = vocab_df[
                [col for col in vocab_columns if col in vocab_df.columns]
            ].copy()
            if vocab_storage.empty:
                continue
            vocab_storage["tokenizer_id"] = vocab_storage["tokenizer"].map(
                tokenizer_ids
            )
            vocab_storage = vocab_storage.dropna(subset=["tokenizer_id"])
            vocab_storage["tokenizer_id"] = vocab_storage["tokenizer_id"].astype(int)
            vocab_storage = vocab_storage[
                ["tokenizer_id", "token_id", "vocabulary_tokens", "decoded_tokens"]
            ]
            database.upsert_into_database(
                vocab_storage, TokenizerVocabulary.__tablename__
            )
            logger.info(
                "Saved %d vocabulary rows for tokenizer: %s",
                len(vocab_storage),
                vocab_df["tokenizer"].iloc[0],
            )

    # -------------------------------------------------------------------------
    def get_dataset_id(self, dataset_name: str) -> int | None:
        stmt = select(Dataset.id).where(Dataset.name == dataset_name).limit(1)
        with self._session() as session:
            dataset_id = session.execute(stmt).scalar_one_or_none()
        return int(dataset_id) if dataset_id is not None else None

    # -------------------------------------------------------------------------
    def ensure_tokenizer_ids(self, tokenizer_names: list[str]) -> dict[str, int]:
        if not tokenizer_names:
            return {}
        deduped_names = list(dict.fromkeys(tokenizer_names))
        with self._session() as session:
            existing_rows = session.execute(
                select(Tokenizer).where(Tokenizer.name.in_(deduped_names))
            ).scalars().all()
            existing_names = {row.name for row in existing_rows}
            for name in deduped_names:
                if name in existing_names:
                    continue
                session.add(Tokenizer(name=name))
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
            mapping_rows = session.execute(
                select(Tokenizer.id, Tokenizer.name).where(
                    Tokenizer.name.in_(deduped_names)
                )
            ).all()
        return {str(name): int(tokenizer_id) for tokenizer_id, name in mapping_rows}

    # -------------------------------------------------------------------------
