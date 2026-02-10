from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy
from matplotlib.figure import Figure
from transformers import AutoTokenizer
from transformers.utils.logging import set_verbosity_error

from TKBEN.server.repositories.database.backend import database
from TKBEN.server.repositories.schemas.models import (
    TokenizationGlobalMetrics,
    TokenizationLocalStats,
    Vocabulary,
    VocabularyStatistics,
)
from TKBEN.server.configurations import server_settings
from TKBEN.server.utils.logger import logger


###############################################################################
class BenchmarkTools:

    def __call__(self) -> None:
        pass

    # -------------------------------------------------------------------------
    def process_tokens(self, text: str, tokenizer: Any) -> tuple[str, list[str]]:
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return "", []

        tokenizer_name = getattr(tokenizer, "name_or_path", type(tokenizer).__name__)

        try:
            encoded = tokenizer.encode(text)
        except Exception:
            logger.debug(
                "Tokenizer %s raised an exception while encoding text",
                tokenizer_name,
                exc_info=True,
            )
            return "", []

        tokens_from_encoding: list[str] = []
        if hasattr(encoded, "tokens"):
            tokens_attr = getattr(encoded, "tokens")
            if callable(tokens_attr):
                try:
                    raw_tokens = tokens_attr()
                except Exception:
                    logger.debug(
                        "Tokenizer %s failed to extract tokens from encoding",
                        tokenizer_name,
                        exc_info=True,
                    )
                else:
                    tokens_from_encoding = self.normalize_token_output(raw_tokens)
            elif isinstance(tokens_attr, (list, tuple)):
                tokens_from_encoding = [str(tok) for tok in tokens_attr]
            elif isinstance(tokens_attr, Iterable) and not isinstance(
                tokens_attr, (str, bytes)
            ):
                tokens_from_encoding = [str(tok) for tok in tokens_attr]

        token_ids = self.extract_token_ids(encoded)
        if token_ids:
            decoded = self.safe_decode(tokenizer, token_ids)
        else:
            decoded = " ".join(tokens_from_encoding)

        if not tokens_from_encoding:
            tokens_from_encoding = self.convert_ids_to_tokens(
                tokenizer, token_ids, decoded
            )

        return decoded, tokens_from_encoding

    # -------------------------------------------------------------------------
    def extract_token_ids(self, encoded: Any) -> list[int]:
        ids_source: Any | None = None
        if hasattr(encoded, "ids"):
            ids_source = getattr(encoded, "ids")
        elif isinstance(encoded, np.ndarray):
            ids_source = encoded.tolist()
        elif isinstance(encoded, (list, tuple)):
            ids_source = encoded

        if ids_source is None:
            return []

        try:
            return [int(i) for i in ids_source]
        except Exception:
            logger.debug("Failed to coerce token ids from encoding", exc_info=True)
            return []

    # -------------------------------------------------------------------------
    def safe_decode(self, tokenizer: Any, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        try:
            decoded = tokenizer.decode(token_ids)
        except Exception:
            logger.debug(
                "Tokenizer %s raised an exception while decoding ids",
                getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
                exc_info=True,
            )
            return ""

        if isinstance(decoded, (list, tuple)):
            return " ".join(str(tok) for tok in decoded)
        return str(decoded)

    # -------------------------------------------------------------------------
    def convert_ids_to_tokens(
        self, tokenizer: Any, token_ids: list[int], fallback_text: str
    ) -> list[str]:
        try:
            converter = getattr(tokenizer, "convert_ids_to_tokens", None)
            if callable(converter):
                tokens = converter(token_ids)
                if isinstance(tokens, np.ndarray):
                    tokens = tokens.tolist()
                if isinstance(tokens, (list, tuple)):
                    return [str(tok) for tok in tokens]
            id_to_token = getattr(tokenizer, "id_to_token", None)
            if callable(id_to_token):
                return [str(id_to_token(idx)) for idx in token_ids]
        except Exception:
            logger.debug(
                "Tokenizer %s failed to convert ids to tokens",
                getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
                exc_info=True,
            )

        if fallback_text:
            return fallback_text.split()
        return []

    # -------------------------------------------------------------------------
    def normalize_token_output(self, tokens: Any) -> list[str]:
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        if hasattr(tokens, "tokens") and callable(getattr(tokens, "tokens")):
            try:
                tokens = tokens.tokens()
            except Exception:
                logger.debug("Failed to normalize token output", exc_info=True)
                return []

        if isinstance(tokens, (list, tuple)):
            return [str(tok) for tok in tokens]

        if isinstance(tokens, str):
            return tokens.split()

        if isinstance(tokens, dict):
            for key in ("tokens", "input_tokens"):
                value = tokens.get(key)
                if isinstance(value, (list, tuple)):
                    return [str(tok) for tok in value]

        if isinstance(tokens, Iterable):
            return [str(tok) for tok in tokens]

        return []

    # -------------------------------------------------------------------------
    def is_tokenizer_compatible(self, tokenizer: Any) -> bool:
        if tokenizer is None or isinstance(tokenizer, bool):
            return False

        if callable(getattr(tokenizer, "tokenize", None)):
            return True

        encode_method = getattr(tokenizer, "encode", None)
        decode_method = getattr(tokenizer, "decode", None)
        if callable(encode_method) and callable(decode_method):
            return True

        call_method = getattr(tokenizer, "__call__", None)
        return callable(call_method)

    # -------------------------------------------------------------------------
    def boundary_preservation_score(
        self, original_text: str, decoded_text: str
    ) -> float:
        if not original_text:
            return 0.0

        safe_original = str(original_text)
        safe_decoded = str(decoded_text)
        limit = min(len(safe_original), len(safe_decoded))
        if limit == 0:
            return 0.0

        boundary_matches = 0
        total_boundaries = 0
        for idx in range(limit):
            char_original = safe_original[idx]
            if char_original.isspace() or re.match(r"[^\w\s]", char_original):
                total_boundaries += 1
                if char_original == safe_decoded[idx]:
                    boundary_matches += 1

        if total_boundaries == 0:
            return 0.0

        return boundary_matches / total_boundaries

    # -------------------------------------------------------------------------
    def jaccard_similarity(
        self, first: Sequence[str], second: Sequence[str]
    ) -> float:
        if not first and not second:
            return 1.0
        if not first or not second:
            return 0.0

        first_set = set(first)
        second_set = set(second)
        union_size = len(first_set.union(second_set))
        if union_size == 0:
            return 0.0

        return len(first_set.intersection(second_set)) / union_size

    # -------------------------------------------------------------------------
    def token_entropy(self, counts: Mapping[str, int]) -> float:
        if not counts:
            return 0.0
        values = np.fromiter(counts.values(), dtype=float)
        total = values.sum()
        if total <= 0:
            return 0.0
        probs = values / total
        probs = probs[probs > 0]
        if probs.size == 0:
            return 0.0

        return float(-np.sum(probs * np.log2(probs)))


###############################################################################
class BenchmarkService:

    def __init__(
        self,
        max_documents: int = 0,
    ) -> None:
        set_verbosity_error()
        self.max_documents = max_documents
        self.reduce_data_size = True  # Always true for webapp
        self.tools = BenchmarkTools()
        
        # Load settings from config
        self.streaming_batch_size = server_settings.benchmarks.streaming_batch_size
        self.log_interval = server_settings.benchmarks.log_interval

    # -------------------------------------------------------------------------
    def load_tokenizers(self, tokenizer_ids: list[str]) -> dict[str, Any]:
        tokenizers: dict[str, Any] = {}
        for tokenizer_id in tokenizer_ids:
            try:
                logger.info("Loading tokenizer: %s", tokenizer_id)
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
                if self.tools.is_tokenizer_compatible(tokenizer):
                    tokenizers[tokenizer_id] = tokenizer
                else:
                    logger.warning(
                        "Tokenizer %s not compatible, skipping", tokenizer_id
                    )
            except Exception:
                logger.warning("Failed to load tokenizer %s", tokenizer_id)
                logger.debug("Tokenizer load error", exc_info=True)
        return tokenizers

    # -------------------------------------------------------------------------
    def stream_texts_from_database(
        self, dataset_name: str
    ) -> Generator[str, None, None]:
        query = sqlalchemy.text(
            'SELECT "text" FROM "TEXT_DATASET" WHERE "dataset_name" = :dataset'
        )
        count = 0
        with database.backend.engine.connect().execution_options(
            stream_results=True
        ) as conn:
            result = conn.execute(query, {"dataset": dataset_name})
            while True:
                rows = result.fetchmany(self.streaming_batch_size)
                if not rows:
                    break
                for row in rows:
                    if hasattr(row, "_mapping"):
                        text = row._mapping.get("text", "")
                    else:
                        text = row[0] if row else ""
                    if text and isinstance(text, str):
                        yield text
                        count += 1
                        if self.max_documents > 0 and count >= self.max_documents:
                            return

    # -------------------------------------------------------------------------
    def get_dataset_document_count(self, dataset_name: str) -> int:
        query = sqlalchemy.text(
            'SELECT COUNT(*) FROM "TEXT_DATASET" WHERE "dataset_name" = :dataset'
        )
        with database.backend.engine.connect() as conn:
            result = conn.execute(query, {"dataset": dataset_name})
            row = result.first()
            return row[0] if row else 0

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
    def calculate_vocabulary_statistics(
        self, tokenizers: dict[str, Any]
    ) -> tuple[list[Any], pd.DataFrame]:
        vocabulary_stats: list[dict[str, Any]] = []
        vocabularies: list[pd.DataFrame] = []
        
        for name, tokenizer in tokenizers.items():
            if not self.tools.is_tokenizer_compatible(tokenizer):
                logger.warning(
                    "Skipping tokenizer %s for vocab stats", name
                )
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
                        "number_tokens_from_vocabulary": len(vocab_words),
                        "number_tokens_from_decode": len(decoded_words),
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

        selected_words = [
            w for w in sorted(base_words) if re.match(r"^[A-Za-z]+$", w)
        ][:200]
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
        progress_callback: Callable[[float], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        logger.info("Starting benchmark run for dataset: %s", dataset_name)
        logger.info("Tokenizers to benchmark: %s", tokenizer_ids)

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
        vocabularies, vocabulary_stats = self.calculate_vocabulary_statistics(tokenizers)
        if progress_callback:
            progress_callback(15.0)

        # Collect texts into memory for processing (respecting max_documents)
        logger.info("Loading texts from database...")
        texts: list[str] = []
        for text in self.stream_texts_from_database(dataset_name):
            if should_stop and should_stop():
                return {}
            texts.append(text)
            if len(texts) % self.log_interval == 0:
                logger.info("Loaded %d texts...", len(texts))

        num_docs = len(texts)
        logger.info("Loaded %d documents for benchmarking", num_docs)
        if progress_callback:
            progress_callback(20.0)

        all_tokenizers: list[pd.DataFrame] = []
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
                    "text": texts,
                }
            )

            data["num_characters"] = pd.Series(texts).str.len()
            data["words_split"] = pd.Series(texts).str.split()
            data["words_count"] = data["words_split"].apply(len)
            data["AVG_words_length"] = data["words_split"].apply(
                lambda ws: np.mean([len(w) for w in ws]) if ws else 0
            )

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
                        progress_value = tokenizer_progress_base + (
                            processed_docs / max(num_docs, 1)
                        ) * per_tokenizer_span
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
            data["tokens_characters"] = data["tokens"].str.len()
            data["AVG_tokens_length"] = data["tokens_split"].apply(
                lambda toks: np.mean([len(tok) for tok in toks]) if toks else 0
            )
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
            characters_per_token_values: list[float] = []
            token_length_variances: list[float] = []
            token_length_stds: list[float] = []
            token_lengths: list[int] = []
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

                if tokens_list:
                    lengths_arr = np.fromiter(
                        (len(tok) for tok in tokens_list), dtype=float
                    )
                    token_lengths.extend(int(len(tok)) for tok in tokens_list)
                    token_length_variances.append(float(np.var(lengths_arr)))
                    token_length_stds.append(float(np.std(lengths_arr)))
                    characters_per_token_values.append(
                        float(len(text_value) / len(tokens_list))
                    )
                else:
                    token_length_variances.append(0.0)
                    token_length_stds.append(0.0)
                    characters_per_token_values.append(0.0)

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
            data["characters_per_token"] = characters_per_token_values
            data["token_length_variance"] = token_length_variances
            data["token_length_std"] = token_length_stds

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
                float(np.mean(boundary_preservation))
                if boundary_preservation
                else 0.0
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
            token_length_variance_global = (
                float(np.var(token_lengths)) if token_lengths else 0.0
            )
            token_length_std_global = (
                float(np.std(token_lengths)) if token_lengths else 0.0
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
                    "token_length_variance": float(token_length_variance_global),
                    "token_length_std": float(token_length_std_global),
                }
            )

            # Reduce data size (always enabled)
            data.drop(
                columns=["tokens", "tokens_split", "words_split"], inplace=True
            )
            all_tokenizers.append(data)

            logger.info("Completed processing tokenizer: %s", name)
            if progress_callback:
                progress_callback(tokenizer_progress_base + per_tokenizer_span)

        benchmark_results = (
            pd.concat(all_tokenizers, ignore_index=True)
            if all_tokenizers
            else pd.DataFrame()
        )

        global_metrics = pd.DataFrame(global_metrics_rows)

        # Persist results to database
        logger.info("Persisting benchmark results to database...")
        self.persist_results(
            vocabularies=vocabularies,
            vocabulary_stats=vocabulary_stats,
            benchmark_results=benchmark_results,
            global_metrics=global_metrics,
            dataset_name=dataset_name,
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
            "dataset_name": dataset_name,
            "documents_processed": num_docs,
            "tokenizers_processed": list(tokenizers.keys()),
            "tokenizers_count": len(tokenizers),
            "global_metrics": global_metrics.to_dict(orient="records"),
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
                vocab_size = int(row.get("number_tokens_from_vocabulary", 0))
                subwords_pct = float(row.get("percentage_subwords", 0.0))
                words_pct = float(row.get("percentage_true_words", 0.0))
                subwords_count = int(vocab_size * subwords_pct / 100.0)
                true_words_count = int(vocab_size * words_pct / 100.0)
                vocabulary_stats.append({
                    "tokenizer": str(row.get("tokenizer", "")),
                    "vocabulary_size": vocab_size,
                    "subwords_count": subwords_count,
                    "true_words_count": true_words_count,
                    "subwords_percentage": subwords_pct,
                })

        # Speed metrics from global metrics
        speed_metrics = []
        if not global_metrics_df.empty:
            for _, row in global_metrics_df.iterrows():
                speed_metrics.append({
                    "tokenizer": str(row.get("tokenizer", "")),
                    "tokens_per_second": float(row.get("tokenization_speed_tps", 0.0)),
                    "chars_per_second": float(row.get("throughput_chars_per_sec", 0.0)),
                    "processing_time_seconds": 0.0,  # Not tracked currently
                })

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
                bins.append({
                    "bin_start": int(edges[i]),
                    "bin_end": int(edges[i + 1]),
                    "count": int(counts[i]),
                })
            
            token_length_distributions.append({
                "tokenizer": tokenizer_name,
                "bins": bins,
                "mean": float(lengths.mean()) if len(lengths) > 0 else 0.0,
                "std": float(lengths.std()) if len(lengths) > 1 else 0.0,
            })

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
        benchmark_results: pd.DataFrame,
        global_metrics: pd.DataFrame,
        dataset_name: str,
    ) -> None:
        # Persist global metrics
        if not global_metrics.empty:
            database.delete_by_key(
                TokenizationGlobalMetrics.__tablename__,
                "dataset_name",
                dataset_name,
            )
            database.insert_dataframe(
                global_metrics, TokenizationGlobalMetrics.__tablename__
            )
            logger.info("Saved %d global metrics records", len(global_metrics))

        # Persist vocabulary statistics
        if not vocabulary_stats.empty:
            for tokenizer_name in vocabulary_stats["tokenizer"].unique():
                database.delete_by_key(
                    VocabularyStatistics.__tablename__,
                    "tokenizer",
                    tokenizer_name,
                )
            database.insert_dataframe(
                vocabulary_stats, VocabularyStatistics.__tablename__
            )
            logger.info("Saved %d vocabulary stats records", len(vocabulary_stats))

        # Persist vocabularies
        for vocab_df in vocabularies:
            if vocab_df.empty:
                continue
            tokenizer_name = vocab_df["tokenizer"].iloc[0]
            database.delete_by_key(
                Vocabulary.__tablename__,
                "tokenizer",
                tokenizer_name,
            )
            database.insert_dataframe(vocab_df, Vocabulary.__tablename__)
            logger.info("Saved vocabulary for tokenizer: %s", tokenizer_name)

    # -------------------------------------------------------------------------
    def generate_plots(
        self,
        vocabularies: list[pd.DataFrame],
        vocabulary_stats: pd.DataFrame,
    ) -> list[dict[str, str]]:
        import base64
        import io

        plots: list[dict[str, str]] = []

        # Plot vocabulary size
        if vocabularies:
            combined_vocab = pd.concat(vocabularies, ignore_index=True)
            if not combined_vocab.empty:
                fig = self.plot_vocabulary_size(combined_vocab)
                if fig:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                    buf.seek(0)
                    plots.append({
                        "name": "vocabulary_size",
                        "data": base64.b64encode(buf.read()).decode("utf-8"),
                    })
                    plt.close(fig)

        # Plot subwords vs words
        if vocabularies:
            combined_vocab = pd.concat(vocabularies, ignore_index=True)
            if not combined_vocab.empty:
                fig = self.plot_subwords_vs_words(combined_vocab)
                if fig:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                    buf.seek(0)
                    plots.append({
                        "name": "subwords_vs_words",
                        "data": base64.b64encode(buf.read()).decode("utf-8"),
                    })
                    plt.close(fig)

        return plots

    # -------------------------------------------------------------------------
    def plot_vocabulary_size(self, data: pd.DataFrame) -> Figure | None:
        df = data.dropna(subset=["vocabulary_tokens"])
        df["vocabulary_tokens"] = df["vocabulary_tokens"].astype(str)
        df = df[df["vocabulary_tokens"].str.len() > 0]

        if df.empty:
            return None

        counts = (
            df.groupby("tokenizer", sort=False)["vocabulary_tokens"]
            .nunique()
            .sort_values(ascending=True)
        )

        n_tok = len(counts)
        width_in = 14
        height_in = max(4, 0.7 * n_tok + 2)

        fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=400)

        y_pos = np.arange(n_tok, dtype=float)
        widths = counts.to_numpy(dtype=float)

        ax.barh(y_pos, widths, edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(counts.index))

        ax.set_title(
            "Vocabulary size by tokenizer",
            fontsize=16,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Number of tokens", fontsize=13, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        fig.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    def plot_subwords_vs_words(self, data: pd.DataFrame) -> Figure | None:
        df = data.loc[:, ["tokenizer", "vocabulary_tokens"]].copy()
        df = df.dropna(subset=["tokenizer", "vocabulary_tokens"])
        if df.empty:
            return None

        df["tokenizer"] = df["tokenizer"].astype(str)
        df["vocabulary_tokens"] = df["vocabulary_tokens"].astype(str).str.strip()
        df = df[df["vocabulary_tokens"].str.len() > 0]
        if df.empty:
            return None

        special_pat = r"^(?:\[.*\]|<.*>|\{.*\}|</?s>|</?pad>|UNK|PAD)$"
        is_special = df["vocabulary_tokens"].str.match(special_pat, case=False)
        df = df[~is_special]
        if df.empty:
            return None

        bert_sub = df["vocabulary_tokens"].str.startswith("##")
        sp_has = df["vocabulary_tokens"].str.contains("▁", regex=False)
        sp_word_start = df["vocabulary_tokens"].str.startswith("▁")
        sp_sub = sp_has & (~sp_word_start)
        bbpe_has = df["vocabulary_tokens"].str.contains("Ġ", regex=False)
        bbpe_word_start = df["vocabulary_tokens"].str.startswith("Ġ")
        bbpe_sub = bbpe_has & (~bbpe_word_start)

        is_subword = bert_sub | sp_sub | bbpe_sub
        is_word = ~is_subword

        df["is_subword"] = is_subword.astype(int)
        df["is_word"] = is_word.astype(int)

        grouped = (
            df.groupby("tokenizer", sort=False)
            .agg(subwords_count=("is_subword", "sum"), words_count=("is_word", "sum"))
            .reset_index()
        )

        grouped = grouped.sort_values("subwords_count", ascending=False).reset_index(
            drop=True
        )

        if grouped.empty:
            return None

        tokenizers = grouped["tokenizer"].tolist()
        n = len(tokenizers)
        x = np.arange(n, dtype=float)
        width = 0.4

        subwords = grouped["subwords_count"].to_numpy(dtype=float)
        words = grouped["words_count"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 7), dpi=400)
        ax.bar(
            x - width / 2,
            subwords,
            width=width,
            edgecolor="black",
            color="#105D8D",
            label="Subwords (vocabulary)",
        )
        ax.bar(
            x + width / 2,
            words,
            width=width,
            edgecolor="black",
            color="#107F40",
            label="Words (vocabulary)",
        )

        ax.set_title(
            "Subwords vs Words by tokenizer (vocabulary)",
            fontsize=16,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Tokenizer", fontsize=13, fontweight="bold")
        ax.set_ylabel("Count", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tokenizers, rotation=35, ha="right")

        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(title="", fontsize=11)

        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            try:
                lbl.set_fontweight("bold")
            except Exception:
                pass

        fig.tight_layout()
        return fig
