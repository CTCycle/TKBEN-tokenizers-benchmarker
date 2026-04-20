from __future__ import annotations

import os
import re
import time  # noqa: F401
from collections.abc import Generator, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from transformers.utils.logging import set_verbosity_error

from TKBEN.server.repositories.benchmarks import BenchmarkRepository
from TKBEN.server.repositories.serialization.data import (
    BenchmarkReportSerializer,
    DatasetSerializer,
)
from TKBEN.server.services.metrics.catalog import BENCHMARK_METRIC_CATALOG
from TKBEN.server.configurations import get_server_settings
from TKBEN.server.common.constants import TOKENIZERS_PATH
from TKBEN.server.common.utils.logger import logger
from TKBEN.server.common.utils.security import (
    ensure_path_is_within,
    normalize_identifier,
)
from TKBEN.server.services.benchmark_execution import BenchmarkServiceExecutionMixin
from TKBEN.server.services.benchmark_plotting import BenchmarkPlottingMixin
from TKBEN.server.services.custom_tokenizers import get_custom_tokenizer_registry


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
    def jaccard_similarity(self, first: Sequence[str], second: Sequence[str]) -> float:
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
class BenchmarkService(BenchmarkServiceExecutionMixin, BenchmarkPlottingMixin):
    TOKENIZER_ID_MAX_LENGTH = 160

    def __init__(
        self,
        max_documents: int = 0,
    ) -> None:
        set_verbosity_error()
        self.max_documents = max_documents
        self.reduce_data_size = True  # Always true for webapp
        self.tools = BenchmarkTools()
        self.repository = BenchmarkRepository()
        self.report_serializer = BenchmarkReportSerializer()
        self.dataset_serializer = DatasetSerializer()

        # Load settings from config
        self.streaming_batch_size = get_server_settings().benchmarks.streaming_batch_size
        self.log_interval = get_server_settings().benchmarks.log_interval

    def get_tokenizer_cache_dir(self, tokenizer_id: str) -> str:
        safe_id = normalize_identifier(
            tokenizer_id,
            "Tokenizer identifier",
            max_length=self.TOKENIZER_ID_MAX_LENGTH,
        )
        safe_name = safe_id.replace("/", "__")
        candidate = os.path.join(TOKENIZERS_PATH, safe_name)
        return ensure_path_is_within(TOKENIZERS_PATH, candidate)

    # -------------------------------------------------------------------------
    def get_missing_persisted_tokenizers(self, tokenizer_ids: list[str]) -> list[str]:
        requested: list[str] = []
        invalid: list[str] = []
        for tokenizer_id in tokenizer_ids:
            raw_name = str(tokenizer_id).strip()
            if not raw_name:
                continue
            try:
                requested.append(
                    normalize_identifier(
                        raw_name,
                        "Tokenizer identifier",
                        max_length=self.TOKENIZER_ID_MAX_LENGTH,
                    )
                )
            except ValueError:
                invalid.append(raw_name)
        if invalid:
            preview = ", ".join(invalid[:3])
            if len(invalid) > 3:
                preview = f"{preview}, ..."
            raise ValueError(f"Invalid tokenizer identifier(s): {preview}")
        if not requested:
            return []

        unique_requested = list(dict.fromkeys(requested))
        missing = set(self.repository.get_missing_persisted_tokenizers(unique_requested))

        missing: list[str] = []
        for tokenizer_name in unique_requested:
            if tokenizer_name in missing:
                missing.append(tokenizer_name)
                continue
            cache_dir = self.get_tokenizer_cache_dir(tokenizer_name)
            has_cached_files = False
            if os.path.isdir(cache_dir):
                for _, _, files in os.walk(cache_dir):
                    if files:
                        has_cached_files = True
                        break
            if not has_cached_files:
                missing.append(tokenizer_name)
        return missing

    # -------------------------------------------------------------------------
    def resolve_custom_tokenizer_selection(
        self,
        custom_tokenizer_name: str | None,
    ) -> dict[str, Any]:
        if not isinstance(custom_tokenizer_name, str) or not custom_tokenizer_name.strip():
            return {}
        selected_name = custom_tokenizer_name.strip()
        tokenizer = get_custom_tokenizer_registry().get(selected_name)
        if tokenizer is None or not self.tools.is_tokenizer_compatible(tokenizer):
            return {}
        return {selected_name: tokenizer}

    # -------------------------------------------------------------------------
    def load_tokenizers(self, tokenizer_ids: list[str]) -> dict[str, Any]:
        tokenizers: dict[str, Any] = {}
        for tokenizer_id in tokenizer_ids:
            try:
                logger.info("Loading tokenizer: %s", tokenizer_id)
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_id,
                    cache_dir=self.get_tokenizer_cache_dir(tokenizer_id),
                    local_files_only=True,
                )
                if self.tools.is_tokenizer_compatible(tokenizer):
                    tokenizers[tokenizer_id] = tokenizer
                else:
                    logger.warning(
                        "Tokenizer %s not compatible, skipping", tokenizer_id
                    )
            except Exception:
                logger.warning(
                    "Failed to load tokenizer %s from local storage. Download it first.",
                    tokenizer_id,
                )
                logger.debug("Tokenizer load error", exc_info=True)
        return tokenizers

    # -------------------------------------------------------------------------
    def stream_dataset_rows_from_database(
        self, dataset_name: str
    ) -> Generator[tuple[int, str], None, None]:
        count = 0
        for row_id, text in self.dataset_serializer.iterate_dataset_rows_for_benchmarks(
            dataset_name=dataset_name,
            batch_size=self.streaming_batch_size,
        ):
            yield row_id, text
            count += 1
            if self.max_documents > 0 and count >= self.max_documents:
                return

    # -------------------------------------------------------------------------
    def get_dataset_document_count(self, dataset_name: str) -> int:
        return self.repository.get_dataset_document_count(dataset_name)

    # -------------------------------------------------------------------------
    def get_metric_catalog(self) -> list[dict[str, Any]]:
        return BENCHMARK_METRIC_CATALOG

    # -------------------------------------------------------------------------
    def list_benchmark_reports(self, limit: int = 200) -> list[dict[str, Any]]:
        safe_limit = max(1, int(limit))
        return self.report_serializer.list_benchmark_reports(safe_limit)

    # -------------------------------------------------------------------------
    def load_benchmark_report_by_id(self, report_id: int) -> dict[str, Any] | None:
        return self.report_serializer.load_benchmark_report_by_id(report_id)

    # -------------------------------------------------------------------------
    def save_benchmark_report(self, payload: dict[str, Any]) -> int:
        report_id = self.report_serializer.save_benchmark_report(payload)
        return int(report_id)

    # -------------------------------------------------------------------------
    def default_selected_metric_keys(self) -> list[str]:
        metric_keys: list[str] = []
        for category in BENCHMARK_METRIC_CATALOG:
            metrics = category.get("metrics", [])
            if not isinstance(metrics, list):
                continue
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                key = metric.get("key")
                if isinstance(key, str) and key:
                    metric_keys.append(key)
        return metric_keys

    # -------------------------------------------------------------------------
    def resolve_selected_metric_keys(
        self,
        selected_metric_keys: list[str] | None,
    ) -> list[str]:
        default_keys = self.default_selected_metric_keys()
        default_key_set = set(default_keys)
        if not isinstance(selected_metric_keys, list):
            return default_keys

        requested = [
            str(key) for key in selected_metric_keys if isinstance(key, str) and key
        ]
        deduplicated = list(dict.fromkeys(requested))
        filtered = [key for key in deduplicated if key in default_key_set]
        return filtered or default_keys

    # -------------------------------------------------------------------------
    def build_per_document_stats(
        self,
        local_stats: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        if local_stats.empty or "tokenizer" not in local_stats.columns:
            return []

        stats: list[dict[str, Any]] = []
        metric_columns = [
            "tokens_count",
            "tokens_to_words_ratio",
            "bytes_per_token",
            "boundary_preservation_rate",
            "round_trip_token_fidelity",
            "round_trip_text_fidelity",
            "determinism_stability",
            "bytes_per_character",
        ]

        grouped = local_stats.groupby("tokenizer", sort=False)
        for tokenizer_name, group_df in grouped:
            group = group_df.copy()
            if "text_id" in group.columns:
                group["text_id"] = pd.to_numeric(group["text_id"], errors="coerce")
                group = group.sort_values("text_id")

            tokenizer_stats: dict[str, Any] = {"tokenizer": str(tokenizer_name)}
            for column in metric_columns:
                if column not in group.columns:
                    tokenizer_stats[column] = []
                    continue
                series = pd.to_numeric(group[column], errors="coerce").fillna(0)
                if column == "tokens_count":
                    tokenizer_stats[column] = [int(value) for value in series.tolist()]
                else:
                    tokenizer_stats[column] = [
                        float(value) for value in series.tolist()
                    ]

            stats.append(tokenizer_stats)

        return stats

    # -------------------------------------------------------------------------
