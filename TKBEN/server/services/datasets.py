from __future__ import annotations

import math
import os
import shutil
import threading
from collections import Counter
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from datasets.exceptions import DataFilesNotFoundError, DatasetNotFoundError
from huggingface_hub.errors import (
    GatedRepoError,
    HFValidationError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException, Timeout

from TKBEN.server.repositories.database.backend import database
from TKBEN.server.repositories.schemas.models import DatasetDocument
from TKBEN.server.repositories.serialization.data import DatasetSerializer
from TKBEN.server.configurations import server_settings
from TKBEN.server.common.constants import DATASETS_PATH
from TKBEN.server.common.utils.logger import logger
from TKBEN.server.services.metrics.catalog import (
    DATASET_METRIC_CATALOG,
    default_selected_metric_keys,
)
from TKBEN.server.services.metrics.engine import DatasetMetricsEngine
from TKBEN.server.services.keys import HFAccessKeyService, HFAccessKeyValidationError


###############################################################################
@dataclass
class LengthStatistics:
    document_count: int = 0
    total_length: int = 0
    min_length: int | None = None
    max_length: int | None = None

    def update(self, length: int) -> None:
        self.document_count += 1
        self.total_length += length
        if self.min_length is None or length < self.min_length:
            self.min_length = length
        if self.max_length is None or length > self.max_length:
            self.max_length = length

    def resolved_min(self) -> int:
        return self.min_length if self.min_length is not None else 0

    def resolved_max(self) -> int:
        return self.max_length if self.max_length is not None else 0

    def mean(self) -> float:
        if self.document_count == 0:
            return 0.0
        return self.total_length / self.document_count


###############################################################################
@dataclass(frozen=True)
class DatasetAlias:
    hf_dataset_id: str
    default_config: str | None = None
    default_split: str | None = None


###############################################################################
@dataclass(frozen=True)
class ResolvedDatasetDownload:
    requested_corpus: str
    requested_config: str | None
    hf_dataset_id: str
    hf_config: str | None
    split: str | None


HF_DATASET_ALIASES: dict[str, DatasetAlias] = {
    "wikitext": DatasetAlias(hf_dataset_id="wikitext", default_config="wikitext-2-v1"),
    "c4": DatasetAlias(hf_dataset_id="allenai/c4", default_config="en"),
    "oscar": DatasetAlias(
        hf_dataset_id="oscar-corpus/oscar",
        default_config="unshuffled_deduplicated_en",
    ),
    "cc_news": DatasetAlias(hf_dataset_id="vblagoje/cc_news"),
    "openwebtext": DatasetAlias(hf_dataset_id="Skylion007/openwebtext"),
    "bookcorpus": DatasetAlias(hf_dataset_id="Yuti/bookcorpus"),
    "ag_news": DatasetAlias(hf_dataset_id="fancyzhx/ag_news"),
    "cnn_dailymail": DatasetAlias(hf_dataset_id="ccdv/cnn_dailymail", default_config="3.0.0"),
    "gigaword": DatasetAlias(hf_dataset_id="SalmanFaroz/gigaword"),
    "multi_news": DatasetAlias(hf_dataset_id="alexfabbri/multi_news"),
    "squad": DatasetAlias(hf_dataset_id="rajpurkar/squad"),
    "natural_questions": DatasetAlias(
        hf_dataset_id="google-research-datasets/natural_questions"
    ),
    "hotpot_qa": DatasetAlias(hf_dataset_id="hotpotqa/hotpot_qa"),
    "daily_dialog": DatasetAlias(hf_dataset_id="DeepPavlov/daily_dialog"),
    "empathetic_dialogues": DatasetAlias(hf_dataset_id="DianaW/empathetic_dialogues"),
    "openassistant_oasst1": DatasetAlias(hf_dataset_id="OpenAssistant/oasst1"),
    "yelp_review_full": DatasetAlias(hf_dataset_id="Yelp/yelp_review_full"),
    "amazon_reviews_multi": DatasetAlias(
        hf_dataset_id="mteb/amazon_reviews_multi",
        default_config="all_languages",
    ),
    "imdb": DatasetAlias(hf_dataset_id="stanfordnlp/imdb"),
    "arxiv": DatasetAlias(hf_dataset_id="ccdv/arxiv-summarization"),
    "pubmed": DatasetAlias(hf_dataset_id="ccdv/pubmed-summarization"),
    "flores": DatasetAlias(hf_dataset_id="facebook/flores", default_config="all"),
    "wiki40b": DatasetAlias(hf_dataset_id="google/wiki40b", default_config="en"),
    "opus_books": DatasetAlias(hf_dataset_id="Helsinki-NLP/opus_books", default_config="en-fr"),
}


###############################################################################
class HistogramBuilder:
    def __init__(self, stats: LengthStatistics, bins: int) -> None:
        self.stats = stats
        self.bin_count = bins if stats.document_count > 0 else 0
        self.bin_width = self._compute_bin_width()
        self.bin_edges = self._build_bin_edges()
        self.counts = [0] * (len(self.bin_edges) - 1) if self.bin_edges else []

    def _compute_bin_width(self) -> int:
        if self.stats.document_count == 0:
            return 1
        span = (self.stats.resolved_max() - self.stats.resolved_min()) + 1
        return max(1, math.ceil(span / max(1, self.bin_count)))

    def _build_bin_edges(self) -> list[int]:
        if self.stats.document_count == 0:
            return []
        edges = [self.stats.resolved_min()]
        for _ in range(self.bin_count):
            edges.append(edges[-1] + self.bin_width)
        return edges

    def add(self, length: int) -> None:
        if not self.counts:
            return
        base = self.stats.resolved_min()
        index = min(
            (length - base) // self.bin_width,
            len(self.counts) - 1,
        )
        self.counts[index] += 1

    def add_batch(self, length: int, count: int) -> None:
        if not self.counts or count <= 0:
            return
        base = self.stats.resolved_min()
        index = min(
            (length - base) // self.bin_width,
            len(self.counts) - 1,
        )
        self.counts[index] += count

    def _midpoint(self, index: int) -> float:
        start = self.bin_edges[index]
        end = self.bin_edges[index + 1]
        return (start + end - 1) / 2.0

    def _build_labels(self) -> list[str]:
        labels: list[str] = []
        if not self.counts:
            return labels
        for idx in range(len(self.counts)):
            left = int(self.bin_edges[idx])
            right = int(self.bin_edges[idx + 1] - 1)
            if right < left:
                right = left
            labels.append(f"{left}-{right}" if left != right else f"{left}")
        return labels

    def _median(self) -> float:
        if not self.counts or self.stats.document_count == 0:
            return 0.0
        low_rank = (self.stats.document_count - 1) // 2
        high_rank = self.stats.document_count // 2
        cumulative = 0
        low_value: float | None = None
        high_value: float | None = None
        for idx, count in enumerate(self.counts):
            if count == 0:
                cumulative += count
                continue
            cumulative += count
            if low_value is None and cumulative > low_rank:
                low_value = self._midpoint(idx)
            if high_value is None and cumulative > high_rank:
                high_value = self._midpoint(idx)
            if low_value is not None and high_value is not None:
                break
        if low_value is None:
            low_value = self.stats.mean()
        if high_value is None:
            high_value = low_value
        return (low_value + high_value) / 2.0

    def build(self) -> dict[str, Any]:
        if self.stats.document_count == 0:
            return {
                "bins": [],
                "counts": [],
                "bin_edges": [],
                "min_length": 0,
                "max_length": 0,
                "mean_length": 0.0,
                "median_length": 0.0,
            }
        return {
            "bins": self._build_labels(),
            "counts": self.counts,
            "bin_edges": [float(edge) for edge in self.bin_edges],
            "min_length": self.stats.resolved_min(),
            "max_length": self.stats.resolved_max(),
            "mean_length": self.stats.mean(),
            "median_length": self._median(),
        }


###############################################################################
class DatasetService:

    SUPPORTED_TEXT_FIELDS = ("text", "content", "sentence", "document", "tokens")
    REPORT_VERSION = 2
    WORD_LIST_LIMIT = 15
    WORD_CLOUD_LIMIT = 60

    def __init__(self) -> None:
        self.key_service = HFAccessKeyService()
        # Load settings from centralized configuration
        self.settings = server_settings.datasets
        self.histogram_bins = self.settings.histogram_bins
        self.streaming_batch_size = self.settings.streaming_batch_size
        self.log_interval = self.settings.log_interval
        self.dataset_serializer = DatasetSerializer()

    # -------------------------------------------------------------------------
    def get_hf_access_token_for_download(self) -> str | None:
        try:
            return self.key_service.get_active_key()
        except HFAccessKeyValidationError:
            logger.warning(
                "No decryptable active Hugging Face key found. "
                "Proceeding with anonymous dataset download."
            )
            return None

    # -------------------------------------------------------------------------
    def normalize_optional_text(self, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped if stripped else None

    # -------------------------------------------------------------------------
    def validate_non_empty_text(self, value: str, field_name: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field_name} must be a non-empty string.")
        return stripped

    # -------------------------------------------------------------------------
    def resolve_dataset_download(
        self,
        corpus: str,
        config: str | None,
    ) -> ResolvedDatasetDownload:
        requested_corpus = self.validate_non_empty_text(corpus, "Dataset id")
        if requested_corpus.lower() == "the_pile":
            raise ValueError(
                "Dataset 'the_pile' is disabled because its source requires legacy dataset scripts. "
                "Use a parquet-based alternative such as 'monology/pile-uncopyrighted'."
            )

        if isinstance(config, str) and not config.strip():
            raise ValueError("Dataset configuration cannot be blank when provided.")
        requested_config = self.normalize_optional_text(config)

        alias = HF_DATASET_ALIASES.get(requested_corpus.lower())
        hf_dataset_id = alias.hf_dataset_id if alias else requested_corpus
        hf_config = requested_config if requested_config is not None else (
            alias.default_config if alias else None
        )
        split = alias.default_split if alias else None

        hf_dataset_id = self.validate_non_empty_text(hf_dataset_id, "Resolved dataset id")
        if hf_config is not None:
            hf_config = self.validate_non_empty_text(hf_config, "Dataset configuration")
        if split is not None:
            split = self.validate_non_empty_text(split, "Dataset split")

        return ResolvedDatasetDownload(
            requested_corpus=requested_corpus,
            requested_config=requested_config,
            hf_dataset_id=hf_dataset_id,
            hf_config=hf_config,
            split=split,
        )

    # -------------------------------------------------------------------------
    def classify_download_exception(self, exc: Exception) -> str:
        message = str(exc).lower()
        if self.is_gated_or_auth_error(exc):
            return "gated_or_auth"

        if "dataset scripts are no longer supported" in message:
            return "unsupported_dataset_script"

        if isinstance(
            exc,
            (
                DatasetNotFoundError,
                DataFilesNotFoundError,
                RepositoryNotFoundError,
                HFValidationError,
            ),
        ):
            return "invalid_dataset_or_config"

        if self.is_network_error(exc):
            return "network_or_transient"

        if (
            "builderconfig" in message
            or "unknown split" in message
            or "not found" in message
            or "no (supported) data files found" in message
        ):
            return "invalid_dataset_or_config"

        if (
            "timed out" in message
            or "temporary" in message
            or "connection" in message
        ):
            return "network_or_transient"

        return "unknown"

    # -------------------------------------------------------------------------
    def is_gated_or_auth_error(self, exc: Exception) -> bool:
        if isinstance(exc, GatedRepoError):
            return True
        if isinstance(exc, HfHubHTTPError):
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            if status_code in (401, 403):
                return True
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "gated",
                "access to this dataset is restricted",
                "authentication required",
                "forbidden",
                "401",
                "403",
            )
        )

    # -------------------------------------------------------------------------
    def is_network_error(self, exc: Exception) -> bool:
        if isinstance(exc, HfHubHTTPError):
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            return status_code is None or status_code >= 500
        if isinstance(exc, (RequestsConnectionError, Timeout, TimeoutError)):
            return True
        if isinstance(exc, RequestException):
            return True
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "timed out",
                "temporary",
                "name resolution",
                "connection reset",
            )
        )

    # -------------------------------------------------------------------------
    def build_download_error_message(
        self,
        category: str,
        job_id: str | None,
        requested_dataset_name: str,
        resolved_dataset_name: str,
        has_access_token: bool,
    ) -> str:
        job_label = job_id if job_id else "n/a"
        if category == "invalid_dataset_or_config":
            return (
                f"Dataset download failed (job={job_label}): invalid dataset id or configuration "
                f"for '{requested_dataset_name}' (resolved HF id '{resolved_dataset_name}'). "
                "Verify dataset id/config on Hugging Face."
            )
        if category == "gated_or_auth":
            auth_hint = (
                "No valid decryptable Hugging Face token is currently configured."
                if not has_access_token
                else "A token was provided but access is still denied."
            )
            return (
                f"Dataset download failed (job={job_label}): access denied for "
                f"'{requested_dataset_name}' (resolved HF id '{resolved_dataset_name}'). "
                "This dataset is gated or requires authentication. Accept dataset terms and "
                f"provide a Hugging Face token with read access. {auth_hint}"
            )
        if category == "network_or_transient":
            return (
                f"Dataset download failed (job={job_label}): network/transient error while "
                f"fetching '{requested_dataset_name}' (resolved HF id '{resolved_dataset_name}'). "
                "Check connectivity and retry."
            )
        if category == "unsupported_dataset_script":
            script_hint = ""
            if "pile" in requested_dataset_name.lower() or "pile" in resolved_dataset_name.lower():
                script_hint = (
                    " For this Pile source, the official dataset currently uses legacy script loading; "
                    "try a parquet mirror such as 'monology/pile-uncopyrighted'."
                )
            return (
                f"Dataset download failed (job={job_label}): '{requested_dataset_name}' "
                f"(resolved HF id '{resolved_dataset_name}') requires a legacy dataset script "
                "that is not supported by the installed datasets library. Use an alternative "
                f"parquet-based dataset id/config or update your datasets library strategy.{script_hint}"
            )
        return (
            f"Dataset download failed (job={job_label}) for '{requested_dataset_name}' "
            f"(resolved HF id '{resolved_dataset_name}')."
        )

    # -------------------------------------------------------------------------
    def estimate_total_rows(self, dataset: Dataset | DatasetDict) -> int | None:
        if isinstance(dataset, DatasetDict):
            total_rows = 0
            for split_name in dataset.keys():
                split = dataset[split_name]
                split_rows = getattr(split, "num_rows", None)
                if not isinstance(split_rows, int) or split_rows <= 0:
                    return None
                total_rows += split_rows
            return total_rows if total_rows > 0 else None

        dataset_rows = getattr(dataset, "num_rows", None)
        if isinstance(dataset_rows, int) and dataset_rows > 0:
            return dataset_rows
        return None

    # -------------------------------------------------------------------------
    def load_dataset_with_progress(
        self,
        hf_dataset_id: str,
        hf_config: str | None,
        cache_path: str,
        hf_access_token: str | None,
        split: str | None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> Dataset | DatasetDict:
        heartbeat_stop = threading.Event()
        heartbeat_thread: threading.Thread | None = None

        if progress_callback:
            progress_callback(5.0)

            def heartbeat() -> None:
                progress_value = 5.0
                while not heartbeat_stop.wait(2.0):
                    progress_value = min(12.0, progress_value + 1.0)
                    progress_callback(progress_value)

            heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
            heartbeat_thread.start()

        try:
            load_kwargs: dict[str, Any] = {}
            if split is not None:
                load_kwargs["split"] = split
            dataset = load_dataset(
                hf_dataset_id,
                hf_config,
                cache_dir=cache_path,
                token=hf_access_token,
                **load_kwargs,
            )
            return dataset
        finally:
            if heartbeat_thread is not None:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=0.2)
            if progress_callback:
                progress_callback(15.0)

    # -------------------------------------------------------------------------
    def stop_requested(self, should_stop: Callable[[], bool] | None) -> bool:
        return bool(callable(should_stop) and should_stop())

    # -------------------------------------------------------------------------
    def cleanup_cancelled_dataset(self, dataset_name: str) -> None:
        try:
            self.dataset_serializer.delete_dataset(dataset_name)
            logger.info("Removed partially persisted dataset after cancellation: %s", dataset_name)
        except Exception:
            logger.warning(
                "Failed to cleanup partially persisted dataset after cancellation: %s",
                dataset_name,
                exc_info=True,
            )


    # -------------------------------------------------------------------------
    def get_dataset_name(self, corpus: str, config: str | None = None) -> str:
        if config:
            return f"{corpus}/{config}"
        return corpus

    # -------------------------------------------------------------------------
    def get_cache_path(self, corpus: str, config: str | None = None) -> str:
        config_suffix = f"_{config}" if config else ""
        folder_name = f"{corpus}{config_suffix}".replace("/", "_")
        return os.path.join(DATASETS_PATH, folder_name)

    # -------------------------------------------------------------------------
    def build_persisted_dataset_payload(
        self,
        dataset_name: str,
        text_column: str = "text",
    ) -> dict[str, Any]:
        length_stream = self.database_length_stream(dataset_name)
        stats = self.collect_length_statistics(length_stream)
        histogram = self.histogram_from_stream(length_stream, stats)
        return {
            "dataset_name": dataset_name,
            "text_column": text_column,
            "document_count": stats.document_count,
            "saved_count": stats.document_count,
            "histogram": histogram,
        }

    # -------------------------------------------------------------------------
    def maybe_cleanup_downloaded_source(self, cache_path: str, dataset_name: str) -> None:
        try:
            if os.path.isdir(cache_path):
                shutil.rmtree(cache_path)
            elif os.path.exists(cache_path):
                os.remove(cache_path)
            else:
                logger.info(
                    "Downloaded source already missing for %s, skipping cleanup: %s",
                    dataset_name,
                    cache_path,
                )
                return
            logger.info("Removed downloaded source for %s: %s", dataset_name, cache_path)
        except FileNotFoundError:
            logger.info(
                "Downloaded source already removed for %s, skipping cleanup: %s",
                dataset_name,
                cache_path,
            )
        except PermissionError as exc:
            if getattr(exc, "winerror", None) == 32:
                logger.warning(
                    "Could not remove downloaded source for %s because it is in use: %s. "
                    "Close processes using this path and retry.",
                    dataset_name,
                    cache_path,
                    exc_info=True,
                )
                return
            logger.warning(
                "Could not remove downloaded source for %s due to permission issues: %s. "
                "Verify delete permissions and retry.",
                dataset_name,
                cache_path,
                exc_info=True,
            )
        except OSError:
            logger.warning(
                "Failed to remove downloaded source for %s due to an OS error: %s. "
                "Check path accessibility and retry.",
                dataset_name,
                cache_path,
                exc_info=True,
            )

    # -------------------------------------------------------------------------
    def find_text_column(self, dataset: Dataset | DatasetDict) -> str | None:
        if isinstance(dataset, DatasetDict):
            sample_split = list(dataset.keys())[0]
            columns = dataset[sample_split].column_names
        else:
            columns = dataset.column_names

        for field in self.SUPPORTED_TEXT_FIELDS:
            if field in columns:
                return field

        for col in columns:
            if "text" in col.lower():
                return col

        return columns[0] if columns else None

    # -------------------------------------------------------------------------
    def _iterate_texts(
        self,
        dataset: Dataset | DatasetDict,
        text_column: str,
        remove_invalid: bool = True,
    ) -> Generator[str, None, None]:
        """
        Generator that yields texts one at a time to minimize memory usage.
        This avoids loading all texts into memory at once.
        """
        if isinstance(dataset, DatasetDict):
            for split_name in dataset.keys():
                split_data = dataset[split_name]
                for row in split_data:
                    text = row[text_column]
                    if remove_invalid:
                        if text is None or not isinstance(text, str) or not text.strip():
                            continue
                    yield text
        else:
            for row in dataset:
                text = row[text_column]
                if remove_invalid:
                    if text is None or not isinstance(text, str) or not text.strip():
                        continue
                yield text

    # -------------------------------------------------------------------------
    def dataset_cached_on_disk(self, cache_path: str) -> bool:
        if not os.path.isdir(cache_path):
            return False
        iterator = os.scandir(cache_path)
        try:
            next(iterator)
        except StopIteration:
            return False
        finally:
            iterator.close()
        return True

    # -------------------------------------------------------------------------
    def is_dataset_in_database(self, dataset_name: str) -> bool:
        return self.dataset_serializer.dataset_exists(dataset_name)

    # -------------------------------------------------------------------------
    def get_available_datasets(self) -> list[str]:
        """Get list of all unique dataset names in the database."""
        return self.dataset_serializer.list_dataset_names()

    # -------------------------------------------------------------------------
    def get_dataset_previews(self) -> list[dict[str, Any]]:
        return self.dataset_serializer.list_dataset_previews()

    # -------------------------------------------------------------------------
    def get_metric_catalog(self) -> list[dict[str, Any]]:
        return DATASET_METRIC_CATALOG

    # -------------------------------------------------------------------------
    def get_default_analysis_parameters(self) -> dict[str, Any]:
        return {
            "mattr_window": 100,
            "near_empty_threshold_words": 3,
            "rare_tail_percent": 0.10,
            "top_k_concentration": 20,
            "near_duplicate_threshold": 0.90,
            "simhash_bands": 4,
            "max_vocab_in_memory": 200_000,
        }

    # -------------------------------------------------------------------------
    def dataset_length_stream(
        self,
        dataset: Dataset | DatasetDict,
        text_column: str,
        remove_invalid: bool,
    ) -> Callable[[], Iterator[int]]:
        def generator() -> Iterator[int]:
            for text in self._iterate_texts(dataset, text_column, remove_invalid):
                yield len(text)

        return generator

    # -------------------------------------------------------------------------
    def database_length_stream(self, dataset_name: str) -> Callable[[], Iterator[int]]:
        batch_size = self.streaming_batch_size

        def generator() -> Iterator[int]:
            for batch in self.dataset_serializer.iterate_dataset_batches(
                dataset_name, batch_size
            ):
                for text in batch:
                    yield len(text)

        return generator

    # -------------------------------------------------------------------------
    def collect_length_statistics(
        self,
        stream_factory: Callable[[], Iterator[int]],
        progress_callback: Callable[[float], None] | None = None,
        progress_base: float = 0.0,
        progress_span: float = 0.0,
        estimated_total: int | None = None,
    ) -> LengthStatistics:
        stats = LengthStatistics()
        processed_count = 0
        update_every = max(1, self.log_interval)
        safe_total = estimated_total if isinstance(estimated_total, int) and estimated_total > 0 else None

        for length in stream_factory():
            stats.update(length)
            processed_count += 1
            if not progress_callback or processed_count % update_every != 0:
                continue

            if safe_total is not None:
                ratio = min(1.0, processed_count / safe_total)
                progress_callback(progress_base + (ratio * progress_span))
            else:
                ratio = min(0.9, processed_count / float(update_every * 20))
                progress_callback(progress_base + (ratio * progress_span))

        if progress_callback and progress_span > 0.0:
            progress_callback(progress_base + progress_span)
        return stats

    # -------------------------------------------------------------------------
    def histogram_from_stream(
        self,
        stream_factory: Callable[[], Iterator[int]],
        stats: LengthStatistics,
    ) -> dict[str, Any]:
        builder = HistogramBuilder(stats, self.histogram_bins)
        for length in stream_factory():
            builder.add(length)
        histogram = builder.build()
        histogram["counts"] = list(histogram.get("counts", []))
        return histogram

    # -------------------------------------------------------------------------
    def histogram_from_counts(
        self,
        stats: LengthStatistics,
        counts: dict[int, int],
    ) -> dict[str, Any]:
        builder = HistogramBuilder(stats, self.histogram_bins)
        for length, count in counts.items():
            builder.add_batch(length, count)
        histogram = builder.build()
        histogram["counts"] = list(histogram.get("counts", []))
        return histogram

    # -------------------------------------------------------------------------
    def build_word_length_items(
        self,
        word_counter: Counter[str],
        descending: bool,
    ) -> list[dict[str, Any]]:
        # Ties are resolved lexicographically to keep output deterministic.
        ranked = sorted(
            (
                {"word": word, "length": len(word), "count": int(count)}
                for word, count in word_counter.items()
                if isinstance(word, str) and word
            ),
            key=lambda item: (
                -item["length"] if descending else item["length"],
                item["word"],
            ),
        )
        return ranked[: self.WORD_LIST_LIMIT]

    # -------------------------------------------------------------------------
    def build_word_cloud_terms(self, word_counter: Counter[str]) -> list[dict[str, Any]]:
        ranked = sorted(
            (
                (word, int(count))
                for word, count in word_counter.items()
                if isinstance(word, str) and word
            ),
            key=lambda item: (-item[1], item[0]),
        )[: self.WORD_CLOUD_LIMIT]
        if not ranked:
            return []
        max_count = max(count for _, count in ranked)
        if max_count <= 0:
            return []
        return [
            {
                "word": word,
                "count": count,
                "weight": max(1, int(round((count / max_count) * 100))),
            }
            for word, count in ranked
        ]

    # -------------------------------------------------------------------------
    def persist_dataset(
        self,
        dataset: Dataset | DatasetDict,
        dataset_name: str,
        text_column: str,
        stats: LengthStatistics,
        remove_invalid: bool,
        progress_callback: Callable[[float], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        progress_base: float = 0.0,
        progress_span: float = 100.0,
    ) -> tuple[dict[str, Any], int]:
        dataset_id = self.dataset_serializer.ensure_dataset_id(dataset_name)
        batch_size = self.streaming_batch_size
        batch: list[dict[str, Any]] = []
        saved_count = 0
        last_logged = 0
        histogram_builder = HistogramBuilder(stats, self.histogram_bins)
        total_documents = stats.document_count if stats.document_count > 0 else 1

        for text in self._iterate_texts(dataset, text_column, remove_invalid):
            if should_stop and should_stop():
                return histogram_builder.build(), saved_count
            histogram_builder.add(len(text))
            batch.append({"dataset_id": dataset_id, "text": text})

            if len(batch) >= batch_size:
                df = pd.DataFrame(batch)
                database.insert_dataframe(df, DatasetDocument.__tablename__)
                saved_count += len(batch)
                if saved_count - last_logged >= self.log_interval:
                    logger.info("Saved %d documents so far...", saved_count)
                    last_logged = saved_count
                if progress_callback:
                    progress_value = progress_base + (
                        saved_count / total_documents
                    ) * progress_span
                    progress_callback(progress_value)
                batch.clear()

        if batch:
            df = pd.DataFrame(batch)
            database.insert_dataframe(df, DatasetDocument.__tablename__)
            saved_count += len(batch)
            if progress_callback:
                progress_value = progress_base + (
                    saved_count / total_documents
                ) * progress_span
                progress_callback(progress_value)

        logger.info("Completed saving %d documents to database", saved_count)
        if progress_callback and stats.document_count == 0:
            progress_callback(progress_base + progress_span)
        return histogram_builder.build(), saved_count

    # -------------------------------------------------------------------------
    def download_and_persist(
        self,
        corpus: str,
        config: str | None = None,
        remove_invalid: bool = True,
        progress_callback: Callable[[float], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_dataset_download(corpus=corpus, config=config)
        requested_dataset_name = self.get_dataset_name(
            target.requested_corpus,
            target.requested_config,
        )
        resolved_dataset_name = self.get_dataset_name(
            target.hf_dataset_id,
            target.hf_config,
        )
        dataset_name = self.get_dataset_name(
            target.requested_corpus,
            target.hf_config,
        )
        cache_path = self.get_cache_path(target.hf_dataset_id, target.hf_config)

        logger.info(
            "Starting dataset download (job=%s): requested=%s, resolved=%s, split=%s",
            job_id if job_id else "n/a",
            requested_dataset_name,
            resolved_dataset_name,
            target.split if target.split is not None else "all",
        )

        os.makedirs(DATASETS_PATH, exist_ok=True)
        os.makedirs(cache_path, exist_ok=True)

        if self.is_dataset_in_database(dataset_name):
            logger.info(
                "Dataset %s already present in database. Reusing persisted texts.",
                dataset_name,
            )
            if progress_callback:
                progress_callback(100.0)
            payload = self.build_persisted_dataset_payload(dataset_name)
            payload["cache_path"] = cache_path
            self.maybe_cleanup_downloaded_source(cache_path, dataset_name)
            return payload

        logger.info("Downloading dataset source for %s", dataset_name)

        hf_access_token = self.get_hf_access_token_for_download()

        try:
            dataset = self.load_dataset_with_progress(
                hf_dataset_id=target.hf_dataset_id,
                hf_config=target.hf_config,
                cache_path=cache_path,
                hf_access_token=hf_access_token,
                split=target.split,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            category = self.classify_download_exception(exc)
            logger.exception(
                "Dataset download failed (job=%s, category=%s): requested=%s, resolved=%s",
                job_id if job_id else "n/a",
                category,
                requested_dataset_name,
                resolved_dataset_name,
            )
            raise RuntimeError(
                self.build_download_error_message(
                    category=category,
                    job_id=job_id,
                    requested_dataset_name=requested_dataset_name,
                    resolved_dataset_name=resolved_dataset_name,
                    has_access_token=bool(hf_access_token),
                )
            ) from exc

        text_column = self.find_text_column(dataset)
        if text_column is None:
            raise ValueError(f"No text column found in dataset {dataset_name}")

        logger.info("Using text column: %s", text_column)

        length_stream = self.dataset_length_stream(
            dataset,
            text_column,
            remove_invalid,
        )
        estimated_total_rows = self.estimate_total_rows(dataset)
        stats = self.collect_length_statistics(
            length_stream,
            progress_callback=progress_callback,
            progress_base=15.0,
            progress_span=35.0,
            estimated_total=estimated_total_rows,
        )
        logger.info("Found %d valid documents", stats.document_count)

        histogram, saved_count = self.persist_dataset(
            dataset=dataset,
            dataset_name=dataset_name,
            text_column=text_column,
            stats=stats,
            remove_invalid=remove_invalid,
            progress_callback=progress_callback,
            should_stop=should_stop,
            progress_base=50.0,
            progress_span=50.0,
        )
        if self.stop_requested(should_stop) and saved_count < stats.document_count:
            self.cleanup_cancelled_dataset(dataset_name)
            return {}

        payload = {
            "dataset_name": dataset_name,
            "text_column": text_column,
            "document_count": stats.document_count,
            "saved_count": saved_count,
            "cache_path": cache_path,
            "histogram": histogram,
        }
        if not should_stop or not should_stop():
            self.maybe_cleanup_downloaded_source(cache_path, dataset_name)
        return payload

    # -------------------------------------------------------------------------
    def upload_and_persist(
        self,
        file_content: bytes,
        filename: str,
        remove_invalid: bool = True,
        progress_callback: Callable[[float], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """
        Process an uploaded CSV/Excel file and persist to database.

        Args:
            file_content: Raw bytes of the uploaded file.
            filename: Original filename (used to detect file type and derive dataset name).
            remove_invalid: If True, filter out empty or non-string texts.

        Returns:
            Dictionary with dataset_name, text_column, document_count, saved_count, histogram.
        """
        import io

        # Derive dataset name from filename (without extension)
        base_name = os.path.splitext(filename)[0]
        dataset_name = f"custom/{base_name}"
        extension = os.path.splitext(filename)[1].lower()

        logger.info("Processing uploaded file: %s (type: %s)", filename, extension)
        if self.is_dataset_in_database(dataset_name):
            logger.info(
                "Dataset %s already present in database. Reusing persisted texts.",
                dataset_name,
            )
            if progress_callback:
                progress_callback(100.0)
            return self.build_persisted_dataset_payload(dataset_name)

        # Load into DataFrame based on file extension
        try:
            if progress_callback:
                progress_callback(5.0)
            file_buffer = io.BytesIO(file_content)
            if extension == ".csv":
                df = pd.read_csv(file_buffer)
            elif extension in (".xlsx", ".xls"):
                df = pd.read_excel(file_buffer)
            else:
                raise ValueError(f"Unsupported file type: {extension}. Use .csv, .xlsx, or .xls")
        except Exception as exc:
            logger.exception("Failed to read uploaded file %s", filename)
            raise ValueError(f"Failed to read file: {exc}") from exc

        if df.empty:
            raise ValueError("Uploaded file contains no data")

        # Find text column
        text_column = self._find_text_column_in_dataframe(df)
        if text_column is None:
            raise ValueError(
                f"No text column found in uploaded file. "
                f"Expected one of: {self.SUPPORTED_TEXT_FIELDS}"
            )

        logger.info("Using text column: %s", text_column)

        # Create a generator for streaming texts from the DataFrame
        def iterate_df_texts() -> Iterator[str]:
            for value in df[text_column]:
                if remove_invalid:
                    if value is None or not isinstance(value, str) or not value.strip():
                        continue
                yield str(value)

        # Collect length statistics (first pass)
        def length_stream() -> Iterator[int]:
            for text in iterate_df_texts():
                yield len(text)

        stats = self.collect_length_statistics(length_stream)
        logger.info("Found %d valid documents in uploaded file", stats.document_count)

        if stats.document_count == 0:
            raise ValueError("No valid text documents found after filtering")
        if progress_callback:
            progress_callback(15.0)

        # Persist to database with histogram computation (second pass)
        batch_size = self.streaming_batch_size
        batch: list[dict[str, Any]] = []
        saved_count = 0
        last_logged = 0
        histogram_builder = HistogramBuilder(stats, self.histogram_bins)
        dataset_id = self.dataset_serializer.ensure_dataset_id(dataset_name)
        cancelled = False

        for text in iterate_df_texts():
            if self.stop_requested(should_stop):
                cancelled = True
                break
            histogram_builder.add(len(text))
            batch.append({"dataset_id": dataset_id, "text": text})

            if len(batch) >= batch_size:
                batch_df = pd.DataFrame(batch)
                database.insert_dataframe(batch_df, DatasetDocument.__tablename__)
                saved_count += len(batch)
                if saved_count - last_logged >= self.log_interval:
                    logger.info("Saved %d documents so far...", saved_count)
                    last_logged = saved_count
                if progress_callback:
                    progress_value = 15.0 + (
                        saved_count / max(stats.document_count, 1)
                    ) * 85.0
                    progress_callback(progress_value)
                batch.clear()

        if batch:
            batch_df = pd.DataFrame(batch)
            database.insert_dataframe(batch_df, DatasetDocument.__tablename__)
            saved_count += len(batch)
            if progress_callback:
                progress_value = 15.0 + (
                    saved_count / max(stats.document_count, 1)
                ) * 85.0
                progress_callback(progress_value)

        if cancelled and saved_count < stats.document_count:
            self.cleanup_cancelled_dataset(dataset_name)
            return {}

        logger.info("Completed saving %d documents from uploaded file", saved_count)

        return {
            "dataset_name": dataset_name,
            "text_column": text_column,
            "document_count": stats.document_count,
            "saved_count": saved_count,
            "histogram": histogram_builder.build(),
        }

    # -------------------------------------------------------------------------
    def _find_text_column_in_dataframe(self, df: pd.DataFrame) -> str | None:
        """Find a suitable text column in a pandas DataFrame."""
        columns = list(df.columns)

        for field in self.SUPPORTED_TEXT_FIELDS:
            if field in columns:
                return field

        for col in columns:
            if "text" in col.lower():
                return col

        return columns[0] if columns else None

    # -------------------------------------------------------------------------
    def analyze_dataset(
        self,
        dataset_name: str,
        session_name: str | None = None,
        selected_metric_keys: list[str] | None = None,
        sampling: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
        metric_parameters: dict[str, Any] | None = None,
        progress_callback: Callable[[float], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        use_cached: bool = True,
    ) -> dict[str, Any]:
        return self.validate_dataset(
            dataset_name=dataset_name,
            session_name=session_name,
            selected_metric_keys=selected_metric_keys,
            sampling=sampling,
            filters=filters,
            metric_parameters=metric_parameters,
            progress_callback=progress_callback,
            should_stop=should_stop,
            use_cached=use_cached,
        )

    # -------------------------------------------------------------------------
    def validate_dataset(
        self,
        dataset_name: str,
        session_name: str | None = None,
        selected_metric_keys: list[str] | None = None,
        sampling: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
        metric_parameters: dict[str, Any] | None = None,
        progress_callback: Callable[[float], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        use_cached: bool = True,
    ) -> dict[str, Any]:
        sampling_config = sampling if isinstance(sampling, dict) else {}
        filter_config = filters if isinstance(filters, dict) else {}
        parameter_overrides = metric_parameters if isinstance(metric_parameters, dict) else {}
        has_custom_request = bool(
            session_name
            or selected_metric_keys
            or sampling_config
            or filter_config
            or parameter_overrides
        )

        cached_report = self.dataset_serializer.load_latest_analysis_report(dataset_name)
        if use_cached and not has_custom_request and cached_report is not None:
            if progress_callback:
                progress_callback(100.0)
            return cached_report

        if not self.dataset_serializer.dataset_exists(dataset_name):
            raise ValueError(f"Dataset '{dataset_name}' not found.")

        selected_keys = (
            [key for key in selected_metric_keys if isinstance(key, str) and key]
            if isinstance(selected_metric_keys, list)
            else default_selected_metric_keys()
        )
        selected_key_set = set(selected_keys)
        if not selected_key_set:
            selected_key_set = set(default_selected_metric_keys())

        parameters = self.get_default_analysis_parameters()
        parameters.update(parameter_overrides)

        self.dataset_serializer.ensure_metric_types_seeded(self.get_metric_catalog())

        session_parameters = {
            "sampling": sampling_config,
            "filters": filter_config,
            "metric_parameters": parameters,
        }
        session_id = self.dataset_serializer.create_analysis_session(
            dataset_name=dataset_name,
            session_name=session_name,
            selected_metric_keys=sorted(selected_key_set),
            parameters=session_parameters,
            report_version=self.REPORT_VERSION,
        )

        min_length = filter_config.get("min_length")
        max_length = filter_config.get("max_length")
        exclude_empty = bool(filter_config.get("exclude_empty", False))
        sample_fraction = sampling_config.get("fraction")
        sample_count = sampling_config.get("count")
        normalized_fraction = (
            float(sample_fraction)
            if isinstance(sample_fraction, (int, float)) and 0 < float(sample_fraction) < 1
            else None
        )
        normalized_count = (
            int(sample_count)
            if isinstance(sample_count, (int, float)) and int(sample_count) > 0
            else None
        )

        engine = DatasetMetricsEngine(parameters=parameters)
        per_doc_buffer: list[dict[str, Any]] = []
        aggregate_total = self.dataset_serializer.count_dataset_documents(dataset_name)
        expected_total = aggregate_total
        if normalized_count is not None:
            expected_total = min(expected_total, normalized_count)
        if normalized_fraction is not None:
            expected_total = max(1, int(math.ceil(expected_total * normalized_fraction)))

        analyzed = 0
        persisted = 0
        for batch in self.dataset_serializer.iterate_dataset_rows(
            dataset_name=dataset_name,
            batch_size=self.streaming_batch_size,
            min_length=min_length if isinstance(min_length, int) else None,
            max_length=max_length if isinstance(max_length, int) else None,
            exclude_empty=exclude_empty,
        ):
            if self.stop_requested(should_stop):
                self.dataset_serializer.complete_analysis_session(session_id, status="cancelled")
                return {}

            for row in batch:
                text_id = row.get("id")
                text = row.get("text")
                if text_id is None or not isinstance(text, str):
                    continue
                if normalized_fraction is not None:
                    gate = (int(text_id) % 1_000_000) / 1_000_000.0
                    if gate > normalized_fraction:
                        continue
                per_doc_metrics = engine.process_document(int(text_id), text)
                for metric_row in per_doc_metrics:
                    if metric_row.get("metric_key") in selected_key_set:
                        per_doc_buffer.append(metric_row)
                analyzed += 1
                if normalized_count is not None and analyzed >= normalized_count:
                    break
                if len(per_doc_buffer) >= self.streaming_batch_size:
                    self.dataset_serializer.save_metric_values_batch(session_id, per_doc_buffer)
                    persisted += len(per_doc_buffer)
                    per_doc_buffer.clear()

            if per_doc_buffer and len(per_doc_buffer) >= self.streaming_batch_size:
                self.dataset_serializer.save_metric_values_batch(session_id, per_doc_buffer)
                persisted += len(per_doc_buffer)
                per_doc_buffer.clear()

            if normalized_count is not None and analyzed >= normalized_count:
                break
            if progress_callback and expected_total > 0:
                progress_callback(min(95.0, (analyzed / float(expected_total)) * 95.0))

        if per_doc_buffer:
            self.dataset_serializer.save_metric_values_batch(session_id, per_doc_buffer)
            persisted += len(per_doc_buffer)

        finalized = engine.finalize(histogram_bins=self.histogram_bins)
        aggregate_rows = [
            row
            for row in finalized.get("metric_rows", [])
            if row.get("metric_key") in selected_key_set
        ]
        self.dataset_serializer.save_metric_values_batch(session_id, aggregate_rows)
        self.dataset_serializer.save_histogram_artifact(
            session_id=session_id,
            metric_key="hist.document_length",
            histogram=finalized.get("document_histogram", {}),
        )
        self.dataset_serializer.save_histogram_artifact(
            session_id=session_id,
            metric_key="hist.word_length",
            histogram=finalized.get("word_histogram", {}),
        )
        self.dataset_serializer.complete_analysis_session(session_id, status="completed")

        report = self.dataset_serializer.load_analysis_report_by_session_id(session_id)
        if report is None:
            raise ValueError("Failed to load persisted analysis report.")
        logger.info(
            "Completed analysis session %d for dataset %s (documents=%d, persisted_rows=%d)",
            session_id,
            dataset_name,
            analyzed,
            persisted + len(aggregate_rows),
        )
        if progress_callback:
            progress_callback(100.0)
        return report

    # -------------------------------------------------------------------------
    def get_latest_validation_report(self, dataset_name: str) -> dict[str, Any] | None:
        return self.dataset_serializer.load_latest_analysis_report(dataset_name)

    # -------------------------------------------------------------------------
    def get_validation_report_by_id(self, report_id: int) -> dict[str, Any] | None:
        return self.dataset_serializer.load_analysis_report_by_session_id(report_id)

    # -------------------------------------------------------------------------
    def get_analysis_summary(self, dataset_name: str) -> dict[str, Any]:
        report = self.dataset_serializer.load_latest_analysis_report(dataset_name)
        if report is None:
            return {
                "total_documents": 0,
                "mean_words_count": 0.0,
                "median_words_count": 0.0,
                "mean_avg_word_length": 0.0,
                "mean_std_word_length": 0.0,
            }
        aggregate = report.get("aggregate_statistics", {})
        return {
            "total_documents": int(aggregate.get("corpus.document_count", 0) or 0),
            "mean_words_count": float(aggregate.get("doc.length_mean", 0.0) or 0.0),
            "median_words_count": float(aggregate.get("doc.length_p50", 0.0) or 0.0),
            "mean_avg_word_length": float(aggregate.get("words.length_mean", 0.0) or 0.0),
            "mean_std_word_length": float(aggregate.get("words.length_std", 0.0) or 0.0),
        }

    # -------------------------------------------------------------------------
    def is_dataset_analyzed(self, dataset_name: str) -> bool:
        """Check if a dataset has been analyzed."""
        report = self.dataset_serializer.load_latest_analysis_report(dataset_name)
        return report is not None

    # -------------------------------------------------------------------------
    def remove_dataset(self, dataset_name: str) -> None:
        self.dataset_serializer.delete_dataset(dataset_name)


