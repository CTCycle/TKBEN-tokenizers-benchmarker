from __future__ import annotations

import math
import os
import shutil
import threading
import time  # noqa: F401
from collections import Counter
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from functools import partial
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
from sqlalchemy.exc import SQLAlchemyError

from TKBEN.server.repositories.serialization.data import DatasetSerializer
from TKBEN.server.configurations import get_server_settings
from TKBEN.server.common.constants import DATASETS_PATH
from TKBEN.server.common.utils.logger import logger
from TKBEN.server.common.utils.security import (
    ensure_path_is_within,
    normalize_identifier,
    normalize_optional_identifier,
)
from TKBEN.server.services.metrics.catalog import DATASET_METRIC_CATALOG
from TKBEN.server.services.keys import HFAccessKeyService, HFAccessKeyValidationError
from TKBEN.server.services.dataset_operations import DatasetServiceOperationsMixin


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
    "cnn_dailymail": DatasetAlias(
        hf_dataset_id="ccdv/cnn_dailymail", default_config="3.0.0"
    ),
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
    "opus_books": DatasetAlias(
        hf_dataset_id="Helsinki-NLP/opus_books", default_config="en-fr"
    ),
}

DATASET_CONFIGURATION_FIELD = "Dataset configuration"
DATASET_ID_FIELD = "Dataset id"


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
        low_value, high_value = self._resolve_median_bounds()
        return (low_value + high_value) / 2.0

    def _resolve_median_bounds(self) -> tuple[float, float]:
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
        return low_value, high_value

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
class DatasetService(DatasetServiceOperationsMixin):
    SUPPORTED_TEXT_FIELDS = ("text", "content", "sentence", "document", "tokens")
    REPORT_VERSION = 2
    WORD_LIST_LIMIT = 15
    WORD_CLOUD_LIMIT = 60

    def __init__(self) -> None:
        self.key_service = HFAccessKeyService()
        # Load settings from centralized configuration
        self.settings = get_server_settings().datasets
        self.histogram_bins = self.settings.histogram_bins
        self.streaming_batch_size = self.settings.streaming_batch_size
        self.log_interval = self.settings.log_interval
        self.download_timeout_seconds = self.settings.download_timeout_seconds
        self.download_retry_attempts = self.settings.download_retry_attempts
        self.download_retry_backoff_seconds = self.settings.download_retry_backoff_seconds
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
        return normalize_optional_identifier(
            value,
            DATASET_CONFIGURATION_FIELD,
            max_length=120,
        )

    # -------------------------------------------------------------------------
    def validate_non_empty_text(self, value: str, field_name: str) -> str:
        max_length = 160
        if field_name == "Dataset split":
            max_length = 120
        return normalize_identifier(value, field_name, max_length=max_length)

    # -------------------------------------------------------------------------
    def resolve_dataset_download(
        self,
        corpus: str,
        config: str | None,
    ) -> ResolvedDatasetDownload:
        requested_corpus = self.validate_non_empty_text(corpus, DATASET_ID_FIELD)
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
        hf_config = requested_config
        if hf_config is None and alias is not None:
            hf_config = alias.default_config
        split = alias.default_split if alias else None

        hf_dataset_id = self.validate_non_empty_text(
            hf_dataset_id, "Resolved dataset id"
        )
        if hf_config is not None:
            hf_config = self.validate_non_empty_text(
                hf_config, DATASET_CONFIGURATION_FIELD
            )
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

        if "timed out" in message or "temporary" in message or "connection" in message:
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
            if (
                "pile" in requested_dataset_name.lower()
                or "pile" in resolved_dataset_name.lower()
            ):
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
    def should_retry_download(self, category: str, attempt: int, max_attempts: int) -> bool:
        return category == "network_or_transient" and attempt < max_attempts

    # -------------------------------------------------------------------------
    def retry_delay_seconds(self, attempt: int) -> float:
        base = max(0.0, float(self.download_retry_backoff_seconds))
        if base <= 0.0:
            return 0.0
        return base * (2 ** max(0, attempt - 1))

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
    @staticmethod
    def _heartbeat_progress(
        stop_event: threading.Event,
        progress_callback: Callable[[float], None],
    ) -> None:
        progress_value = 5.0
        while not stop_event.wait(2.0):
            progress_value = min(12.0, progress_value + 1.0)
            progress_callback(progress_value)

    # -------------------------------------------------------------------------
    @staticmethod
    def _load_dataset_worker(
        result_holder: dict[str, Dataset | DatasetDict],
        error_holder: dict[str, Exception],
        hf_dataset_id: str,
        hf_config: str | None,
        cache_path: str,
        hf_access_token: str | None,
        load_kwargs: dict[str, str],
    ) -> None:
        try:
            result_holder["dataset"] = load_dataset(
                hf_dataset_id,
                hf_config,
                cache_dir=cache_path,
                token=hf_access_token,
                **load_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            error_holder["error"] = exc

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
            heartbeat_thread = threading.Thread(
                target=self._heartbeat_progress,
                args=(heartbeat_stop, progress_callback),
                daemon=True,
            )
            heartbeat_thread.start()

        try:
            load_kwargs: dict[str, str] = {}
            if split is not None:
                load_kwargs["split"] = split

            result_holder: dict[str, Dataset | DatasetDict] = {}
            error_holder: dict[str, Exception] = {}

            worker_thread = threading.Thread(
                target=self._load_dataset_worker,
                args=(
                    result_holder,
                    error_holder,
                    hf_dataset_id,
                    hf_config,
                    cache_path,
                    hf_access_token,
                    load_kwargs,
                ),
                daemon=True,
            )
            worker_thread.start()
            worker_thread.join(timeout=float(self.download_timeout_seconds))

            if worker_thread.is_alive():
                raise TimeoutError(
                    f"Dataset download timed out after {self.download_timeout_seconds:.1f} seconds."
                )

            worker_error = error_holder.get("error")
            if worker_error is not None:
                raise worker_error

            dataset = result_holder.get("dataset")
            if dataset is None:
                raise RuntimeError(
                    "Dataset download failed before producing a dataset result."
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
            logger.info(
                "Removed partially persisted dataset after cancellation: %s",
                dataset_name,
            )
        except (SQLAlchemyError, OSError, ValueError, RuntimeError):
            logger.warning(
                "Failed to cleanup partially persisted dataset after cancellation: %s",
                dataset_name,
                exc_info=True,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Unexpected error while cleaning up cancelled dataset: %s",
                dataset_name,
            )

    # -------------------------------------------------------------------------
    def get_dataset_name(self, corpus: str, config: str | None = None) -> str:
        safe_corpus = normalize_identifier(corpus, DATASET_ID_FIELD, max_length=160)
        if config:
            safe_config = normalize_identifier(
                config,
                DATASET_CONFIGURATION_FIELD,
                max_length=120,
            )
            return f"{safe_corpus}/{safe_config}"
        return safe_corpus

    # -------------------------------------------------------------------------
    def get_cache_path(self, corpus: str, config: str | None = None) -> str:
        safe_corpus = normalize_identifier(corpus, DATASET_ID_FIELD, max_length=160)
        safe_config = normalize_optional_identifier(
            config,
            DATASET_CONFIGURATION_FIELD,
            max_length=120,
        )
        folder_name = safe_corpus.replace("/", "__")
        if safe_config:
            folder_name = f"{folder_name}__{safe_config.replace('/', '__')}"
        candidate = os.path.join(DATASETS_PATH, folder_name)
        return ensure_path_is_within(DATASETS_PATH, candidate)

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
    def maybe_cleanup_downloaded_source(
        self, cache_path: str, dataset_name: str
    ) -> None:
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
            logger.info(
                "Removed downloaded source for %s: %s", dataset_name, cache_path
            )
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
        for row in self._iterate_rows(dataset):
            text = self._extract_text_value(row, text_column, remove_invalid)
            if text is not None:
                yield text

    # -------------------------------------------------------------------------
    @staticmethod
    def _iterate_rows(dataset: Dataset | DatasetDict) -> Iterator[dict[str, Any]]:
        if isinstance(dataset, DatasetDict):
            for split_name in dataset.keys():
                for row in dataset[split_name]:
                    yield row
            return
        for row in dataset:
            yield row

    # -------------------------------------------------------------------------
    @staticmethod
    def _extract_text_value(
        row: dict[str, Any],
        text_column: str,
        remove_invalid: bool,
    ) -> str | None:
        text = row[text_column]
        if remove_invalid and (
            text is None or not isinstance(text, str) or not text.strip()
        ):
            return None
        return text if isinstance(text, str) else str(text)

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
    def _iterate_dataset_lengths(
        self,
        dataset: Dataset | DatasetDict,
        text_column: str,
        remove_invalid: bool,
    ) -> Iterator[int]:
        for text in self._iterate_texts(dataset, text_column, remove_invalid):
            yield len(text)

    # -------------------------------------------------------------------------
    def _iterate_database_lengths(
        self,
        dataset_name: str,
        batch_size: int,
    ) -> Iterator[int]:
        for batch in self.dataset_serializer.iterate_dataset_batches(
            dataset_name, batch_size
        ):
            for text in batch:
                yield len(text)

    # -------------------------------------------------------------------------
    @staticmethod
    def _iterate_dataframe_texts(
        dataframe: pd.DataFrame,
        text_column: str,
        remove_invalid: bool,
    ) -> Iterator[str]:
        for value in dataframe[text_column]:
            if remove_invalid and (
                value is None or not isinstance(value, str) or not value.strip()
            ):
                continue
            yield str(value)

    # -------------------------------------------------------------------------
    def _dataframe_length_stream(
        self,
        dataframe: pd.DataFrame,
        text_column: str,
        remove_invalid: bool,
    ) -> Iterator[int]:
        for text in self._iterate_dataframe_texts(
            dataframe, text_column, remove_invalid
        ):
            yield len(text)

    # -------------------------------------------------------------------------
    def dataset_length_stream(
        self,
        dataset: Dataset | DatasetDict,
        text_column: str,
        remove_invalid: bool,
    ) -> Callable[[], Iterator[int]]:
        return partial(
            self._iterate_dataset_lengths,
            dataset,
            text_column,
            remove_invalid,
        )

    # -------------------------------------------------------------------------
    def database_length_stream(self, dataset_name: str) -> Callable[[], Iterator[int]]:
        return partial(
            self._iterate_database_lengths,
            dataset_name,
            self.streaming_batch_size,
        )

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
        safe_total = (
            estimated_total
            if isinstance(estimated_total, int) and estimated_total > 0
            else None
        )

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
    def build_word_cloud_terms(
        self, word_counter: Counter[str]
    ) -> list[dict[str, Any]]:
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
