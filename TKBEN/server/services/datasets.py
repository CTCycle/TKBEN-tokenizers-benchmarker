from __future__ import annotations

import heapq
import math
import os
from collections import Counter
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from typing import Any

import pandas as pd
import sqlalchemy
from datasets import Dataset, DatasetDict, load_dataset

from TKBEN.server.repositories.database import database
from TKBEN.server.repositories.schema import TextDataset
from TKBEN.server.repositories.serializer import DatasetSerializer
from TKBEN.server.configurations import server_settings
from TKBEN.server.utils.constants import SOURCES_PATH
from TKBEN.server.utils.logger import logger


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

    def __init__(self, hf_access_token: str | None = None) -> None:
        self.hf_access_token = hf_access_token
        # Load settings from centralized configuration
        self.settings = server_settings.datasets
        self.histogram_bins = self.settings.histogram_bins
        self.streaming_batch_size = self.settings.streaming_batch_size
        self.log_interval = self.settings.log_interval
        self.dataset_serializer = DatasetSerializer()


    # -------------------------------------------------------------------------
    def get_dataset_name(self, corpus: str, config: str | None = None) -> str:
        if config:
            return f"{corpus}/{config}"
        return corpus

    # -------------------------------------------------------------------------
    def get_cache_path(self, corpus: str, config: str | None = None) -> str:
        config_suffix = f"_{config}" if config else ""
        folder_name = f"{corpus}{config_suffix}".replace("/", "_")
        return os.path.join(SOURCES_PATH, folder_name)

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
        self, stream_factory: Callable[[], Iterator[int]]
    ) -> LengthStatistics:
        stats = LengthStatistics()
        for length in stream_factory():
            stats.update(length)
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
        batch_size = self.streaming_batch_size
        batch: list[dict[str, str]] = []
        saved_count = 0
        last_logged = 0
        histogram_builder = HistogramBuilder(stats, self.histogram_bins)
        total_documents = stats.document_count if stats.document_count > 0 else 1

        database.delete_by_key(
            TextDataset.__tablename__,
            "dataset_name",
            dataset_name,
        )

        for text in self._iterate_texts(dataset, text_column, remove_invalid):
            if should_stop and should_stop():
                return histogram_builder.build(), saved_count
            histogram_builder.add(len(text))
            batch.append({"dataset_name": dataset_name, "text": text})

            if len(batch) >= batch_size:
                df = pd.DataFrame(batch)
                database.insert_dataframe(df, TextDataset.__tablename__)
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
            database.insert_dataframe(df, TextDataset.__tablename__)
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
    ) -> dict[str, Any]:
        dataset_name = self.get_dataset_name(corpus, config)
        cache_path = self.get_cache_path(corpus, config)

        os.makedirs(cache_path, exist_ok=True)

        if self.is_dataset_in_database(dataset_name):
            logger.info(
                "Dataset %s already present in database. Reusing persisted texts.",
                dataset_name,
            )
            length_stream = self.database_length_stream(dataset_name)
            stats = self.collect_length_statistics(length_stream)
            histogram = self.histogram_from_stream(length_stream, stats)
            if progress_callback:
                progress_callback(100.0)
            return {
                "dataset_name": dataset_name,
                "text_column": "text",
                "document_count": stats.document_count,
                "saved_count": stats.document_count,
                "cache_path": cache_path,
                "histogram": histogram,
            }

        if self.dataset_cached_on_disk(cache_path):
            logger.info("Dataset cache found on disk: %s", cache_path)
        else:
            logger.info("Dataset cache not found. Downloading %s", dataset_name)

        try:
            if progress_callback:
                progress_callback(5.0)
            dataset = load_dataset(
                corpus,
                config,
                cache_dir=cache_path,
                token=self.hf_access_token,
            )
        except Exception:
            logger.exception("Failed to download dataset %s", dataset_name)
            raise

        text_column = self.find_text_column(dataset)
        if text_column is None:
            raise ValueError(f"No text column found in dataset {dataset_name}")

        logger.info("Using text column: %s", text_column)
        if progress_callback:
            progress_callback(15.0)

        length_stream = self.dataset_length_stream(
            dataset,
            text_column,
            remove_invalid,
        )
        stats = self.collect_length_statistics(length_stream)
        logger.info("Found %d valid documents", stats.document_count)

        histogram, saved_count = self.persist_dataset(
            dataset=dataset,
            dataset_name=dataset_name,
            text_column=text_column,
            stats=stats,
            remove_invalid=remove_invalid,
            progress_callback=progress_callback,
            should_stop=should_stop,
            progress_base=20.0,
            progress_span=80.0,
        )

        return {
            "dataset_name": dataset_name,
            "text_column": text_column,
            "document_count": stats.document_count,
            "saved_count": saved_count,
            "cache_path": cache_path,
            "histogram": histogram,
        }

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
        batch: list[dict[str, str]] = []
        saved_count = 0
        last_logged = 0
        histogram_builder = HistogramBuilder(stats, self.histogram_bins)

        # Delete any existing entries with this dataset name
        database.delete_by_key(
            TextDataset.__tablename__,
            "dataset_name",
            dataset_name,
        )

        for text in iterate_df_texts():
            if should_stop and should_stop():
                return {}
            histogram_builder.add(len(text))
            batch.append({"dataset_name": dataset_name, "text": text})

            if len(batch) >= batch_size:
                batch_df = pd.DataFrame(batch)
                database.insert_dataframe(batch_df, TextDataset.__tablename__)
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
            database.insert_dataframe(batch_df, TextDataset.__tablename__)
            saved_count += len(batch)
            if progress_callback:
                progress_value = 15.0 + (
                    saved_count / max(stats.document_count, 1)
                ) * 85.0
                progress_callback(progress_value)

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
        progress_callback: Callable[[float], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        return self.validate_dataset(
            dataset_name=dataset_name,
            progress_callback=progress_callback,
            should_stop=should_stop,
        )

    # -------------------------------------------------------------------------
    def validate_dataset(
        self,
        dataset_name: str,
        progress_callback: Callable[[float], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        cached_report = self.dataset_serializer.load_dataset_report(dataset_name)
        if cached_report is not None:
            if progress_callback:
                progress_callback(100.0)
            return cached_report

        if not self.dataset_serializer.dataset_exists(dataset_name):
            raise ValueError(f"Dataset '{dataset_name}' not found.")

        total_documents = self.dataset_serializer.count_dataset_documents(dataset_name)
        if progress_callback:
            progress_callback(5.0)

        self.dataset_serializer.delete_dataset_statistics(dataset_name)

        logger.info("Starting validation for dataset: %s", dataset_name)

        document_stats = LengthStatistics()
        word_stats = LengthStatistics()
        document_counts: dict[int, int] = {}
        word_counts: dict[int, int] = {}
        word_counter: Counter[str] = Counter()
        analyzed_count = 0

        batch_size = self.streaming_batch_size

        for batch in self.dataset_serializer.iterate_dataset_batches(
            dataset_name, batch_size
        ):
            if should_stop and should_stop():
                return {}

            stats_batch: list[dict[str, Any]] = []

            for text in batch:
                if not text or not isinstance(text, str):
                    continue

                document_length = len(text)
                document_stats.update(document_length)
                document_counts[document_length] = document_counts.get(document_length, 0) + 1

                words = text.split()
                words_count = len(words)

                if words_count > 0:
                    word_lengths = [len(word) for word in words]
                    avg_word_length = sum(word_lengths) / words_count
                    variance = (
                        sum((length - avg_word_length) ** 2 for length in word_lengths)
                        / words_count
                    )
                    std_word_length = variance**0.5
                    for length in word_lengths:
                        word_stats.update(length)
                        word_counts[length] = word_counts.get(length, 0) + 1
                    word_counter.update(words)
                else:
                    avg_word_length = 0.0
                    std_word_length = 0.0

                stats_batch.append(
                    {
                        "dataset_name": dataset_name,
                        "text": text,
                        "words_count": words_count,
                        "AVG_words_length": avg_word_length,
                        "STD_words_length": std_word_length,
                    }
                )
                analyzed_count += 1

            self.dataset_serializer.save_dataset_statistics_batch(stats_batch)

            if analyzed_count and analyzed_count % self.log_interval == 0:
                logger.info("Validated %d documents so far...", analyzed_count)

            if progress_callback and total_documents > 0:
                progress_value = (analyzed_count / total_documents) * 100.0
                progress_callback(progress_value)

        document_histogram = self.histogram_from_counts(
            document_stats, document_counts
        )
        word_histogram = self.histogram_from_counts(word_stats, word_counts)

        most_common_words = [
            {"word": word, "count": int(count)}
            for word, count in word_counter.most_common(10)
        ]
        least_common_words = [
            {"word": word, "count": int(count)}
            for word, count in heapq.nsmallest(
                10, word_counter.items(), key=lambda item: (item[1], item[0])
            )
        ]

        report = {
            "dataset_name": dataset_name,
            "document_count": document_stats.document_count,
            "document_length_histogram": document_histogram,
            "word_length_histogram": word_histogram,
            "min_document_length": document_histogram.get("min_length", 0),
            "max_document_length": document_histogram.get("max_length", 0),
            "most_common_words": most_common_words,
            "least_common_words": least_common_words,
        }

        self.dataset_serializer.save_dataset_report(report)

        logger.info("Completed validation: %d documents analyzed", analyzed_count)

        if progress_callback:
            progress_callback(100.0)

        return report

    # -------------------------------------------------------------------------
    def get_analysis_summary(self, dataset_name: str) -> dict[str, Any]:
        """
        Query aggregate word-level statistics from persisted analysis data.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            Dictionary with aggregate statistics.
        """
        query = sqlalchemy.text(
            'SELECT COUNT(*) as total, '
            'AVG("words_count") as mean_words, '
            'AVG("AVG_words_length") as mean_avg_len, '
            'AVG("STD_words_length") as mean_std_len '
            'FROM "TEXT_DATASET_STATISTICS" WHERE "dataset_name" = :dataset'
        )

        with database.backend.engine.connect() as conn:
            result = conn.execute(query, {"dataset": dataset_name})
            row = result.first()

        if row is None:
            return {
                "total_documents": 0,
                "mean_words_count": 0.0,
                "median_words_count": 0.0,
                "mean_avg_word_length": 0.0,
                "mean_std_word_length": 0.0,
            }

        if hasattr(row, "_mapping"):
            data = row._mapping
        else:
            data = {
                "total": row[0],
                "mean_words": row[1],
                "mean_avg_len": row[2],
                "mean_std_len": row[3],
            }

        total = data.get("total", 0) or 0
        mean_words = data.get("mean_words", 0.0) or 0.0
        mean_avg_len = data.get("mean_avg_len", 0.0) or 0.0
        mean_std_len = data.get("mean_std_len", 0.0) or 0.0

        # Compute median using streaming (more memory efficient)
        median_words = self._compute_median_words_count(dataset_name)

        return {
            "total_documents": int(total),
            "mean_words_count": float(mean_words),
            "median_words_count": float(median_words),
            "mean_avg_word_length": float(mean_avg_len),
            "mean_std_word_length": float(mean_std_len),
        }

    # -------------------------------------------------------------------------
    def _compute_median_words_count(self, dataset_name: str) -> float:
        """Compute median word count using SQL for efficiency."""
        # For SQLite and Postgres, use PERCENTILE_CONT or ORDER BY with LIMIT
        count_query = sqlalchemy.text(
            'SELECT COUNT(*) FROM "TEXT_DATASET_STATISTICS" '
            'WHERE "dataset_name" = :dataset'
        )

        with database.backend.engine.connect() as conn:
            result = conn.execute(count_query, {"dataset": dataset_name})
            count_row = result.first()
            total = count_row[0] if count_row else 0

        if total == 0:
            return 0.0

        # Get median using OFFSET/LIMIT
        offset = total // 2
        median_query = sqlalchemy.text(
            'SELECT "words_count" FROM "TEXT_DATASET_STATISTICS" '
            'WHERE "dataset_name" = :dataset '
            'ORDER BY "words_count" LIMIT 1 OFFSET :offset'
        )

        with database.backend.engine.connect() as conn:
            result = conn.execute(median_query, {"dataset": dataset_name, "offset": offset})
            median_row = result.first()

        if median_row is None:
            return 0.0

        median_val = median_row[0] if median_row else 0
        return float(median_val)

    # -------------------------------------------------------------------------
    def is_dataset_analyzed(self, dataset_name: str) -> bool:
        """Check if a dataset has been analyzed."""
        report = self.dataset_serializer.load_dataset_report(dataset_name)
        return report is not None

    # -------------------------------------------------------------------------
    def remove_dataset(self, dataset_name: str) -> None:
        self.dataset_serializer.delete_dataset(dataset_name)

