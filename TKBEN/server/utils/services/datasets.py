from __future__ import annotations

import math
import os
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from typing import Any

import pandas as pd
import sqlalchemy
from datasets import Dataset, DatasetDict, load_dataset

from TKBEN.server.database.database import database
from TKBEN.server.database.schema import TextDataset
from TKBEN.server.utils.configurations import server_settings
from TKBEN.server.utils.constants import DATASETS_PATH
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
        self._settings = server_settings.datasets
        self.histogram_bins = self._settings.histogram_bins
        self.streaming_batch_size = self._settings.streaming_batch_size
        self.log_interval = self._settings.log_interval


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
        query = sqlalchemy.text(
            'SELECT 1 FROM "TEXT_DATASET" WHERE "dataset_name" = :dataset LIMIT 1'
        )
        with database.backend.engine.connect() as conn:
            result = conn.execute(query, {"dataset": dataset_name})
            return result.first() is not None

    # -------------------------------------------------------------------------
    def get_available_datasets(self) -> list[str]:
        """Get list of all unique dataset names in the database."""
        return database.backend.get_distinct_values("TEXT_DATASET", "dataset_name")

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
        query = sqlalchemy.text(
            'SELECT LENGTH(text) AS text_length '
            'FROM "TEXT_DATASET" WHERE "dataset_name" = :dataset'
        )
        fetch_size = self.streaming_batch_size

        def generator() -> Iterator[int]:
            with database.backend.engine.connect().execution_options(
                stream_results=True
            ) as conn:
                result = conn.execute(query, {"dataset": dataset_name})
                while True:
                    rows = result.fetchmany(fetch_size)
                    if not rows:
                        break
                    for row in rows:
                        if hasattr(row, "_mapping"):
                            length_value = row._mapping.get("text_length")
                        else:
                            length_value = row[0]
                        if length_value is None:
                            continue
                        yield int(length_value)

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
        """
        Compute word-level statistics for all documents in a dataset.
        Uses streaming to avoid loading entire dataset into memory.

        Args:
            dataset_name: Name of the dataset to analyze.

        Returns:
            Dictionary with analyzed_count and statistics summary.
        """
        from TKBEN.server.database.schema import TextDatasetStatistics

        # Query to stream documents from TEXT_DATASET
        query = sqlalchemy.text(
            'SELECT "dataset_name", "text" FROM "TEXT_DATASET" '
            'WHERE "dataset_name" = :dataset'
        )

        count_query = sqlalchemy.text(
            'SELECT COUNT(*) FROM "TEXT_DATASET" WHERE "dataset_name" = :dataset'
        )

        with database.backend.engine.connect() as conn:
            count_result = conn.execute(count_query, {"dataset": dataset_name})
            count_row = count_result.first()
        total_documents = count_row[0] if count_row else 0
        if progress_callback:
            progress_callback(5.0)

        batch_size = self.streaming_batch_size
        analyzed_count = 0
        last_logged = 0

        # Delete any existing statistics for this dataset
        database.delete_by_key(
            TextDatasetStatistics.__tablename__,
            "dataset_name",
            dataset_name,
        )

        logger.info("Starting analysis for dataset: %s", dataset_name)

        # Collect batches to insert AFTER closing the read connection
        # This avoids SQLite "database is locked" errors from concurrent connections
        pending_batches: list[list[dict[str, Any]]] = []
        current_batch: list[dict[str, Any]] = []

        with database.backend.engine.connect().execution_options(
            stream_results=True
        ) as conn:
            result = conn.execute(query, {"dataset": dataset_name})

            while True:
                rows = result.fetchmany(batch_size)
                if not rows:
                    break

                for row in rows:
                    if should_stop and should_stop():
                        return {}
                    if hasattr(row, "_mapping"):
                        text = row._mapping.get("text", "")
                    else:
                        text = row[1] if len(row) > 1 else ""

                    if not text or not isinstance(text, str):
                        continue

                    # Compute word-level statistics
                    words = text.split()
                    words_count = len(words)

                    if words_count > 0:
                        word_lengths = [len(w) for w in words]
                        avg_word_length = sum(word_lengths) / len(word_lengths)
                        # Standard deviation
                        variance = sum((l - avg_word_length) ** 2 for l in word_lengths) / len(word_lengths)
                        std_word_length = variance ** 0.5
                    else:
                        avg_word_length = 0.0
                        std_word_length = 0.0

                    current_batch.append({
                        "dataset_name": dataset_name,
                        "text": text,
                        "words_count": words_count,
                        "AVG_words_length": avg_word_length,
                        "STD_words_length": std_word_length,
                    })

                    if len(current_batch) >= batch_size:
                        # Queue batch for insertion after connection closes
                        pending_batches.append(current_batch)
                        current_batch = []

            # Queue remaining items
            if current_batch:
                pending_batches.append(current_batch)

        # Now that the read connection is closed, insert all pending batches
        for batch in pending_batches:
            df = pd.DataFrame(batch)
            database.insert_dataframe(df, TextDatasetStatistics.__tablename__)
            analyzed_count += len(batch)
            if analyzed_count - last_logged >= self.log_interval:
                logger.info("Analyzed %d documents so far...", analyzed_count)
                last_logged = analyzed_count
            if progress_callback and total_documents > 0:
                progress_value = (analyzed_count / total_documents) * 100.0
                progress_callback(progress_value)

        logger.info("Completed analysis: %d documents analyzed", analyzed_count)

        # Get summary statistics
        statistics = self.get_analysis_summary(dataset_name)

        return {
            "dataset_name": dataset_name,
            "analyzed_count": analyzed_count,
            "statistics": statistics,
        }

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
        query = sqlalchemy.text(
            'SELECT 1 FROM "TEXT_DATASET_STATISTICS" '
            'WHERE "dataset_name" = :dataset LIMIT 1'
        )

        with database.backend.engine.connect() as conn:
            result = conn.execute(query, {"dataset": dataset_name})
            return result.first() is not None

