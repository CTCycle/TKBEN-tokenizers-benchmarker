from __future__ import annotations

import math
import os
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from typing import Any

import pandas as pd
import sqlalchemy
from datasets import Dataset, DatasetDict, load_dataset

from TKBEN_webapp.server.database.database import database
from TKBEN_webapp.server.database.schema import TextDataset
from TKBEN_webapp.server.utils.constants import DATASETS_PATH
from TKBEN_webapp.server.utils.logger import logger


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
    HISTOGRAM_BINS = 20
    # Batch size for streaming operations (in-memory chunk size)
    STREAMING_BATCH_SIZE = 10000

    def __init__(self, hf_access_token: str | None = None) -> None:
        self.hf_access_token = hf_access_token

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
    def dataset_in_database(self, dataset_name: str) -> bool:
        query = sqlalchemy.text(
            'SELECT 1 FROM "TEXT_DATASET" WHERE "dataset_name" = :dataset LIMIT 1'
        )
        with database.backend.engine.connect() as conn:
            result = conn.execute(query, {"dataset": dataset_name})
            return result.first() is not None

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
        fetch_size = self.STREAMING_BATCH_SIZE

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
        builder = HistogramBuilder(stats, self.HISTOGRAM_BINS)
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
    ) -> tuple[dict[str, Any], int]:
        batch_size = self.STREAMING_BATCH_SIZE
        batch: list[dict[str, str]] = []
        saved_count = 0
        histogram_builder = HistogramBuilder(stats, self.HISTOGRAM_BINS)

        database.delete_by_key(
            TextDataset.__tablename__,
            "dataset_name",
            dataset_name,
        )

        for text in self._iterate_texts(dataset, text_column, remove_invalid):
            histogram_builder.add(len(text))
            batch.append({"dataset_name": dataset_name, "text": text})

            if len(batch) >= batch_size:
                df = pd.DataFrame(batch)
                database.insert_dataframe(df, TextDataset.__tablename__)
                saved_count += len(batch)
                logger.info("Saved %d documents so far...", saved_count)
                batch.clear()

        if batch:
            df = pd.DataFrame(batch)
            database.insert_dataframe(df, TextDataset.__tablename__)
            saved_count += len(batch)

        logger.info("Completed saving %d documents to database", saved_count)
        return histogram_builder.build(), saved_count

    # -------------------------------------------------------------------------
    def download_and_persist(
        self,
        corpus: str,
        config: str | None = None,
        remove_invalid: bool = True,
    ) -> dict[str, Any]:
        dataset_name = self.get_dataset_name(corpus, config)
        cache_path = self.get_cache_path(corpus, config)

        os.makedirs(cache_path, exist_ok=True)

        if self.dataset_in_database(dataset_name):
            logger.info(
                "Dataset %s already present in database. Reusing persisted texts.",
                dataset_name,
            )
            length_stream = self.database_length_stream(dataset_name)
            stats = self.collect_length_statistics(length_stream)
            histogram = self.histogram_from_stream(length_stream, stats)
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

        # Persist to database with histogram computation (second pass)
        batch_size = self.STREAMING_BATCH_SIZE
        batch: list[dict[str, str]] = []
        saved_count = 0
        histogram_builder = HistogramBuilder(stats, self.HISTOGRAM_BINS)

        # Delete any existing entries with this dataset name
        database.delete_by_key(
            TextDataset.__tablename__,
            "dataset_name",
            dataset_name,
        )

        for text in iterate_df_texts():
            histogram_builder.add(len(text))
            batch.append({"dataset_name": dataset_name, "text": text})

            if len(batch) >= batch_size:
                batch_df = pd.DataFrame(batch)
                database.insert_dataframe(batch_df, TextDataset.__tablename__)
                saved_count += len(batch)
                logger.info("Saved %d documents so far...", saved_count)
                batch.clear()

        if batch:
            batch_df = pd.DataFrame(batch)
            database.insert_dataframe(batch_df, TextDataset.__tablename__)
            saved_count += len(batch)

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
    # Legacy methods kept for backwards compatibility
