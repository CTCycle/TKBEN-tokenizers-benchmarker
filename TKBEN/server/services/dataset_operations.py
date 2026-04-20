from __future__ import annotations

import io
import math
import os
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict

from TKBEN.server.repositories.database.backend import database
from TKBEN.server.repositories.schemas.models import DatasetDocument
from TKBEN.server.common.constants import DATASETS_PATH
from TKBEN.server.common.utils.logger import logger
from TKBEN.server.common.utils.security import normalize_upload_stem
from TKBEN.server.services.metrics.catalog import default_selected_metric_keys
from TKBEN.server.services.metrics.engine import DatasetMetricsEngine


###############################################################################
class DatasetServiceOperationsMixin:
    def persist_dataset(
        self,
        dataset: Dataset | DatasetDict,
        dataset_name: str,
        text_column: str,
        stats: Any,
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
        length_counts: dict[int, int] = {}
        total_documents = stats.document_count if stats.document_count > 0 else 1

        for text in self._iterate_texts(dataset, text_column, remove_invalid):
            if should_stop and should_stop():
                return self.histogram_from_counts(stats, length_counts), saved_count
            text_length = len(text)
            length_counts[text_length] = length_counts.get(text_length, 0) + 1
            batch.append({"dataset_id": dataset_id, "text": text})

            if len(batch) >= batch_size:
                df = pd.DataFrame(batch)
                database.insert_dataframe(df, DatasetDocument.__tablename__)
                saved_count += len(batch)
                if saved_count - last_logged >= self.log_interval:
                    logger.info("Saved %d documents so far...", saved_count)
                    last_logged = saved_count
                if progress_callback:
                    progress_value = (
                        progress_base + (saved_count / total_documents) * progress_span
                    )
                    progress_callback(progress_value)
                batch.clear()

        if batch:
            df = pd.DataFrame(batch)
            database.insert_dataframe(df, DatasetDocument.__tablename__)
            saved_count += len(batch)
            if progress_callback:
                progress_value = (
                    progress_base + (saved_count / total_documents) * progress_span
                )
                progress_callback(progress_value)

        logger.info("Completed saving %d documents to database", saved_count)
        if progress_callback and stats.document_count == 0:
            progress_callback(progress_base + progress_span)
        return self.histogram_from_counts(stats, length_counts), saved_count

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

        max_attempts = max(1, int(self.download_retry_attempts))
        dataset: Dataset | DatasetDict | None = None
        last_exc: Exception | None = None
        last_category = "unknown"

        for attempt in range(1, max_attempts + 1):
            try:
                dataset = self.load_dataset_with_progress(
                    hf_dataset_id=target.hf_dataset_id,
                    hf_config=target.hf_config,
                    cache_path=cache_path,
                    hf_access_token=hf_access_token,
                    split=target.split,
                    progress_callback=progress_callback,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                last_category = self.classify_download_exception(exc)
                should_retry = self.should_retry_download(
                    category=last_category,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                logger.warning(
                    "Dataset download attempt %d/%d failed (job=%s, category=%s): requested=%s, resolved=%s",
                    attempt,
                    max_attempts,
                    job_id if job_id else "n/a",
                    last_category,
                    requested_dataset_name,
                    resolved_dataset_name,
                    exc_info=True,
                )
                if not should_retry:
                    break
                delay_seconds = self.retry_delay_seconds(attempt)
                logger.info(
                    "Retrying dataset download for %s in %.2f seconds (attempt %d/%d)",
                    requested_dataset_name,
                    delay_seconds,
                    attempt + 1,
                    max_attempts,
                )
                if delay_seconds > 0.0:
                    time.sleep(delay_seconds)

        if dataset is None:
            failure_exc = last_exc if last_exc is not None else RuntimeError(
                "Dataset download failed without an exception payload."
            )
            raise RuntimeError(
                self.build_download_error_message(
                    category=last_category,
                    job_id=job_id,
                    requested_dataset_name=requested_dataset_name,
                    resolved_dataset_name=resolved_dataset_name,
                    has_access_token=bool(hf_access_token),
                )
            ) from failure_exc

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
        normalized_name = os.path.basename(filename.strip().replace("\\", "/"))
        safe_stem = normalize_upload_stem(normalized_name)
        dataset_name = f"custom/{safe_stem}"
        extension = os.path.splitext(normalized_name)[1].lower()

        logger.info(
            "Processing uploaded file: %s (type: %s)", normalized_name, extension
        )
        if self.is_dataset_in_database(dataset_name):
            logger.info(
                "Dataset %s already present in database. Reusing persisted texts.",
                dataset_name,
            )
            if progress_callback:
                progress_callback(100.0)
            return self.build_persisted_dataset_payload(dataset_name)

        if extension not in (".csv", ".xlsx", ".xls"):
            raise ValueError(
                f"Unsupported file type: {extension}. Use .csv, .xlsx, or .xls"
            )

        # Load into DataFrame based on file extension
        try:
            if progress_callback:
                progress_callback(5.0)
            file_buffer = io.BytesIO(file_content)
            if extension == ".csv":
                df = pd.read_csv(file_buffer)
            else:
                df = pd.read_excel(file_buffer)
        except (
            OSError,
            UnicodeDecodeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            ValueError,
        ) as exc:
            logger.exception("Failed to read uploaded file %s", filename)
            raise ValueError(
                "Could not read the uploaded file. Check the file format and try again."
            ) from exc

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

        stats = self.collect_length_statistics(
            partial(self._dataframe_length_stream, df, text_column, remove_invalid)
        )
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
        length_counts: dict[int, int] = {}
        dataset_id = self.dataset_serializer.ensure_dataset_id(dataset_name)
        cancelled = False

        for text in self._iterate_dataframe_texts(df, text_column, remove_invalid):
            if self.stop_requested(should_stop):
                cancelled = True
                break
            text_length = len(text)
            length_counts[text_length] = length_counts.get(text_length, 0) + 1
            batch.append({"dataset_id": dataset_id, "text": text})

            if len(batch) >= batch_size:
                batch_df = pd.DataFrame(batch)
                database.insert_dataframe(batch_df, DatasetDocument.__tablename__)
                saved_count += len(batch)
                if saved_count - last_logged >= self.log_interval:
                    logger.info("Saved %d documents so far...", saved_count)
                    last_logged = saved_count
                if progress_callback:
                    progress_value = (
                        15.0 + (saved_count / max(stats.document_count, 1)) * 85.0
                    )
                    progress_callback(progress_value)
                batch.clear()

        if batch:
            batch_df = pd.DataFrame(batch)
            database.insert_dataframe(batch_df, DatasetDocument.__tablename__)
            saved_count += len(batch)
            if progress_callback:
                progress_value = (
                    15.0 + (saved_count / max(stats.document_count, 1)) * 85.0
                )
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
            "histogram": self.histogram_from_counts(stats, length_counts),
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
        parameter_overrides = (
            metric_parameters if isinstance(metric_parameters, dict) else {}
        )
        has_custom_request = bool(
            session_name
            or selected_metric_keys
            or sampling_config
            or filter_config
            or parameter_overrides
        )

        cached_report = self.dataset_serializer.load_latest_analysis_report(
            dataset_name
        )
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
            if isinstance(sample_fraction, (int, float))
            and 0 < float(sample_fraction) < 1
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
            expected_total = max(
                1, int(math.ceil(expected_total * normalized_fraction))
            )

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
                self.dataset_serializer.complete_analysis_session(
                    session_id, status="cancelled"
                )
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
                    self.dataset_serializer.save_metric_values_batch(
                        session_id, per_doc_buffer
                    )
                    persisted += len(per_doc_buffer)
                    per_doc_buffer.clear()

            if per_doc_buffer and len(per_doc_buffer) >= self.streaming_batch_size:
                self.dataset_serializer.save_metric_values_batch(
                    session_id, per_doc_buffer
                )
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
        self.dataset_serializer.complete_analysis_session(
            session_id, status="completed"
        )

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
            "mean_avg_word_length": float(
                aggregate.get("words.length_mean", 0.0) or 0.0
            ),
            "mean_std_word_length": float(
                aggregate.get("words.length_std", 0.0) or 0.0
            ),
        }

    # -------------------------------------------------------------------------
    def is_dataset_analyzed(self, dataset_name: str) -> bool:
        """Check if a dataset has been analyzed."""
        report = self.dataset_serializer.load_latest_analysis_report(dataset_name)
        return report is not None

    # -------------------------------------------------------------------------
    def remove_dataset(self, dataset_name: str) -> None:
        self.dataset_serializer.delete_dataset(dataset_name)
