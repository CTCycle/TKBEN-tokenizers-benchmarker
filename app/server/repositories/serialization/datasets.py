from __future__ import annotations

import json
import math
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any, cast

import pandas as pd
from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from server.common.constants import DATASET_REPORT_VERSION
from server.repositories.database.seeding import seed_metric_types
from server.repositories.queries.data import DataRepositoryQueries
from server.repositories.schemas.models import (
    AnalysisSession,
    Dataset,
    DatasetDocument,
    HistogramArtifact,
    MetricType,
    MetricValue,
)

K_ERROR = "k error"

###############################################################################
def _parse_timestamp(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(cast(Any, value), utc=True, errors="coerce")
    return parsed if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed) else None

###############################################################################
class DatasetSerializer:

    # -------------------------------------------------------------------------
    def __init__(self, queries: DataRepositoryQueries | None = None) -> None:
        self.queries = queries or DataRepositoryQueries()
        self.dataset_dimension_table = Dataset.__tablename__
        self.metric_value_table = MetricValue.__tablename__
        self.histogram_table = HistogramArtifact.__tablename__

    # -------------------------------------------------------------------------
    def _session(self) -> Session:
        return Session(bind=self.queries.engine)

    # -------------------------------------------------------------------------
    def parse_json(self, value: Any, default: Any | None = None) -> Any:
        if default is None:
            default = {}
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return default
        if isinstance(value, (dict, list)):
            return value
        return default

    # -------------------------------------------------------------------------
    def serialize_series(self, col: Any) -> Any:
        if isinstance(col, list):
            return " ".join(map(str, col))
        if isinstance(col, str):
            return [int(value) for value in col.split() if value.strip()]
        return []

    # -------------------------------------------------------------------------
    def serialize_json_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df_copy = df.copy()
        for col in df_copy.columns:
            first_valid = (
                df_copy[col].dropna().iloc[0]
                if not df_copy[col].dropna().empty
                else None
            )
            if isinstance(first_valid, (list, dict)):
                df_copy[col] = df_copy[col].apply(
                    lambda value: (
                        json.dumps(value) if isinstance(value, (list, dict)) else value
                    )
                )
        return df_copy

    # -------------------------------------------------------------------------
    def list_dataset_previews(
        self,
        search: str | None = None,
        source: str = "all",
        document_count_operator: str = "at_least",
        document_count: int | None = None,
    ) -> list[dict[str, Any]]:
        conditions = [Dataset.status == "ready"]
        normalized_search = (search or "").strip().lower()
        if normalized_search:
            conditions.append(func.lower(Dataset.name).contains(normalized_search))
        if source == "custom":
            conditions.append(Dataset.name.like("custom/%"))
        elif source == "public":
            conditions.append(~Dataset.name.like("custom/%"))
        if document_count is not None:
            conditions.append(
                Dataset.document_count <= document_count
                if document_count_operator == "at_most"
                else Dataset.document_count >= document_count
            )

        stmt = (
            select(
                Dataset.name.label("dataset_name"),
                Dataset.document_count,
            )
            .where(*conditions)
            .order_by(Dataset.name.asc())
        )
        with self._session() as session:
            rows = session.execute(stmt).all()
        return [
            {
                "dataset_name": str(dataset_name),
                "document_count": int(document_count or 0),
            }
            for dataset_name, document_count in rows
            if dataset_name is not None
        ]

    # -------------------------------------------------------------------------
    def list_dataset_names(self) -> list[str]:
        stmt = select(Dataset.name).where(Dataset.status == "ready").order_by(Dataset.name.asc())
        with self._session() as session:
            return [str(name) for name in session.execute(stmt).scalars()]

    # -------------------------------------------------------------------------
    def get_dataset_id(self, dataset_name: str) -> int | None:
        stmt = select(Dataset.id).where(Dataset.name == dataset_name, Dataset.status == "ready").limit(1)
        with self._session() as session:
            dataset_id = session.execute(stmt).scalar_one_or_none()
        return int(dataset_id) if dataset_id is not None else None

    # -------------------------------------------------------------------------
    def ensure_dataset_id(self, dataset_name: str) -> int:
        existing = self.begin_dataset_import(dataset_name)
        return existing

    # -------------------------------------------------------------------------
    def begin_dataset_import(self, dataset_name: str) -> int:
        now = datetime.now(timezone.utc)
        with self._session() as session:
            dataset_row = session.execute(
                select(Dataset.id, Dataset.status).where(Dataset.name == dataset_name).limit(1)
            ).first()
            dataset_id = dataset_row[0] if dataset_row is not None else None
            if dataset_row is not None:
                dataset_status = dataset_row[1]
                if dataset_status == "ready":
                    assert dataset_id is not None
                    session.execute(delete(Dataset).where(Dataset.id == int(dataset_id)))
                    session.flush()
                    dataset_id = None
            if dataset_id is None:
                session.add(Dataset(name=dataset_name, status="loading", created_at=now, updated_at=now))
                try:
                    session.commit()
                except IntegrityError:
                    session.rollback()
                dataset_id = session.execute(
                    select(Dataset.id).where(Dataset.name == dataset_name).limit(1)
                ).scalar_one_or_none()
        if dataset_id is None:
            raise ValueError(f"Failed to resolve dataset id for '{dataset_name}'")
        return int(dataset_id)

    # -------------------------------------------------------------------------
    def finalize_dataset_import(self, dataset_id: int, document_count: int) -> None:
        now = datetime.now(timezone.utc)
        with self._session() as session:
            result = session.execute(update(Dataset).where(Dataset.id == int(dataset_id), Dataset.status == "loading").values(status="ready", document_count=max(0, int(document_count)), ready_at=now, updated_at=now))
            if cast(Any, result).rowcount != 1:
                raise ValueError(f"Dataset import {dataset_id} is not loading")
            session.commit()

    # -------------------------------------------------------------------------
    def delete_incomplete_dataset(self, dataset_id: int) -> None:
        with self._session() as session:
            session.execute(delete(Dataset).where(Dataset.id == int(dataset_id), Dataset.status == "loading"))
            session.commit()

    # -------------------------------------------------------------------------
    def save_document_batch(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        self.queries.insert_records(DatasetDocument.__tablename__, batch)

    # -------------------------------------------------------------------------
    def dataset_exists(self, dataset_name: str) -> bool:
        stmt = select(Dataset.id).where(Dataset.name == dataset_name, Dataset.status == "ready").limit(1)
        with self._session() as session:
            return session.execute(stmt).first() is not None

    # -------------------------------------------------------------------------
    def count_dataset_documents(self, dataset_name: str) -> int:
        stmt = (
            select(func.count(DatasetDocument.id))
            .join(Dataset, Dataset.id == DatasetDocument.dataset_id)
            .where(Dataset.name == dataset_name, Dataset.status == "ready")
        )
        with self._session() as session:
            value = session.execute(stmt).scalar_one_or_none() or 0
        return int(value)

    # -------------------------------------------------------------------------
    def iterate_dataset_batches(
        self,
        dataset_name: str,
        batch_size: int,
    ) -> Iterator[list[str]]:
        last_seen_id = 0
        while True:
            stmt = (
                select(DatasetDocument.id, DatasetDocument.text)
                .join(Dataset, Dataset.id == DatasetDocument.dataset_id)
                .where(
                    Dataset.name == dataset_name, Dataset.status == "ready",
                    DatasetDocument.id > int(last_seen_id),
                )
                .order_by(DatasetDocument.id.asc())
                .limit(int(batch_size))
            )
            with self._session() as session:
                rows = session.execute(stmt).all()

            if not rows:
                break

            texts: list[str] = []
            for row_id, text_value in rows:
                if row_id is None:
                    continue
                last_seen_id = int(row_id)
                if text_value is not None:
                    texts.append(str(text_value))

            if texts:
                yield texts

    # -------------------------------------------------------------------------
    def iterate_dataset_rows(
        self,
        dataset_name: str,
        batch_size: int,
        min_length: int | None = None,
        max_length: int | None = None,
        exclude_empty: bool = False,
    ) -> Iterator[list[dict[str, Any]]]:
        last_seen_id = 0
        conditions = [Dataset.name == dataset_name]
        if isinstance(min_length, int):
            conditions.append(func.length(DatasetDocument.text) >= int(min_length))
        if isinstance(max_length, int):
            conditions.append(func.length(DatasetDocument.text) <= int(max_length))
        if exclude_empty:
            conditions.append(func.length(func.trim(DatasetDocument.text)) > 0)
        while True:
            stmt = (
                select(DatasetDocument.id, DatasetDocument.text)
                .join(Dataset, Dataset.id == DatasetDocument.dataset_id)
                .where(and_(*conditions), Dataset.status == "ready", DatasetDocument.id > int(last_seen_id))
                .order_by(DatasetDocument.id.asc())
                .limit(int(batch_size))
            )
            with self._session() as session:
                rows = session.execute(stmt).all()

            if not rows:
                break

            batch: list[dict[str, Any]] = []
            for row_id, text_value in rows:
                if row_id is None or text_value is None:
                    continue
                last_seen_id = int(row_id)
                batch.append({"id": int(row_id), "text": str(text_value)})

            if batch:
                yield batch

    # -------------------------------------------------------------------------
    def iterate_dataset_rows_for_benchmarks(
        self,
        dataset_name: str,
        batch_size: int,
    ) -> Iterator[tuple[int, str]]:
        for batch in self.iterate_dataset_rows(
            dataset_name=dataset_name,
            batch_size=batch_size,
        ):
            for item in batch:
                row_id = item.get("id")
                text = item.get("text")
                if isinstance(row_id, int) and isinstance(text, str):
                    yield row_id, text

    # -------------------------------------------------------------------------
    def delete_dataset(self, dataset_name: str) -> None:
        stmt = delete(Dataset).where(Dataset.name == dataset_name)
        with self._session() as session:
            session.execute(stmt)
            session.commit()

    # -------------------------------------------------------------------------
    def ensure_metric_types_seeded(self, metric_catalog: list[dict[str, Any]]) -> None:
        seed_metric_types(self.queries.engine, metric_catalog)

    # -------------------------------------------------------------------------
    def get_metric_type_map(self) -> dict[str, int]:
        stmt = select(MetricType.id, MetricType.key)
        with self._session() as session:
            rows = session.execute(stmt).all()
        return {str(metric_key): int(metric_id) for metric_id, metric_key in rows}

    # -------------------------------------------------------------------------
    def create_analysis_session(
        self,
        dataset_name: str,
        session_name: str | None,
        selected_metric_keys: list[str],
        parameters: dict[str, Any],
        report_version: int = DATASET_REPORT_VERSION,
    ) -> int:
        dataset_id = self.get_dataset_id(dataset_name)
        if dataset_id is None:
            raise ValueError(f"Dataset '{dataset_name}' is not ready")
        created_at = pd.Timestamp.utcnow().to_pydatetime()
        session_row = AnalysisSession(
            dataset_id=int(dataset_id),
            session_name=session_name,
            status="running",
            report_version=int(report_version),
            created_at=created_at,
            completed_at=None,
            parameters=parameters,
            selected_metric_keys=selected_metric_keys,
        )
        with self._session() as session:
            session.add(session_row)
            session.commit()
            session.refresh(session_row)
        if session_row.id is None:
            raise ValueError("Failed to create analysis session.")
        return int(session_row.id)

    # -------------------------------------------------------------------------
    def complete_analysis_session(
        self, session_id: int, status: str = "completed"
    ) -> None:
        stmt = (
            update(AnalysisSession)
            .where(AnalysisSession.id == int(session_id))
            .values(
                status=str(status),
                completed_at=pd.Timestamp.utcnow().to_pydatetime(),
            )
        )
        with self._session() as session:
            session.execute(stmt)
            session.commit()

    # -------------------------------------------------------------------------
    def save_metric_values_batch(
        self, session_id: int, batch: list[dict[str, Any]]
    ) -> None:
        if not batch:
            return
        with self._session() as session:
            owning_dataset_id = session.execute(select(AnalysisSession.dataset_id).where(AnalysisSession.id == int(session_id))).scalar_one_or_none()
        if owning_dataset_id is None:
            raise ValueError(f"Analysis session {session_id} does not exist")
        created_at = datetime.now(timezone.utc)
        metric_type_map = self.get_metric_type_map()
        rows: list[dict[str, Any]] = []
        for item in batch:
            metric_key = str(item.get("metric_key") or "")
            metric_type_id = metric_type_map.get(metric_key)
            if metric_type_id is None:
                continue
            raw_numeric = item.get("numeric_value")
            numeric_value = None
            if raw_numeric is not None:
                numeric_candidate = float(raw_numeric)
                if math.isfinite(numeric_candidate):
                    numeric_value = numeric_candidate
            raw_text = item.get("text_value")
            text_value = None
            if raw_text is not None and not (isinstance(raw_text, float) and math.isnan(raw_text)):
                text_value = str(raw_text)
            json_value = item.get("json_value")
            if json_value is not None:
                numeric_value = None
                text_value = None
            elif text_value is not None:
                numeric_value = None
            value_count = sum(value is not None for value in (numeric_value, text_value, json_value))
            if value_count != 1:
                raise ValueError(
                    f"Metric '{metric_key}' must contain exactly one value representation; "
                    f"received numeric={numeric_value!r}, text={text_value!r}, json={json_value!r}"
                )
            rows.append(
                {
                    "session_id": int(session_id),
                    "dataset_id": int(owning_dataset_id),
                    "metric_type_id": int(metric_type_id),
                    "document_id": (
                        int(item["document_id"])
                        if item.get("document_id") is not None
                        else None
                    ),
                    "numeric_value": numeric_value,
                    "text_value": text_value,
                    "json_value": json_value,
                    "created_at": created_at,
                }
            )
        if not rows:
            return
        chunk_size = 100
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            self.queries.insert_records(self.metric_value_table, chunk, ignore_duplicates=False)

    # -------------------------------------------------------------------------
    def save_histogram_artifact(
        self,
        session_id: int,
        metric_key: str,
        histogram: dict[str, Any],
    ) -> None:
        metric_type_id = self.get_metric_type_map().get(metric_key)
        if metric_type_id is None:
            return
        row = {
            "session_id": int(session_id),
            "metric_type_id": int(metric_type_id),
            "bins": histogram.get("bins", []),
            "bin_edges": histogram.get("bin_edges", []),
            "counts": histogram.get("counts", []),
            "min_value": float(histogram.get("min_length", 0.0) or 0.0),
            "max_value": float(histogram.get("max_length", 0.0) or 0.0),
            "mean_value": float(histogram.get("mean_length", 0.0) or 0.0),
            "median_value": float(histogram.get("median_length", 0.0) or 0.0),
            "created_at": datetime.now(timezone.utc),
        }
        self.queries.upsert_records(self.histogram_table, [row], ["session_id", "metric_type_id"])

    # -------------------------------------------------------------------------
    def _load_metric_rows_for_session(self, session_id: int) -> list[dict[str, Any]]:
        stmt = (
            select(
                MetricValue.document_id,
                MetricType.key,
                MetricValue.numeric_value,
                MetricValue.text_value,
                MetricValue.json_value,
            )
            .join(MetricType, MetricType.id == MetricValue.metric_type_id)
            .where(MetricValue.session_id == int(session_id))
            .order_by(MetricValue.id.asc())
        )
        with self._session() as session:
            rows = session.execute(stmt).all()
        return [
            {
                "document_id": document_id,
                "key": str(metric_key or ""),
                "numeric_value": numeric_value,
                "text_value": text_value,
                "json_value": json_value,
            }
            for document_id, metric_key, numeric_value, text_value, json_value in rows
        ]

    # -------------------------------------------------------------------------
    def _load_histogram_rows_for_session(self, session_id: int) -> dict[str, Any]:
        stmt = (
            select(
                MetricType.key,
                HistogramArtifact.bins,
                HistogramArtifact.counts,
                HistogramArtifact.bin_edges,
                HistogramArtifact.min_value,
                HistogramArtifact.max_value,
                HistogramArtifact.mean_value,
                HistogramArtifact.median_value,
            )
            .join(MetricType, MetricType.id == HistogramArtifact.metric_type_id)
            .where(HistogramArtifact.session_id == int(session_id))
        )
        with self._session() as session:
            rows = session.execute(stmt).all()
        result: dict[str, Any] = {}
        for (
            key,
            bins_value,
            counts_value,
            edges_value,
            min_value,
            max_value,
            mean_value,
            median_value,
        ) in rows:
            key = str(key or "")
            bins = self.parse_json(bins_value, default=[])
            counts = self.parse_json(counts_value, default=[])
            bin_edges = self.parse_json(edges_value, default=[])
            min_value = float(min_value or 0.0)
            max_value = float(max_value or 0.0)
            mean_value = float(mean_value or 0.0)
            median_value = float(median_value or 0.0)
            result[key] = {
                "bins": bins,
                "counts": counts,
                "bin_edges": bin_edges,
                "min_length": int(min_value),
                "max_length": int(max_value),
                "mean_length": mean_value,
                "median_length": median_value,
            }
        return result

    # -------------------------------------------------------------------------
    def _build_session_report_response(
        self, session_row: dict[str, Any]
    ) -> dict[str, Any]:
        session_id = int(session_row.get("id") or 0)
        metric_rows = self._load_metric_rows_for_session(session_id)
        histogram_rows = self._load_histogram_rows_for_session(session_id)

        aggregate_statistics: dict[str, Any] = {}
        per_document: dict[int, dict[str, Any]] = {}
        for row in metric_rows:
            key = str(row.get("key") or "")
            numeric_value = row.get("numeric_value")
            if isinstance(numeric_value, float) and pd.isna(numeric_value):
                numeric_value = None

            text_value = row.get("text_value")
            if isinstance(text_value, float) and pd.isna(text_value):
                text_value = None

            json_value = row.get("json_value")
            if isinstance(json_value, float) and pd.isna(json_value):
                json_value = None

            value: Any = numeric_value
            if value is None and text_value is not None:
                value = text_value
            if value is None and json_value is not None:
                value = self.parse_json(json_value, default={})
            document_id = row.get("document_id")
            if document_id is None:
                aggregate_statistics[key] = value
                continue
            doc_key = int(document_id)
            if doc_key not in per_document:
                per_document[doc_key] = {"document_id": doc_key}
            per_document[doc_key][key] = value

        per_document_stats = {
            "document_ids": [],
            "document_lengths": [],
            "word_counts": [],
            "avg_word_lengths": [],
            "std_word_lengths": [],
        }
        for doc_id in sorted(per_document.keys()):
            payload = per_document[doc_id]
            per_document_stats["document_ids"].append(doc_id)
            per_document_stats["document_lengths"].append(
                int(payload.get("doc.length_chars", 0) or 0)
            )
            per_document_stats["word_counts"].append(
                int(payload.get("doc.word_count", 0) or 0)
            )
            per_document_stats["avg_word_lengths"].append(
                float(payload.get("doc.avg_word_length", 0.0) or 0.0)
            )
            per_document_stats["std_word_lengths"].append(
                float(payload.get("doc.std_word_length", 0.0) or 0.0)
            )

        document_histogram = histogram_rows.get("hist.document_length", {})
        word_histogram = histogram_rows.get("hist.word_length", {})
        created_at = _parse_timestamp(session_row.get("created_at"))
        created_at_iso = created_at.isoformat().replace("+00:00", "Z") if created_at else None

        return {
            "report_id": session_id,
            "report_version": int(session_row["report_version"]),
            "created_at": created_at_iso,
            "dataset_name": str(session_row.get("dataset_name") or ""),
            "session_name": session_row.get("session_name"),
            "selected_metric_keys": self.parse_json(
                session_row.get("selected_metric_keys"), default=[]
            ),
            "session_parameters": self.parse_json(
                session_row.get("parameters"), default={}
            ),
            "document_count": int(
                aggregate_statistics.get("corpus.document_count", 0) or 0
            ),
            "document_length_histogram": {
                "bins": list(document_histogram.get("bins", [])),
                "counts": list(document_histogram.get("counts", [])),
                "bin_edges": list(document_histogram.get("bin_edges", [])),
                "min_length": int(document_histogram.get("min_length", 0) or 0),
                "max_length": int(document_histogram.get("max_length", 0) or 0),
                "mean_length": float(document_histogram.get("mean_length", 0.0) or 0.0),
                "median_length": float(
                    document_histogram.get("median_length", 0.0) or 0.0
                ),
            },
            "word_length_histogram": {
                "bins": list(word_histogram.get("bins", [])),
                "counts": list(word_histogram.get("counts", [])),
                "bin_edges": list(word_histogram.get("bin_edges", [])),
                "min_length": int(word_histogram.get("min_length", 0) or 0),
                "max_length": int(word_histogram.get("max_length", 0) or 0),
                "mean_length": float(word_histogram.get("mean_length", 0.0) or 0.0),
                "median_length": float(word_histogram.get("median_length", 0.0) or 0.0),
            },
            "min_document_length": int(document_histogram.get("min_length", 0) or 0),
            "max_document_length": int(document_histogram.get("max_length", 0) or 0),
            "most_common_words": self.parse_json(
                aggregate_statistics.get("words.most_common"), default=[]
            ),
            "least_common_words": self.parse_json(
                aggregate_statistics.get("words.least_common"), default=[]
            ),
            "longest_words": self.parse_json(
                aggregate_statistics.get("words.longest"), default=[]
            ),
            "shortest_words": self.parse_json(
                aggregate_statistics.get("words.shortest"), default=[]
            ),
            "word_cloud_terms": self.parse_json(
                aggregate_statistics.get("words.word_cloud"), default=[]
            ),
            "aggregate_statistics": aggregate_statistics,
            "per_document_stats": per_document_stats,
        }

    # -------------------------------------------------------------------------
    def load_latest_analysis_report(self, dataset_name: str) -> dict[str, Any] | None:
        stmt = (
            select(AnalysisSession, Dataset.name.label("dataset_name"))
            .join(Dataset, Dataset.id == AnalysisSession.dataset_id)
            .where(
                Dataset.name == dataset_name,
                AnalysisSession.status == "completed",
                AnalysisSession.report_version == DATASET_REPORT_VERSION,
            )
            .order_by(AnalysisSession.id.desc())
            .limit(1)
        )
        with self._session() as session:
            row = session.execute(stmt).first()
        if row is None or row[0] is None:
            return None
        session_row, dataset_name_value = row
        mapped = {
            "id": session_row.id,
            "dataset_id": session_row.dataset_id,
            "session_name": session_row.session_name,
            "status": session_row.status,
            "report_version": session_row.report_version,
            "created_at": session_row.created_at,
            "completed_at": session_row.completed_at,
            "parameters": session_row.parameters,
            "selected_metric_keys": session_row.selected_metric_keys,
            "dataset_name": dataset_name_value,
        }
        return self._build_session_report_response(mapped)

    # -------------------------------------------------------------------------
    def load_analysis_report_by_session_id(
        self, session_id: int
    ) -> dict[str, Any] | None:
        stmt = (
            select(AnalysisSession, Dataset.name.label("dataset_name"))
            .join(Dataset, Dataset.id == AnalysisSession.dataset_id)
            .where(
                AnalysisSession.id == int(session_id),
                AnalysisSession.report_version == DATASET_REPORT_VERSION,
            )
            .limit(1)
        )
        with self._session() as session:
            row = session.execute(stmt).first()
        if row is None or row[0] is None:
            return None
        session_row, dataset_name_value = row
        mapped = {
            "id": session_row.id,
            "dataset_id": session_row.dataset_id,
            "session_name": session_row.session_name,
            "status": session_row.status,
            "report_version": session_row.report_version,
            "created_at": session_row.created_at,
            "completed_at": session_row.completed_at,
            "parameters": session_row.parameters,
            "selected_metric_keys": session_row.selected_metric_keys,
            "dataset_name": dataset_name_value,
        }
        return self._build_session_report_response(mapped)
