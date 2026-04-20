from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pandas as pd
from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from TKBEN.server.domain.benchmarks import BenchmarkReportSummary, BenchmarkRunResponse
from TKBEN.server.repositories.queries.data import DataRepositoryQueries
from TKBEN.server.repositories.benchmarks import BenchmarkRepository
from TKBEN.server.repositories.schemas.models import (
    AnalysisSession,
    BenchmarkReport,
    Dataset,
    DatasetDocument,
    HistogramArtifact,
    MetricType,
    MetricValue,
    Tokenizer,
    TokenizerReport,
    TokenizerVocabulary,
)

K_ERROR = "k error"


###############################################################################
class DatasetSerializer:
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
    def list_dataset_previews(self) -> list[dict[str, Any]]:
        stmt = (
            select(
                Dataset.name.label("dataset_name"),
                func.count(DatasetDocument.id).label("document_count"),
            )
            .outerjoin(DatasetDocument, DatasetDocument.dataset_id == Dataset.id)
            .group_by(Dataset.id, Dataset.name)
            .order_by(Dataset.name.asc())
        )
        with self._session() as session:
            rows = session.execute(stmt).all()
        return [
            {"dataset_name": str(dataset_name), "document_count": int(document_count or 0)}
            for dataset_name, document_count in rows
            if dataset_name is not None
        ]

    # -------------------------------------------------------------------------
    def list_dataset_names(self) -> list[str]:
        return self.queries.get_distinct_values(self.dataset_dimension_table, "name")

    # -------------------------------------------------------------------------
    def get_dataset_id(self, dataset_name: str) -> int | None:
        stmt = select(Dataset.id).where(Dataset.name == dataset_name).limit(1)
        with self._session() as session:
            dataset_id = session.execute(stmt).scalar_one_or_none()
        return int(dataset_id) if dataset_id is not None else None

    # -------------------------------------------------------------------------
    def ensure_dataset_id(self, dataset_name: str) -> int:
        with self._session() as session:
            dataset_id = session.execute(
                select(Dataset.id).where(Dataset.name == dataset_name).limit(1)
            ).scalar_one_or_none()
            if dataset_id is None:
                session.add(Dataset(name=dataset_name))
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
    def dataset_exists(self, dataset_name: str) -> bool:
        stmt = select(Dataset.id).where(Dataset.name == dataset_name).limit(1)
        with self._session() as session:
            return session.execute(stmt).first() is not None

    # -------------------------------------------------------------------------
    def count_dataset_documents(self, dataset_name: str) -> int:
        stmt = (
            select(func.count(DatasetDocument.id))
            .join(Dataset, Dataset.id == DatasetDocument.dataset_id)
            .where(Dataset.name == dataset_name)
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
        offset = 0
        while True:
            stmt = (
                select(DatasetDocument.text)
                .join(Dataset, Dataset.id == DatasetDocument.dataset_id)
                .where(Dataset.name == dataset_name)
                .order_by(DatasetDocument.id.asc())
                .limit(int(batch_size))
                .offset(int(offset))
            )
            with self._session() as session:
                rows = session.execute(stmt).all()

            if not rows:
                break

            texts = [str(text_value) for (text_value,) in rows if text_value is not None]

            if texts:
                yield texts
            offset += len(rows)

    # -------------------------------------------------------------------------
    def iterate_dataset_rows(
        self,
        dataset_name: str,
        batch_size: int,
        min_length: int | None = None,
        max_length: int | None = None,
        exclude_empty: bool = False,
    ) -> Iterator[list[dict[str, Any]]]:
        offset = 0
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
                .where(and_(*conditions))
                .order_by(DatasetDocument.id.asc())
                .limit(int(batch_size))
                .offset(int(offset))
            )
            with self._session() as session:
                rows = session.execute(stmt).all()

            if not rows:
                break

            batch: list[dict[str, Any]] = []
            for row_id, text_value in rows:
                if row_id is None or text_value is None:
                    continue
                batch.append({"id": int(row_id), "text": str(text_value)})

            if batch:
                yield batch
            offset += len(rows)

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
                if isinstance(row_id, int) and isinstance(text, str) and text:
                    yield row_id, text

    # -------------------------------------------------------------------------
    def delete_dataset(self, dataset_name: str) -> None:
        stmt = delete(Dataset).where(Dataset.name == dataset_name)
        with self._session() as session:
            session.execute(stmt)
            session.commit()

    def ensure_metric_types_seeded(self, metric_catalog: list[dict[str, Any]]) -> None:
        entries: list[dict[str, str]] = []
        for category in metric_catalog:
            category_key = str(category.get("category_key", "uncategorized"))
            metrics = category.get("metrics")
            if not isinstance(metrics, list):
                continue
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                key = metric.get("key")
                label = metric.get("label")
                if not key or not label:
                    continue
                entries.append(
                    {
                        "key": str(key),
                        "category": category_key,
                        "label": str(label),
                        "description": str(metric.get("description") or ""),
                        "scope": str(metric.get("scope") or "aggregate"),
                        "value_kind": str(metric.get("value_kind") or "number"),
                    }
                )
        if not entries:
            return
        metric_keys = [entry["key"] for entry in entries]
        with self._session() as session:
            existing_types = {
                metric_type.key: metric_type
                for metric_type in session.execute(
                    select(MetricType).where(MetricType.key.in_(metric_keys))
                ).scalars()
            }
            for entry in entries:
                metric_type = existing_types.get(entry["key"])
                if metric_type is None:
                    session.add(MetricType(**entry))
                    continue
                metric_type.category = entry["category"]
                metric_type.label = entry["label"]
                metric_type.description = entry["description"]
                metric_type.scope = entry["scope"]
                metric_type.value_kind = entry["value_kind"]
            session.commit()

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
        report_version: int = 2,
    ) -> int:
        dataset_id = self.ensure_dataset_id(dataset_name)
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
        metric_type_map = self.get_metric_type_map()
        rows: list[dict[str, Any]] = []
        for item in batch:
            metric_key = str(item.get("metric_key") or "")
            metric_type_id = metric_type_map.get(metric_key)
            if metric_type_id is None:
                continue
            rows.append(
                {
                    "session_id": int(session_id),
                    "metric_type_id": int(metric_type_id),
                    "document_id": (
                        int(item["document_id"])
                        if item.get("document_id") is not None
                        else None
                    ),
                    "numeric_value": (
                        float(item["numeric_value"])
                        if item.get("numeric_value") is not None
                        else None
                    ),
                    "text_value": (
                        str(item["text_value"])
                        if item.get("text_value") is not None
                        else None
                    ),
                    "json_value": item.get("json_value"),
                }
            )
        if not rows:
            return
        chunk_size = 100
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            df = pd.DataFrame(chunk, dtype=object)
            df = df.where(pd.notna(df), None)
            self.queries.insert_table(
                df,
                self.metric_value_table,
                ignore_duplicates=False,
            )

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
        }
        df = pd.DataFrame([row])
        self.queries.upsert_table(df, self.histogram_table)

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
        for key, bins_value, counts_value, edges_value, min_value, max_value, mean_value, median_value in rows:
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
        created_at = pd.to_datetime(
            session_row.get("created_at"), utc=True, errors="coerce"
        )
        created_at_iso = (
            created_at.isoformat().replace("+00:00", "Z")
            if not pd.isna(created_at)
            else None
        )

        return {
            "report_id": session_id,
            "report_version": int(session_row.get("report_version", 2) or 2),
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
            .where(AnalysisSession.id == int(session_id))
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


###############################################################################
class TokenizerReportSerializer:
    def __init__(self, queries: DataRepositoryQueries | None = None) -> None:
        self.queries = queries or DataRepositoryQueries()
        self.tokenizer_table = Tokenizer.__tablename__
        self.tokenizer_report_table = TokenizerReport.__tablename__
        self.tokenizer_vocabulary_table = TokenizerVocabulary.__tablename__

    # -------------------------------------------------------------------------
    def _session(self) -> Session:
        return Session(bind=self.queries.engine)

    # -------------------------------------------------------------------------
    def get_tokenizer_id(self, tokenizer_name: str) -> int | None:
        stmt = select(Tokenizer.id).where(Tokenizer.name == tokenizer_name).limit(1)
        with self._session() as session:
            tokenizer_id = session.execute(stmt).scalar_one_or_none()
        return int(tokenizer_id) if tokenizer_id is not None else None

    # -------------------------------------------------------------------------
    def ensure_tokenizer_id(self, tokenizer_name: str) -> int:
        with self._session() as session:
            tokenizer_id = session.execute(
                select(Tokenizer.id).where(Tokenizer.name == tokenizer_name).limit(1)
            ).scalar_one_or_none()
            if tokenizer_id is None:
                session.add(Tokenizer(name=tokenizer_name))
                try:
                    session.commit()
                except IntegrityError:
                    session.rollback()
                tokenizer_id = session.execute(
                    select(Tokenizer.id).where(Tokenizer.name == tokenizer_name).limit(1)
                ).scalar_one_or_none()
        if tokenizer_id is None:
            raise ValueError(f"Failed to resolve tokenizer id for '{tokenizer_name}'")
        return int(tokenizer_id)

    # -------------------------------------------------------------------------
    def save_tokenizer_report(self, report: dict[str, Any]) -> int:
        tokenizer_name = str(report.get("tokenizer_name") or "")
        tokenizer_id = self.ensure_tokenizer_id(tokenizer_name)
        global_stats = report.get("global_stats", {})
        metadata_payload = dict(global_stats) if isinstance(global_stats, dict) else {}
        if "huggingface_url" not in metadata_payload:
            metadata_payload["huggingface_url"] = report.get("huggingface_url")
        created_at = pd.to_datetime(report.get("created_at"), utc=True, errors="coerce")
        if pd.isna(created_at):
            created_at = pd.Timestamp.utcnow()
        report_row = TokenizerReport(
            tokenizer_id=int(tokenizer_id),
            report_version=int(report.get("report_version", 1) or 1),
            created_at=created_at.to_pydatetime(),
            metadata_json=metadata_payload,
            token_length_histogram=report.get("token_length_histogram", {}),
            description=report.get("description"),
        )
        with self._session() as session:
            session.add(report_row)
            session.commit()
            session.refresh(report_row)
        if report_row.id is None:
            raise ValueError("Failed to resolve saved tokenizer report id.")
        return int(report_row.id)

    # -------------------------------------------------------------------------
    def replace_tokenizer_vocabulary(
        self,
        tokenizer_name: str,
        vocabulary_rows: list[dict[str, Any]],
    ) -> int:
        tokenizer_id = self.ensure_tokenizer_id(tokenizer_name)
        if not vocabulary_rows:
            return tokenizer_id

        df = pd.DataFrame(vocabulary_rows)
        df["tokenizer_id"] = tokenizer_id
        df = df[["tokenizer_id", "token_id", "vocabulary_tokens", "decoded_tokens"]]
        self.queries.upsert_table(df, self.tokenizer_vocabulary_table)
        return tokenizer_id

    # -------------------------------------------------------------------------
    def _build_tokenizer_report_response(
        self, storage: dict[str, Any]
    ) -> dict[str, Any]:
        created_at = pd.to_datetime(
            storage.get("created_at"), utc=True, errors="coerce"
        )
        created_at_iso = (
            created_at.isoformat().replace("+00:00", "Z")
            if not pd.isna(created_at)
            else ""
        )
        metadata = storage.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        metadata_payload = dict(metadata) if isinstance(metadata, dict) else {}
        histogram = storage.get("token_length_histogram", {})
        if isinstance(histogram, str):
            try:
                histogram = json.loads(histogram)
            except json.JSONDecodeError:
                histogram = {}
        histogram_payload = {
            "bins": list(histogram.get("bins", [])),
            "counts": list(histogram.get("counts", [])),
            "bin_edges": list(histogram.get("bin_edges", [])),
            "min_length": int(histogram.get("min_length", 0) or 0),
            "max_length": int(histogram.get("max_length", 0) or 0),
            "mean_length": float(histogram.get("mean_length", 0.0) or 0.0),
            "median_length": float(histogram.get("median_length", 0.0) or 0.0),
        }
        huggingface_url = metadata_payload.pop("huggingface_url", None)
        if not isinstance(huggingface_url, str) or not huggingface_url.strip():
            huggingface_url = None
        return {
            "report_id": int(storage.get("id") or 0),
            "report_version": int(storage.get("report_version", 1) or 1),
            "created_at": created_at_iso,
            "tokenizer_name": storage.get("tokenizer_name", ""),
            "description": storage.get("description"),
            "huggingface_url": huggingface_url,
            "global_stats": metadata_payload,
            "token_length_histogram": histogram_payload,
            "vocabulary_size": int(metadata_payload.get("vocabulary_size", 0) or 0),
        }

    # -------------------------------------------------------------------------
    def load_latest_tokenizer_report(
        self, tokenizer_name: str
    ) -> dict[str, Any] | None:
        stmt = (
            select(TokenizerReport, Tokenizer.name.label("tokenizer_name"))
            .join(Tokenizer, Tokenizer.id == TokenizerReport.tokenizer_id)
            .where(Tokenizer.name == tokenizer_name)
            .order_by(TokenizerReport.id.desc())
            .limit(1)
        )
        with self._session() as session:
            row = session.execute(stmt).first()
        if row is None or row[0] is None:
            return None
        report_row, tokenizer_name_value = row
        storage = {
            "id": report_row.id,
            "tokenizer_id": report_row.tokenizer_id,
            "report_version": report_row.report_version,
            "created_at": report_row.created_at,
            "metadata": report_row.metadata_json,
            "token_length_histogram": report_row.token_length_histogram,
            "description": report_row.description,
            "tokenizer_name": tokenizer_name_value,
        }
        return self._build_tokenizer_report_response(storage)

    # -------------------------------------------------------------------------
    def load_tokenizer_report_by_id(self, report_id: int) -> dict[str, Any] | None:
        stmt = (
            select(TokenizerReport, Tokenizer.name.label("tokenizer_name"))
            .join(Tokenizer, Tokenizer.id == TokenizerReport.tokenizer_id)
            .where(TokenizerReport.id == int(report_id))
            .limit(1)
        )
        with self._session() as session:
            row = session.execute(stmt).first()
        if row is None or row[0] is None:
            return None
        report_row, tokenizer_name_value = row
        storage = {
            "id": report_row.id,
            "tokenizer_id": report_row.tokenizer_id,
            "report_version": report_row.report_version,
            "created_at": report_row.created_at,
            "metadata": report_row.metadata_json,
            "token_length_histogram": report_row.token_length_histogram,
            "description": report_row.description,
            "tokenizer_name": tokenizer_name_value,
        }
        return self._build_tokenizer_report_response(storage)

    # -------------------------------------------------------------------------
    def load_tokenizer_vocabulary_page(
        self,
        report_id: int,
        offset: int,
        limit: int,
    ) -> dict[str, Any] | None:
        report = self.load_tokenizer_report_by_id(report_id)
        if report is None:
            return None
        tokenizer_name = str(report.get("tokenizer_name", ""))
        tokenizer_id = self.get_tokenizer_id(tokenizer_name)
        if tokenizer_id is None:
            return None
        count_stmt = select(func.count(TokenizerVocabulary.id)).where(
            TokenizerVocabulary.tokenizer_id == int(tokenizer_id)
        )
        page_stmt = (
            select(TokenizerVocabulary.token_id, TokenizerVocabulary.vocabulary_tokens)
            .where(TokenizerVocabulary.tokenizer_id == int(tokenizer_id))
            .order_by(TokenizerVocabulary.token_id.asc())
            .limit(int(limit))
            .offset(int(offset))
        )
        with self._session() as session:
            total = int(session.execute(count_stmt).scalar_one_or_none() or 0)
            rows = session.execute(page_stmt).all()

        items: list[dict[str, Any]] = []
        for token_id, token_value in rows:
            token = str(token_value or "")
            items.append(
                {
                    "token_id": int(token_id),
                    "token": token,
                    "length": len(token),
                }
            )

        return {
            "report_id": int(report_id),
            "tokenizer_name": tokenizer_name,
            "offset": int(offset),
            "limit": int(limit),
            "total": total,
            "items": items,
        }


###############################################################################
class BenchmarkReportSerializer:
    def __init__(self, queries: DataRepositoryQueries | None = None) -> None:
        self.queries = queries or DataRepositoryQueries()
        self.benchmark_report_table = BenchmarkReport.__tablename__
        self.repository = BenchmarkRepository()

    # -------------------------------------------------------------------------
    def save_benchmark_report(self, report_payload: dict[str, Any]) -> int:
        dataset_name = str(report_payload.get("dataset_name") or "")
        dataset_id = self.repository.get_dataset_id(dataset_name)
        if dataset_id is None:
            raise ValueError(
                f"Dataset '{dataset_name}' not found while saving benchmark report."
            )

        selected_metric_keys = report_payload.get("selected_metric_keys", [])
        if not isinstance(selected_metric_keys, list):
            selected_metric_keys = []
        selected_metric_keys = [
            str(key) for key in selected_metric_keys if isinstance(key, str) and key
        ]

        created_at = pd.to_datetime(
            report_payload.get("created_at"), utc=True, errors="coerce"
        )
        if pd.isna(created_at):
            created_at = pd.Timestamp.utcnow()
        created_at_value = created_at.to_pydatetime()

        run_name = report_payload.get("run_name")
        if isinstance(run_name, str):
            run_name = run_name.strip() or None
        else:
            run_name = None

        return self.repository.save_benchmark_report(
            dataset_id=int(dataset_id),
            report_version=int(report_payload.get("report_version", 2) or 2),
            created_at=created_at_value,
            run_name=run_name,
            selected_metric_keys=selected_metric_keys,
            payload=report_payload,
        )

    # -------------------------------------------------------------------------
    def _parse_json(self, payload: Any, default: Any) -> Any:
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return default
        if payload is None:
            return default
        return payload

    # -------------------------------------------------------------------------
    def _normalize_report_row(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = self._parse_json(row.get("payload"), {})
        if not isinstance(payload, dict):
            payload = {}
        created_at = pd.to_datetime(row.get("created_at"), utc=True, errors="coerce")
        created_at_iso = (
            created_at.isoformat().replace("+00:00", "Z")
            if not pd.isna(created_at)
            else None
        )

        selected_metric_keys = self._parse_json(
            row.get("selected_metric_keys"),
            [],
        )
        if not isinstance(selected_metric_keys, list):
            selected_metric_keys = []
        selected_metric_keys = [
            str(key) for key in selected_metric_keys if isinstance(key, str) and key
        ]

        normalized_payload = dict(payload)
        normalized_payload["report_id"] = int(row.get("id") or normalized_payload.get("report_id") or 0)
        normalized_payload["report_version"] = int(
            row.get("report_version") or normalized_payload.get("report_version") or 2
        )
        normalized_payload["created_at"] = created_at_iso
        normalized_payload["run_name"] = row.get("run_name") or normalized_payload.get("run_name")
        normalized_payload["selected_metric_keys"] = selected_metric_keys
        normalized_payload["dataset_name"] = str(
            normalized_payload.get("dataset_name") or row.get("dataset_name") or ""
        )

        return BenchmarkRunResponse.model_validate(normalized_payload).model_dump(mode="json")

    # -------------------------------------------------------------------------
    def list_benchmark_reports(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.repository.list_benchmark_reports(limit)

        summaries: list[dict[str, Any]] = []
        for report_row, dataset_name in rows:
            row = {
                "id": report_row.id,
                "report_version": report_row.report_version,
                "created_at": report_row.created_at,
                "run_name": report_row.run_name,
                "selected_metric_keys": report_row.selected_metric_keys,
                "payload": report_row.payload,
                "dataset_name": dataset_name,
            }
            normalized = self._normalize_report_row(row)
            summaries.append(BenchmarkReportSummary.model_validate(normalized).model_dump(mode="json"))
        return summaries

    # -------------------------------------------------------------------------
    def load_benchmark_report_by_id(self, report_id: int) -> dict[str, Any] | None:
        row = self.repository.get_benchmark_report_by_id(report_id)
        if row is None:
            return None

        report_row, dataset_name = row
        mapped = {
            "id": report_row.id,
            "report_version": report_row.report_version,
            "created_at": report_row.created_at,
            "run_name": report_row.run_name,
            "selected_metric_keys": report_row.selected_metric_keys,
            "payload": report_row.payload,
            "dataset_name": dataset_name,
        }
        return self._normalize_report_row(mapped)
