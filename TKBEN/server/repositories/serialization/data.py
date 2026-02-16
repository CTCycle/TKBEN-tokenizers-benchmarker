from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pandas as pd
import sqlalchemy

from TKBEN.server.repositories.queries.data import DataRepositoryQueries
from TKBEN.server.repositories.schemas.models import (
    AnalysisSession,
    Dataset,
    DatasetDocument,
    DatasetDocumentStatistics,
    DatasetReport,
    DatasetValidationReport,
    HistogramArtifact,
    MetricType,
    MetricValue,
    Tokenizer,
    TokenizerReport,
    TokenizerVocabulary,
    TokenizationDocumentStats,
)

K_ERROR = "k error"

###############################################################################
class DatasetSerializer:
    def __init__(self, queries: DataRepositoryQueries | None = None) -> None:
        self.queries = queries or DataRepositoryQueries()
        self.dataset_dimension_table = Dataset.__tablename__
        self.dataset_table = DatasetDocument.__tablename__
        self.stats_table = DatasetDocumentStatistics.__tablename__
        self.reports_table = DatasetReport.__tablename__
        self.validation_reports_table = DatasetValidationReport.__tablename__
        self.analysis_session_table = AnalysisSession.__tablename__
        self.metric_type_table = MetricType.__tablename__
        self.metric_value_table = MetricValue.__tablename__
        self.histogram_table = HistogramArtifact.__tablename__
        self.local_stats_table = TokenizationDocumentStats.__tablename__

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
                    lambda value: json.dumps(value) if isinstance(value, (list, dict)) else value
                )
        return df_copy

    # -------------------------------------------------------------------------
    def list_dataset_previews(self) -> list[dict[str, Any]]:
        query = sqlalchemy.text(
            'SELECT d."name" as dataset_name, '
            'COUNT(dd."id") as document_count '
            'FROM "dataset" d '
            'LEFT JOIN "dataset_document" dd ON dd."dataset_id" = d."id" '
            'GROUP BY d."name" ORDER BY d."name"'
        )
        with self.queries.engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        previews: list[dict[str, Any]] = []
        for row in rows:
            if hasattr(row, "_mapping"):
                data = row._mapping
                name = data.get("dataset_name")
                count = data.get("document_count", 0)
            else:
                name = row[0]
                count = row[1] if len(row) > 1 else 0
            if name is None:
                continue
            previews.append(
                {"dataset_name": str(name), "document_count": int(count or 0)}
            )
        return previews

    # -------------------------------------------------------------------------
    def list_dataset_names(self) -> list[str]:
        return self.queries.get_distinct_values(self.dataset_dimension_table, "name")

    # -------------------------------------------------------------------------
    def get_dataset_id(self, dataset_name: str) -> int | None:
        query = sqlalchemy.text('SELECT "id" FROM "dataset" WHERE "name" = :dataset LIMIT 1')
        with self.queries.engine.connect() as conn:
            result = conn.execute(query, {"dataset": dataset_name})
            row = result.first()
        if row is None:
            return None
        if hasattr(row, "_mapping"):
            return int(row._mapping.get("id"))
        return int(row[0])

    # -------------------------------------------------------------------------
    def ensure_dataset_id(self, dataset_name: str) -> int:
        query = sqlalchemy.text(
            'INSERT INTO "dataset" ("name") '
            'VALUES (:dataset) '
            'ON CONFLICT ("name") DO NOTHING'
        )
        with self.queries.engine.begin() as conn:
            conn.execute(query, {"dataset": dataset_name})
        dataset_id = self.get_dataset_id(dataset_name)
        if dataset_id is None:
            raise ValueError(f"Failed to resolve dataset id for '{dataset_name}'")
        return dataset_id

    # -------------------------------------------------------------------------
    def dataset_exists(self, dataset_name: str) -> bool:
        query = sqlalchemy.text(
            'SELECT 1 FROM "dataset" WHERE "name" = :dataset LIMIT 1'
        )
        with self.queries.engine.connect() as conn:
            result = conn.execute(query, {"dataset": dataset_name})
            return result.first() is not None

    # -------------------------------------------------------------------------
    def count_dataset_documents(self, dataset_name: str) -> int:
        query = sqlalchemy.text(
            'SELECT COUNT(*) '
            'FROM "dataset_document" dd '
            'JOIN "dataset" d ON d."id" = dd."dataset_id" '
            'WHERE d."name" = :dataset'
        )
        with self.queries.engine.connect() as conn:
            result = conn.execute(query, {"dataset": dataset_name})
            value = result.scalar() or 0
        return int(value)

    # -------------------------------------------------------------------------
    def iterate_dataset_batches(
        self,
        dataset_name: str,
        batch_size: int,
    ) -> Iterator[list[str]]:
        offset = 0
        while True:
            query = sqlalchemy.text(
                'SELECT dd."text" FROM "dataset_document" dd '
                'JOIN "dataset" d ON d."id" = dd."dataset_id" '
                'WHERE d."name" = :dataset '
                'ORDER BY dd."id" LIMIT :limit OFFSET :offset'
            )
            with self.queries.engine.connect() as conn:
                result = conn.execute(
                    query,
                    {"dataset": dataset_name, "limit": batch_size, "offset": offset},
                )
                rows = result.fetchall()

            if not rows:
                break

            texts: list[str] = []
            for row in rows:
                if hasattr(row, "_mapping"):
                    text_value = row._mapping.get("text")
                else:
                    text_value = row[0] if row else None
                if text_value is None:
                    continue
                texts.append(str(text_value))

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
        conditions = ['d."name" = :dataset']
        params: dict[str, Any] = {"dataset": dataset_name}
        if isinstance(min_length, int):
            conditions.append('LENGTH(dd."text") >= :min_length')
            params["min_length"] = int(min_length)
        if isinstance(max_length, int):
            conditions.append('LENGTH(dd."text") <= :max_length')
            params["max_length"] = int(max_length)
        if exclude_empty:
            conditions.append('LENGTH(TRIM(dd."text")) > 0')
        where_clause = " AND ".join(conditions)
        while True:
            query = sqlalchemy.text(
                'SELECT dd."id", dd."text" FROM "dataset_document" dd '
                'JOIN "dataset" d ON d."id" = dd."dataset_id" '
                f"WHERE {where_clause} "
                'ORDER BY dd."id" LIMIT :limit OFFSET :offset'
            )
            with self.queries.engine.connect() as conn:
                batch_params = {
                    **params,
                    "limit": batch_size,
                    "offset": offset,
                }
                result = conn.execute(
                    query,
                    batch_params,
                )
                rows = result.fetchall()

            if not rows:
                break

            batch: list[dict[str, Any]] = []
            for row in rows:
                if hasattr(row, "_mapping"):
                    mapping = row._mapping
                    row_id = mapping.get("id")
                    text_value = mapping.get("text")
                else:
                    row_id = row[0] if row else None
                    text_value = row[1] if len(row) > 1 else None

                if row_id is None or text_value is None:
                    continue
                batch.append({"id": int(row_id), "text": str(text_value)})

            if batch:
                yield batch
            offset += len(rows)

    # -------------------------------------------------------------------------
    def delete_dataset(self, dataset_name: str) -> None:
        query = sqlalchemy.text('DELETE FROM "dataset" WHERE "name" = :dataset')
        with self.queries.engine.begin() as conn:
            conn.execute(query, {"dataset": dataset_name})

    # -------------------------------------------------------------------------
    def delete_dataset_statistics(self, dataset_name: str) -> None:
        query = sqlalchemy.text(
            'DELETE FROM "dataset_document_statistics" '
            'WHERE "document_id" IN ('
            'SELECT dd."id" '
            'FROM "dataset_document" dd '
            'JOIN "dataset" d ON d."id" = dd."dataset_id" '
            'WHERE d."name" = :dataset)'
        )
        with self.queries.engine.begin() as conn:
            conn.execute(query, {"dataset": dataset_name})

    # -------------------------------------------------------------------------
    def save_dataset_statistics_batch(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        df = pd.DataFrame(batch)
        self.queries.insert_table(df, self.stats_table)

    # -------------------------------------------------------------------------
    def _coerce_datetime(self, value: Any) -> Any:
        if value is None:
            return pd.Timestamp.utcnow().to_pydatetime()
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            return pd.Timestamp.utcnow().to_pydatetime()
        return parsed.to_pydatetime()

    # -------------------------------------------------------------------------
    def _normalize_histogram(self, storage: Any) -> dict[str, Any]:
        histogram = self.parse_json(storage, default={})
        return {
            "bins": self.parse_json(histogram.get("bins"), default=[]),
            "counts": self.parse_json(histogram.get("counts"), default=[]),
            "bin_edges": self.parse_json(histogram.get("bin_edges"), default=[]),
            "min_length": int(histogram.get("min_length", 0) or 0),
            "max_length": int(histogram.get("max_length", 0) or 0),
            "mean_length": float(histogram.get("mean_length", 0.0) or 0.0),
            "median_length": float(histogram.get("median_length", 0.0) or 0.0),
        }

    # -------------------------------------------------------------------------
    def build_validation_report_storage(self, report: dict[str, Any]) -> dict[str, Any]:
        return {
            "dataset_name": report.get("dataset_name", ""),
            "report_version": int(report.get("report_version", 1) or 1),
            "created_at": report.get("created_at"),
            "aggregate_statistics": report.get("aggregate_statistics", {}),
            "document_histogram": report.get("document_length_histogram", {}),
            "word_histogram": report.get("word_length_histogram", {}),
            "most_common_words": report.get("most_common_words", []),
            "least_common_words": report.get("least_common_words", []),
            "longest_words": report.get("longest_words", []),
            "shortest_words": report.get("shortest_words", []),
            "word_cloud_terms": report.get("word_cloud_terms", []),
            "per_document_stats": report.get("per_document_stats", {}),
        }

    # -------------------------------------------------------------------------
    def build_report_storage(self, report: dict[str, Any]) -> dict[str, Any]:
        document_histogram = report.get("document_length_histogram", {})
        word_histogram = report.get("word_length_histogram", {})
        return {
            "dataset_name": report.get("dataset_name", ""),
            "document_statistics": {
                "document_count": report.get("document_count", 0),
                "document_bins": document_histogram.get("bins", []),
                "document_counts": document_histogram.get("counts", []),
                "document_bin_edges": document_histogram.get("bin_edges", []),
                "document_min_length": document_histogram.get("min_length", 0),
                "document_max_length": document_histogram.get("max_length", 0),
                "document_mean_length": document_histogram.get("mean_length", 0.0),
                "document_median_length": document_histogram.get("median_length", 0.0),
            },
            "word_statistics": {
                "word_bins": word_histogram.get("bins", []),
                "word_counts": word_histogram.get("counts", []),
                "word_bin_edges": word_histogram.get("bin_edges", []),
                "word_min_length": word_histogram.get("min_length", 0),
                "word_max_length": word_histogram.get("max_length", 0),
                "word_mean_length": word_histogram.get("mean_length", 0.0),
                "word_median_length": word_histogram.get("median_length", 0.0),
            },
            "most_common_words": report.get("most_common_words", []),
            "least_common_words": report.get("least_common_words", []),
        }

    # -------------------------------------------------------------------------
    def save_dataset_validation_report(self, report: dict[str, Any]) -> int:
        storage = self.build_validation_report_storage(report)
        dataset_name = str(storage.get("dataset_name") or "")
        dataset_id = self.get_dataset_id(dataset_name)
        if dataset_id is None:
            raise ValueError(f"Dataset '{dataset_name}' not found while saving report.")

        insert_query = sqlalchemy.text(
            'INSERT INTO "dataset_validation_report" ('
            '"dataset_id", "report_version", "created_at", "aggregate_statistics", '
            '"document_histogram", "word_histogram", "most_common_words", '
            '"least_common_words", "longest_words", "shortest_words", '
            '"word_cloud_terms", "per_document_stats") '
            "VALUES ("
            ':dataset_id, :report_version, :created_at, :aggregate_statistics, '
            ':document_histogram, :word_histogram, :most_common_words, '
            ':least_common_words, :longest_words, :shortest_words, '
            ':word_cloud_terms, :per_document_stats)'
        )

        with self.queries.engine.begin() as conn:
            conn.execute(
                insert_query,
                {
                    "dataset_id": dataset_id,
                    "report_version": int(storage.get("report_version", 1) or 1),
                    "created_at": self._coerce_datetime(storage.get("created_at")),
                    "aggregate_statistics": json.dumps(
                        storage.get("aggregate_statistics", {})
                    ),
                    "document_histogram": json.dumps(
                        storage.get("document_histogram", {})
                    ),
                    "word_histogram": json.dumps(storage.get("word_histogram", {})),
                    "most_common_words": json.dumps(
                        storage.get("most_common_words", [])
                    ),
                    "least_common_words": json.dumps(
                        storage.get("least_common_words", [])
                    ),
                    "longest_words": json.dumps(storage.get("longest_words", [])),
                    "shortest_words": json.dumps(storage.get("shortest_words", [])),
                    "word_cloud_terms": json.dumps(
                        storage.get("word_cloud_terms", [])
                    ),
                    "per_document_stats": json.dumps(
                        storage.get("per_document_stats", {})
                    ),
                },
            )
            result = conn.execute(
                sqlalchemy.text(
                    'SELECT "id" FROM "dataset_validation_report" '
                    'WHERE "dataset_id" = :dataset_id ORDER BY "id" DESC LIMIT 1'
                ),
                {"dataset_id": dataset_id},
            )
            row = result.first()

        if row is None:
            raise ValueError("Failed to resolve saved dataset validation report id.")
        if hasattr(row, "_mapping"):
            return int(row._mapping["id"])
        return int(row[0])

    # -------------------------------------------------------------------------
    def save_dataset_report(self, report: dict[str, Any]) -> int:
        storage = self.build_report_storage(report)
        dataset_name = str(storage.get("dataset_name") or "")
        dataset_id = self.get_dataset_id(dataset_name)
        if dataset_id is None:
            raise ValueError(f"Dataset '{dataset_name}' not found while saving report.")
        row = {
            "dataset_id": dataset_id,
            "document_statistics": storage.get("document_statistics"),
            "word_statistics": storage.get("word_statistics"),
            "most_common_words": storage.get("most_common_words"),
            "least_common_words": storage.get("least_common_words"),
        }
        df = pd.DataFrame([row])
        df = self.serialize_json_columns(df)
        self.queries.upsert_table(df, self.reports_table)
        return self.save_dataset_validation_report(report)

    # -------------------------------------------------------------------------
    def build_validation_report_response(self, storage: dict[str, Any]) -> dict[str, Any]:
        aggregate_statistics = self.parse_json(
            storage.get("aggregate_statistics"), default={}
        )
        document_histogram = self._normalize_histogram(storage.get("document_histogram"))
        word_histogram = self._normalize_histogram(storage.get("word_histogram"))
        created_at_raw = storage.get("created_at")
        created_at = pd.to_datetime(created_at_raw, utc=True, errors="coerce")
        created_at_iso = (
            created_at.isoformat().replace("+00:00", "Z")
            if not pd.isna(created_at)
            else ""
        )
        return {
            "report_id": int(storage.get("id") or 0),
            "report_version": int(storage.get("report_version", 1) or 1),
            "created_at": created_at_iso,
            "dataset_name": storage.get("dataset_name", ""),
            "document_count": int(aggregate_statistics.get("document_count", 0) or 0),
            "document_length_histogram": document_histogram,
            "word_length_histogram": word_histogram,
            "min_document_length": document_histogram["min_length"],
            "max_document_length": document_histogram["max_length"],
            "most_common_words": self.parse_json(storage.get("most_common_words"), default=[]),
            "least_common_words": self.parse_json(storage.get("least_common_words"), default=[]),
            "longest_words": self.parse_json(storage.get("longest_words"), default=[]),
            "shortest_words": self.parse_json(storage.get("shortest_words"), default=[]),
            "word_cloud_terms": self.parse_json(storage.get("word_cloud_terms"), default=[]),
            "aggregate_statistics": aggregate_statistics,
            "per_document_stats": self.parse_json(storage.get("per_document_stats"), default={}),
        }

    # -------------------------------------------------------------------------
    def build_report_response(self, storage: dict[str, Any]) -> dict[str, Any]:
        document_statistics = self.parse_json(
            storage.get("document_statistics"), default={}
        )
        word_statistics = self.parse_json(storage.get("word_statistics"), default={})
        document_histogram = {
            "bins": self.parse_json(
                document_statistics.get("document_bins"), default=[]
            ),
            "counts": self.parse_json(
                document_statistics.get("document_counts"), default=[]
            ),
            "bin_edges": self.parse_json(
                document_statistics.get("document_bin_edges"), default=[]
            ),
            "min_length": int(document_statistics.get("document_min_length", 0) or 0),
            "max_length": int(document_statistics.get("document_max_length", 0) or 0),
            "mean_length": float(
                document_statistics.get("document_mean_length", 0.0) or 0.0
            ),
            "median_length": float(
                document_statistics.get("document_median_length", 0.0) or 0.0
            ),
        }
        word_histogram = {
            "bins": self.parse_json(word_statistics.get("word_bins"), default=[]),
            "counts": self.parse_json(word_statistics.get("word_counts"), default=[]),
            "bin_edges": self.parse_json(
                word_statistics.get("word_bin_edges"), default=[]
            ),
            "min_length": int(word_statistics.get("word_min_length", 0) or 0),
            "max_length": int(word_statistics.get("word_max_length", 0) or 0),
            "mean_length": float(word_statistics.get("word_mean_length", 0.0) or 0.0),
            "median_length": float(
                word_statistics.get("word_median_length", 0.0) or 0.0
            ),
        }
        return {
            "report_id": None,
            "report_version": 1,
            "created_at": None,
            "dataset_name": storage.get("dataset_name", ""),
            "document_count": int(
                document_statistics.get("document_count", 0) or 0
            ),
            "document_length_histogram": document_histogram,
            "word_length_histogram": word_histogram,
            "min_document_length": document_histogram["min_length"],
            "max_document_length": document_histogram["max_length"],
            "most_common_words": self.parse_json(storage.get("most_common_words"), default=[]),
            "least_common_words": self.parse_json(storage.get("least_common_words"), default=[]),
            "longest_words": [],
            "shortest_words": [],
            "word_cloud_terms": [],
            "aggregate_statistics": {
                "document_count": int(document_statistics.get("document_count", 0) or 0),
            },
            "per_document_stats": {},
        }

    # -------------------------------------------------------------------------
    def load_latest_dataset_validation_report(
        self,
        dataset_name: str,
    ) -> dict[str, Any] | None:
        return self.load_latest_analysis_report(dataset_name)

    # -------------------------------------------------------------------------
    def load_dataset_validation_report_by_id(
        self,
        report_id: int,
    ) -> dict[str, Any] | None:
        return self.load_analysis_report_by_session_id(report_id)

    # -------------------------------------------------------------------------
    def load_legacy_dataset_report(self, dataset_name: str) -> dict[str, Any] | None:
        query = sqlalchemy.text(
            'SELECT dr.*, d."name" AS "dataset_name" '
            'FROM "dataset_report" dr '
            'JOIN "dataset" d ON d."id" = dr."dataset_id" '
            'WHERE d."name" = :dataset LIMIT 1'
        )
        with self.queries.engine.connect() as conn:
            result = conn.execute(query, {"dataset": dataset_name})
            row = result.first()

        if row is None:
            return None
        if hasattr(row, "_mapping"):
            storage = dict(row._mapping)
        else:
            storage = {key: value for key, value in zip(result.keys(), row, strict=False)}
        return self.build_report_response(storage)

    # -------------------------------------------------------------------------
    def load_dataset_report(self, dataset_name: str) -> dict[str, Any] | None:
        return self.load_latest_analysis_report(dataset_name)

    # -------------------------------------------------------------------------
    def ensure_metric_types_seeded(self, metric_catalog: list[dict[str, Any]]) -> None:
        query = sqlalchemy.text(
            'INSERT INTO "metric_type" ("key", "category", "label", "description", "scope", "value_kind") '
            "VALUES (:key, :category, :label, :description, :scope, :value_kind) "
            'ON CONFLICT ("key") DO UPDATE SET '
            '"category" = EXCLUDED."category", '
            '"label" = EXCLUDED."label", '
            '"description" = EXCLUDED."description", '
            '"scope" = EXCLUDED."scope", '
            '"value_kind" = EXCLUDED."value_kind"'
        )
        with self.queries.engine.begin() as conn:
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
                    conn.execute(
                        query,
                        {
                            "key": str(key),
                            "category": category_key,
                            "label": str(label),
                            "description": str(metric.get("description") or ""),
                            "scope": str(metric.get("scope") or "aggregate"),
                            "value_kind": str(metric.get("value_kind") or "number"),
                        },
                    )

    # -------------------------------------------------------------------------
    def get_metric_type_map(self) -> dict[str, int]:
        query = sqlalchemy.text('SELECT "id", "key" FROM "metric_type"')
        with self.queries.engine.connect() as conn:
            rows = conn.execute(query).fetchall()
        result: dict[str, int] = {}
        for row in rows:
            if hasattr(row, "_mapping"):
                result[str(row._mapping["key"])] = int(row._mapping["id"])
            else:
                result[str(row[1])] = int(row[0])
        return result

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
        query = sqlalchemy.text(
            'INSERT INTO "analysis_session" ('
            '"dataset_id", "session_name", "status", "report_version", '
            '"created_at", "completed_at", "parameters", "selected_metric_keys") '
            "VALUES ("
            ':dataset_id, :session_name, :status, :report_version, '
            ':created_at, :completed_at, :parameters, :selected_metric_keys)'
        )
        with self.queries.engine.begin() as conn:
            conn.execute(
                query,
                {
                    "dataset_id": dataset_id,
                    "session_name": session_name,
                    "status": "running",
                    "report_version": int(report_version),
                    "created_at": created_at,
                    "completed_at": None,
                    "parameters": json.dumps(parameters),
                    "selected_metric_keys": json.dumps(selected_metric_keys),
                },
            )
            row = conn.execute(
                sqlalchemy.text(
                    'SELECT "id" FROM "analysis_session" '
                    'WHERE "dataset_id" = :dataset_id '
                    'ORDER BY "id" DESC LIMIT 1'
                ),
                {"dataset_id": dataset_id},
            ).first()
        if row is None:
            raise ValueError("Failed to create analysis session.")
        if hasattr(row, "_mapping"):
            return int(row._mapping["id"])
        return int(row[0])

    # -------------------------------------------------------------------------
    def complete_analysis_session(self, session_id: int, status: str = "completed") -> None:
        query = sqlalchemy.text(
            'UPDATE "analysis_session" '
            'SET "status" = :status, "completed_at" = :completed_at '
            'WHERE "id" = :session_id'
        )
        with self.queries.engine.begin() as conn:
            conn.execute(
                query,
                {
                    "status": str(status),
                    "completed_at": pd.Timestamp.utcnow().to_pydatetime(),
                    "session_id": int(session_id),
                },
            )

    # -------------------------------------------------------------------------
    def save_metric_values_batch(self, session_id: int, batch: list[dict[str, Any]]) -> None:
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
        df = pd.DataFrame(rows)
        self.queries.insert_table(df, self.metric_value_table, ignore_duplicates=False)

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
        query = sqlalchemy.text(
            'SELECT mv."document_id", mt."key", mv."numeric_value", mv."text_value", mv."json_value" '
            'FROM "metric_value" mv '
            'JOIN "metric_type" mt ON mt."id" = mv."metric_type_id" '
            'WHERE mv."session_id" = :session_id '
            'ORDER BY mv."id" ASC'
        )
        with self.queries.engine.connect() as conn:
            rows = conn.execute(query, {"session_id": int(session_id)}).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            if hasattr(row, "_mapping"):
                result.append(
                    {
                        "document_id": row._mapping.get("document_id"),
                        "key": str(row._mapping.get("key") or ""),
                        "numeric_value": row._mapping.get("numeric_value"),
                        "text_value": row._mapping.get("text_value"),
                        "json_value": row._mapping.get("json_value"),
                    }
                )
            else:
                result.append(
                    {
                        "document_id": row[0],
                        "key": str(row[1] or ""),
                        "numeric_value": row[2],
                        "text_value": row[3],
                        "json_value": row[4],
                    }
                )
        return result

    # -------------------------------------------------------------------------
    def _load_histogram_rows_for_session(self, session_id: int) -> dict[str, Any]:
        query = sqlalchemy.text(
            'SELECT mt."key", ha."bins", ha."counts", ha."bin_edges", '
            'ha."min_value", ha."max_value", ha."mean_value", ha."median_value" '
            'FROM "histogram_artifact" ha '
            'JOIN "metric_type" mt ON mt."id" = ha."metric_type_id" '
            'WHERE ha."session_id" = :session_id'
        )
        with self.queries.engine.connect() as conn:
            rows = conn.execute(query, {"session_id": int(session_id)}).fetchall()
        result: dict[str, Any] = {}
        for row in rows:
            if hasattr(row, "_mapping"):
                key = str(row._mapping.get("key") or "")
                bins = self.parse_json(row._mapping.get("bins"), default=[])
                counts = self.parse_json(row._mapping.get("counts"), default=[])
                bin_edges = self.parse_json(row._mapping.get("bin_edges"), default=[])
                min_value = float(row._mapping.get("min_value", 0.0) or 0.0)
                max_value = float(row._mapping.get("max_value", 0.0) or 0.0)
                mean_value = float(row._mapping.get("mean_value", 0.0) or 0.0)
                median_value = float(row._mapping.get("median_value", 0.0) or 0.0)
            else:
                key = str(row[0] or "")
                bins = self.parse_json(row[1], default=[])
                counts = self.parse_json(row[2], default=[])
                bin_edges = self.parse_json(row[3], default=[])
                min_value = float(row[4] or 0.0)
                max_value = float(row[5] or 0.0)
                mean_value = float(row[6] or 0.0)
                median_value = float(row[7] or 0.0)
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
    def _build_session_report_response(self, session_row: dict[str, Any]) -> dict[str, Any]:
        session_id = int(session_row.get("id") or 0)
        metric_rows = self._load_metric_rows_for_session(session_id)
        histogram_rows = self._load_histogram_rows_for_session(session_id)

        aggregate_statistics: dict[str, Any] = {}
        per_document: dict[int, dict[str, Any]] = {}
        for row in metric_rows:
            key = str(row.get("key") or "")
            value: Any = row.get("numeric_value")
            if value is None and row.get("text_value") is not None:
                value = row.get("text_value")
            if value is None and row.get("json_value") is not None:
                value = self.parse_json(row.get("json_value"), default={})
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
        created_at = pd.to_datetime(session_row.get("created_at"), utc=True, errors="coerce")
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
            "selected_metric_keys": self.parse_json(session_row.get("selected_metric_keys"), default=[]),
            "session_parameters": self.parse_json(session_row.get("parameters"), default={}),
            "document_count": int(aggregate_statistics.get("corpus.document_count", 0) or 0),
            "document_length_histogram": {
                "bins": list(document_histogram.get("bins", [])),
                "counts": list(document_histogram.get("counts", [])),
                "bin_edges": list(document_histogram.get("bin_edges", [])),
                "min_length": int(document_histogram.get("min_length", 0) or 0),
                "max_length": int(document_histogram.get("max_length", 0) or 0),
                "mean_length": float(document_histogram.get("mean_length", 0.0) or 0.0),
                "median_length": float(document_histogram.get("median_length", 0.0) or 0.0),
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
            "most_common_words": self.parse_json(aggregate_statistics.get("words.most_common"), default=[]),
            "least_common_words": self.parse_json(aggregate_statistics.get("words.least_common"), default=[]),
            "longest_words": self.parse_json(aggregate_statistics.get("words.longest"), default=[]),
            "shortest_words": self.parse_json(aggregate_statistics.get("words.shortest"), default=[]),
            "word_cloud_terms": self.parse_json(aggregate_statistics.get("words.word_cloud"), default=[]),
            "aggregate_statistics": aggregate_statistics,
            "per_document_stats": per_document_stats,
        }

    # -------------------------------------------------------------------------
    def load_latest_analysis_report(self, dataset_name: str) -> dict[str, Any] | None:
        query = sqlalchemy.text(
            'SELECT s.*, d."name" AS "dataset_name" '
            'FROM "analysis_session" s '
            'JOIN "dataset" d ON d."id" = s."dataset_id" '
            'WHERE d."name" = :dataset AND s."status" = :status '
            'ORDER BY s."id" DESC LIMIT 1'
        )
        with self.queries.engine.connect() as conn:
            row = conn.execute(
                query,
                {"dataset": dataset_name, "status": "completed"},
            ).first()
        if row is None:
            return None
        if hasattr(row, "_mapping"):
            return self._build_session_report_response(dict(row._mapping))
        return None

    # -------------------------------------------------------------------------
    def load_analysis_report_by_session_id(self, session_id: int) -> dict[str, Any] | None:
        query = sqlalchemy.text(
            'SELECT s.*, d."name" AS "dataset_name" '
            'FROM "analysis_session" s '
            'JOIN "dataset" d ON d."id" = s."dataset_id" '
            'WHERE s."id" = :session_id LIMIT 1'
        )
        with self.queries.engine.connect() as conn:
            row = conn.execute(query, {"session_id": int(session_id)}).first()
        if row is None:
            return None
        if hasattr(row, "_mapping"):
            return self._build_session_report_response(dict(row._mapping))
        return None


###############################################################################
class TokenizerReportSerializer:
    def __init__(self, queries: DataRepositoryQueries | None = None) -> None:
        self.queries = queries or DataRepositoryQueries()
        self.tokenizer_table = Tokenizer.__tablename__
        self.tokenizer_report_table = TokenizerReport.__tablename__
        self.tokenizer_vocabulary_table = TokenizerVocabulary.__tablename__

    # -------------------------------------------------------------------------
    def get_tokenizer_id(self, tokenizer_name: str) -> int | None:
        query = sqlalchemy.text(
            'SELECT "id" FROM "tokenizer" WHERE "name" = :name LIMIT 1'
        )
        with self.queries.engine.connect() as conn:
            row = conn.execute(query, {"name": tokenizer_name}).first()
        if row is None:
            return None
        if hasattr(row, "_mapping"):
            return int(row._mapping["id"])
        return int(row[0])

    # -------------------------------------------------------------------------
    def ensure_tokenizer_id(self, tokenizer_name: str) -> int:
        query = sqlalchemy.text(
            'INSERT INTO "tokenizer" ("name") VALUES (:name) '
            'ON CONFLICT ("name") DO NOTHING'
        )
        with self.queries.engine.begin() as conn:
            conn.execute(query, {"name": tokenizer_name})
        tokenizer_id = self.get_tokenizer_id(tokenizer_name)
        if tokenizer_id is None:
            raise ValueError(f"Failed to resolve tokenizer id for '{tokenizer_name}'")
        return tokenizer_id

    # -------------------------------------------------------------------------
    def save_tokenizer_report(self, report: dict[str, Any]) -> int:
        tokenizer_name = str(report.get("tokenizer_name") or "")
        tokenizer_id = self.ensure_tokenizer_id(tokenizer_name)
        insert_query = sqlalchemy.text(
            'INSERT INTO "tokenizer_report" ('
            '"tokenizer_id", "report_version", "created_at", "metadata", '
            '"token_length_histogram", "description") '
            "VALUES ("
            ':tokenizer_id, :report_version, :created_at, :metadata, '
            ':token_length_histogram, :description)'
        )
        created_at = pd.to_datetime(
            report.get("created_at"), utc=True, errors="coerce"
        )
        if pd.isna(created_at):
            created_at = pd.Timestamp.utcnow()

        with self.queries.engine.begin() as conn:
            conn.execute(
                insert_query,
                {
                    "tokenizer_id": tokenizer_id,
                    "report_version": int(report.get("report_version", 1) or 1),
                    "created_at": created_at.to_pydatetime(),
                    "metadata": json.dumps(report.get("global_stats", {})),
                    "token_length_histogram": json.dumps(
                        report.get("token_length_histogram", {})
                    ),
                    "description": report.get("description"),
                },
            )
            row = conn.execute(
                sqlalchemy.text(
                    'SELECT "id" FROM "tokenizer_report" '
                    'WHERE "tokenizer_id" = :tokenizer_id ORDER BY "id" DESC LIMIT 1'
                ),
                {"tokenizer_id": tokenizer_id},
            ).first()

        if row is None:
            raise ValueError("Failed to resolve saved tokenizer report id.")
        if hasattr(row, "_mapping"):
            return int(row._mapping["id"])
        return int(row[0])

    # -------------------------------------------------------------------------
    def replace_tokenizer_vocabulary(
        self,
        tokenizer_name: str,
        vocabulary_rows: list[dict[str, Any]],
    ) -> int:
        tokenizer_id = self.ensure_tokenizer_id(tokenizer_name)
        with self.queries.engine.begin() as conn:
            conn.execute(
                sqlalchemy.text(
                    'DELETE FROM "tokenizer_vocabulary" WHERE "tokenizer_id" = :tokenizer_id'
                ),
                {"tokenizer_id": tokenizer_id},
            )
        if vocabulary_rows:
            df = pd.DataFrame(vocabulary_rows)
            df["tokenizer_id"] = tokenizer_id
            df = df[["tokenizer_id", "token_id", "vocabulary_tokens", "decoded_tokens"]]
            self.queries.insert_table(
                df,
                self.tokenizer_vocabulary_table,
                ignore_duplicates=False,
            )
        return tokenizer_id

    # -------------------------------------------------------------------------
    def _build_tokenizer_report_response(self, storage: dict[str, Any]) -> dict[str, Any]:
        created_at = pd.to_datetime(storage.get("created_at"), utc=True, errors="coerce")
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
        return {
            "report_id": int(storage.get("id") or 0),
            "report_version": int(storage.get("report_version", 1) or 1),
            "created_at": created_at_iso,
            "tokenizer_name": storage.get("tokenizer_name", ""),
            "description": storage.get("description"),
            "global_stats": metadata,
            "token_length_histogram": histogram_payload,
            "vocabulary_size": int(metadata.get("vocabulary_size", 0) or 0),
        }

    # -------------------------------------------------------------------------
    def load_latest_tokenizer_report(self, tokenizer_name: str) -> dict[str, Any] | None:
        query = sqlalchemy.text(
            'SELECT tr.*, t."name" AS "tokenizer_name" '
            'FROM "tokenizer_report" tr '
            'JOIN "tokenizer" t ON t."id" = tr."tokenizer_id" '
            'WHERE t."name" = :tokenizer_name '
            'ORDER BY tr."id" DESC LIMIT 1'
        )
        with self.queries.engine.connect() as conn:
            row = conn.execute(query, {"tokenizer_name": tokenizer_name}).first()
        if row is None:
            return None
        storage = dict(row._mapping) if hasattr(row, "_mapping") else {}
        return self._build_tokenizer_report_response(storage)

    # -------------------------------------------------------------------------
    def load_tokenizer_report_by_id(self, report_id: int) -> dict[str, Any] | None:
        query = sqlalchemy.text(
            'SELECT tr.*, t."name" AS "tokenizer_name" '
            'FROM "tokenizer_report" tr '
            'JOIN "tokenizer" t ON t."id" = tr."tokenizer_id" '
            'WHERE tr."id" = :report_id LIMIT 1'
        )
        with self.queries.engine.connect() as conn:
            row = conn.execute(query, {"report_id": int(report_id)}).first()
        if row is None:
            return None
        storage = dict(row._mapping) if hasattr(row, "_mapping") else {}
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

        count_query = sqlalchemy.text(
            'SELECT COUNT(*) FROM "tokenizer_vocabulary" WHERE "tokenizer_id" = :tokenizer_id'
        )
        page_query = sqlalchemy.text(
            'SELECT "token_id", "vocabulary_tokens" '
            'FROM "tokenizer_vocabulary" '
            'WHERE "tokenizer_id" = :tokenizer_id '
            'ORDER BY "token_id" ASC '
            'LIMIT :limit OFFSET :offset'
        )
        with self.queries.engine.connect() as conn:
            total = int(conn.execute(count_query, {"tokenizer_id": tokenizer_id}).scalar() or 0)
            rows = conn.execute(
                page_query,
                {
                    "tokenizer_id": tokenizer_id,
                    "limit": int(limit),
                    "offset": int(offset),
                },
            ).fetchall()

        items: list[dict[str, Any]] = []
        for row in rows:
            if hasattr(row, "_mapping"):
                token_id = int(row._mapping["token_id"])
                token = str(row._mapping.get("vocabulary_tokens") or "")
            else:
                token_id = int(row[0])
                token = str(row[1] if len(row) > 1 else "")
            items.append(
                {
                    "token_id": token_id,
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
