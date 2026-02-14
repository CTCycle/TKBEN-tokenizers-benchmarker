from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pandas as pd
import sqlalchemy

from TKBEN.server.repositories.queries.data import DataRepositoryQueries
from TKBEN.server.repositories.schemas.models import (
    Dataset,
    DatasetDocument,
    DatasetDocumentStatistics,
    DatasetReport,
    DatasetValidationReport,
    Tokenizer,
    TokenizerReport,
    TokenizerVocabulary,
    TokenizationDocumentStats,
)

K_ERROR = "k error"

###############################################################################
class DataSerializer:
    experiment_table = "ADSORPTION_EXPERIMENT"
    best_fit_table = "ADSORPTION_BEST_FIT"
    experiment_columns = [
        "experiment",
        "temperature [K]",
        "pressure [Pa]",
        "uptake [mol/g]",
        "measurement_count",
        "min_pressure",
        "max_pressure",
        "min_uptake",
        "max_uptake",
    ]
    model_schemas: dict[str, dict[str, Any]] = {
        "LANGMUIR": {
            "prefix": "Langmuir",
            "table": "ADSORPTION_LANGMUIR",
            "fields": {
                "lss": "LSS",
                "k": "k",
                "k_error": K_ERROR,
                "qsat": "qsat",
                "qsat_error": "qsat error",
            },
        },
        "SIPS": {
            "prefix": "Sips",
            "table": "ADSORPTION_SIPS",
            "fields": {
                "lss": "LSS",
                "k": "k",
                "k_error": K_ERROR,
                "qsat": "qsat",
                "qsat_error": "qsat error",
                "exponent": "exponent",
                "exponent_error": "exponent error",
            },
        },
        "FREUNDLICH": {
            "prefix": "Freundlich",
            "table": "ADSORPTION_FREUNDLICH",
            "fields": {
                "lss": "LSS",
                "k": "k",
                "k_error": K_ERROR,
                "exponent": "exponent",
                "exponent_error": "exponent error",
            },
        },
        "TEMKIN": {
            "prefix": "Temkin",
            "table": "ADSORPTION_TEMKIN",
            "fields": {
                "lss": "LSS",
                "k": "k",
                "k_error": K_ERROR,
                "beta": "beta",
                "beta_error": "beta error",
            },
        },
        "TOTH": {
            "prefix": "Toth",
            "table": "ADSORPTION_TOTH",
            "fields": {
                "lss": "LSS",
                "k": "k",
                "k_error": K_ERROR,
                "qsat": "qsat",
                "qsat_error": "qsat error",
                "exponent": "exponent",
                "exponent_error": "exponent error",
            },
        },
        "DUBININ_RADUSHKEVICH": {
            "prefix": "Dubinin-Radushkevich",
            "table": "ADSORPTION_DUBININ_RADUSHKEVICH",
            "fields": {
                "lss": "LSS",
                "qsat": "qsat",
                "qsat_error": "qsat error",
                "beta": "beta",
                "beta_error": "beta error",
            },
        },
        "DUAL_SITE_LANGMUIR": {
            "prefix": "Dual-Site Langmuir",
            "table": "ADSORPTION_DUAL_SITE_LANGMUIR",
            "fields": {
                "lss": "LSS",
                "k1": "k1",
                "k1_error": "k1 error",
                "qsat1": "qsat1",
                "qsat1_error": "qsat1 error",
                "k2": "k2",
                "k2_error": "k2 error",
                "qsat2": "qsat2",
                "qsat2_error": "qsat2 error",
            },
        },
        "REDLICH_PETERSON": {
            "prefix": "Redlich-Peterson",
            "table": "ADSORPTION_REDLICH_PETERSON",
            "fields": {
                "lss": "LSS",
                "k": "k",
                "k_error": K_ERROR,
                "a": "a",
                "a_error": "a error",
                "beta": "beta",
                "beta_error": "beta error",
            },
        },
        "JOVANOVIC": {
            "prefix": "Jovanovic",
            "table": "ADSORPTION_JOVANOVIC",
            "fields": {
                "lss": "LSS",
                "k": "k",
                "k_error": K_ERROR,
                "qsat": "qsat",
                "qsat_error": "qsat error",
            },
        },
    }

    def __init__(self, queries: DataRepositoryQueries | None = None) -> None:
        self.queries = queries or DataRepositoryQueries()
    
    # -------------------------------------------------------------------------
    def save_raw_dataset(self, dataset: pd.DataFrame) -> None:
        self.queries.save_table(dataset, "ADSORPTION_DATA")

    # -------------------------------------------------------------------------
    def load_raw_dataset(self) -> pd.DataFrame:
        return self.queries.load_table("ADSORPTION_DATA")

    # -------------------------------------------------------------------------
    def save_processed_dataset(self, dataset: pd.DataFrame) -> None:
        self.queries.save_table(dataset, "ADSORPTION_PROCESSED_DATA")

    # -------------------------------------------------------------------------
    def load_processed_dataset(self) -> pd.DataFrame:
        encoded = self.queries.load_table("ADSORPTION_PROCESSED_DATA")
        return encoded

    # -------------------------------------------------------------------------
    def save_fitting_results(self, dataset: pd.DataFrame) -> None:
        if dataset.empty:
            empty_experiments = pd.DataFrame(
                columns=["id", *self.experiment_columns]
            )
            self.queries.save_table(empty_experiments, self.experiment_table)
            for schema in self.model_schemas.values():
                empty_model = pd.DataFrame(
                    columns=["id", "experiment_id", *schema["fields"].values()]
                )
                self.queries.save_table(empty_model, schema["table"])
            return
        encoded = self.convert_lists_to_strings(dataset)
        experiments = self.build_experiment_frame(encoded)
        experiment_map = self.build_experiment_map(experiments)
        self.queries.save_table(experiments, self.experiment_table)
        for schema in self.model_schemas.values():
            model_frame = self.build_model_frame(encoded, experiment_map, schema)
            if model_frame is None:
                continue
            self.queries.save_table(model_frame, schema["table"])

    # -------------------------------------------------------------------------
    def load_fitting_results(self) -> pd.DataFrame:
        experiments = self.queries.load_table(self.experiment_table)
        if experiments.empty:
            return experiments
        experiments = experiments.rename(columns={"id": "experiment_id"})
        experiments = self.convert_strings_to_lists(experiments)
        combined = experiments.copy()
        for schema in self.model_schemas.values():
            model_frame = self.queries.load_table(schema["table"])
            if model_frame.empty:
                continue
            renamed = self.rename_model_columns(model_frame, schema)
            combined = combined.merge(renamed, how="left", on="experiment_id")
        return combined

    # -------------------------------------------------------------------------
    def save_best_fit(self, dataset: pd.DataFrame) -> None:
        if dataset.empty:
            empty_best = pd.DataFrame(
                columns=["id", "experiment_id", "best model", "worst model"]
            )
            self.queries.save_table(empty_best, self.best_fit_table)
            return
        experiments = self.queries.load_table(self.experiment_table)
        if experiments.empty:
            raise ValueError("No experiments available to link best fit results.")
        experiment_map = self.build_experiment_map(experiments)
        best = pd.DataFrame()
        best["experiment_id"] = dataset["experiment"].map(experiment_map)
        if best["experiment_id"].isnull().any():
            raise ValueError("Unmapped experiments found while saving best fit results.")
        best["best model"] = dataset.get("best model")
        best["worst model"] = dataset.get("worst model")
        best.insert(0, "id", range(1, len(best) + 1))
        self.queries.save_table(best, self.best_fit_table)

    # -------------------------------------------------------------------------
    def load_best_fit(self) -> pd.DataFrame:
        best = self.queries.load_table(self.best_fit_table)
        if best.empty:
            return best
        experiments = self.queries.load_table(self.experiment_table)
        if experiments.empty:
            return pd.DataFrame()
        experiments = experiments.rename(columns={"id": "experiment_id"})
        experiments = self.convert_strings_to_lists(experiments)
        merged = experiments.merge(
            best.drop(columns=["id"]), how="left", on="experiment_id"
        )
        return merged

    # -------------------------------------------------------------------------
    def build_experiment_frame(self, dataset: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.experiment_columns if column not in dataset]
        if missing:
            raise ValueError(f"Missing experiment columns: {missing}")
        experiments = dataset.loc[:, self.experiment_columns].copy()
        experiments.insert(0, "id", range(1, len(experiments) + 1))
        return experiments

    # -------------------------------------------------------------------------
    def build_experiment_map(self, experiments: pd.DataFrame) -> dict[str, int]:
        return {
            name: int(identifier)
            for name, identifier in zip(
                experiments["experiment"], experiments["id"], strict=False
            )
        }

    # -------------------------------------------------------------------------
    def resolve_dataset_column(
        self, prefix: str, suffix: str, columns: list[str] | pd.Index
    ) -> str | None:
        target = f"{prefix} {suffix}".lower()
        for column in columns:
            if str(column).lower() == target:
                return column
        return None

    # -------------------------------------------------------------------------
    def build_model_frame(
        self,
        dataset: pd.DataFrame,
        experiment_map: dict[str, int],
        schema: dict[str, Any],
    ) -> pd.DataFrame | None:
        resolved = {
            field: self.resolve_dataset_column(schema["prefix"], suffix, dataset.columns)
            for field, suffix in schema["fields"].items()
        }
        if all(column is None for column in resolved.values()):
            return None
        model_frame = pd.DataFrame()
        model_frame["experiment_id"] = dataset["experiment"].map(experiment_map)
        if model_frame["experiment_id"].isnull().any():
            raise ValueError("Unmapped experiments found while building model results.")
        for field, column in resolved.items():
            target = schema["fields"][field]
            if column is None:
                model_frame[target] = pd.NA
            else:
                model_frame[target] = dataset[column]
        model_frame.insert(0, "id", range(1, len(model_frame) + 1))
        return model_frame

    # -------------------------------------------------------------------------
    def rename_model_columns(
        self, model_frame: pd.DataFrame, schema: dict[str, Any]
    ) -> pd.DataFrame:
        rename_map = {
            column_name: f"{schema['prefix']} {column_name}"
            for column_name in schema["fields"].values()
        }
        trimmed = model_frame.rename(columns=rename_map)
        return trimmed.drop(columns=["id"])

    # -------------------------------------------------------------------------
    def convert_list_to_string(self, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            parts: list[str] = []
            for element in value:
                if element is None:
                    continue
                text = str(element)
                if text:
                    parts.append(text)
            return ",".join(parts)
        return value

    # -------------------------------------------------------------------------
    def convert_string_to_list(self, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            parts = [segment.strip() for segment in stripped.split(",")]
            converted: list[float] = []
            for part in parts:
                if not part:
                    continue
                try:
                    converted.append(float(part))
                except ValueError:
                    return value
            return converted
        return value

    # -------------------------------------------------------------------------
    def convert_lists_to_strings(self, dataset: pd.DataFrame) -> pd.DataFrame:
        converted = dataset.copy()
        for column in converted.columns:
            converted[column] = converted[column].apply(self.convert_list_to_string)
        return converted

    # -------------------------------------------------------------------------
    def convert_strings_to_lists(self, dataset: pd.DataFrame) -> pd.DataFrame:
        converted = dataset.copy()
        for column in converted.columns:
            if converted[column].dtype == object:
                converted[column] = converted[column].apply(self.convert_string_to_list)
        return converted


###############################################################################
class DatasetSerializer:
    def __init__(self, queries: DataRepositoryQueries | None = None) -> None:
        self.queries = queries or DataRepositoryQueries()
        self.dataset_dimension_table = Dataset.__tablename__
        self.dataset_table = DatasetDocument.__tablename__
        self.stats_table = DatasetDocumentStatistics.__tablename__
        self.reports_table = DatasetReport.__tablename__
        self.validation_reports_table = DatasetValidationReport.__tablename__
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
    ) -> Iterator[list[dict[str, Any]]]:
        offset = 0
        while True:
            query = sqlalchemy.text(
                'SELECT dd."id", dd."text" FROM "dataset_document" dd '
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
        query = sqlalchemy.text(
            'SELECT dvr.*, d."name" AS "dataset_name" '
            'FROM "dataset_validation_report" dvr '
            'JOIN "dataset" d ON d."id" = dvr."dataset_id" '
            'WHERE d."name" = :dataset '
            'ORDER BY dvr."id" DESC LIMIT 1'
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
        return self.build_validation_report_response(storage)

    # -------------------------------------------------------------------------
    def load_dataset_validation_report_by_id(
        self,
        report_id: int,
    ) -> dict[str, Any] | None:
        query = sqlalchemy.text(
            'SELECT dvr.*, d."name" AS "dataset_name" '
            'FROM "dataset_validation_report" dvr '
            'JOIN "dataset" d ON d."id" = dvr."dataset_id" '
            'WHERE dvr."id" = :report_id LIMIT 1'
        )
        with self.queries.engine.connect() as conn:
            result = conn.execute(query, {"report_id": int(report_id)})
            row = result.first()

        if row is None:
            return None
        if hasattr(row, "_mapping"):
            storage = dict(row._mapping)
        else:
            storage = {key: value for key, value in zip(result.keys(), row, strict=False)}
        return self.build_validation_report_response(storage)

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
        latest = self.load_latest_dataset_validation_report(dataset_name)
        if latest is not None:
            return latest
        return self.load_legacy_dataset_report(dataset_name)


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
