from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pandas as pd
import sqlalchemy

from TKBEN.server.repositories.database import database
from TKBEN.server.repositories.schema import (
    TextDataset,
    TextDatasetReports,
    TextDatasetStatistics,
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
    
    # -------------------------------------------------------------------------
    def save_raw_dataset(self, dataset: pd.DataFrame) -> None:
        database.save_into_database(dataset, "ADSORPTION_DATA")

    # -------------------------------------------------------------------------
    def load_raw_dataset(self) -> pd.DataFrame:
        return database.load_from_database("ADSORPTION_DATA")

    # -------------------------------------------------------------------------
    def save_processed_dataset(self, dataset: pd.DataFrame) -> None:
        database.save_into_database(dataset, "ADSORPTION_PROCESSED_DATA")

    # -------------------------------------------------------------------------
    def load_processed_dataset(self) -> pd.DataFrame:
        encoded = database.load_from_database("ADSORPTION_PROCESSED_DATA")
        return encoded

    # -------------------------------------------------------------------------
    def save_fitting_results(self, dataset: pd.DataFrame) -> None:
        if dataset.empty:
            empty_experiments = pd.DataFrame(
                columns=["id", *self.experiment_columns]
            )
            database.save_into_database(empty_experiments, self.experiment_table)
            for schema in self.model_schemas.values():
                empty_model = pd.DataFrame(
                    columns=["id", "experiment_id", *schema["fields"].values()]
                )
                database.save_into_database(empty_model, schema["table"])
            return
        encoded = self.convert_lists_to_strings(dataset)
        experiments = self.build_experiment_frame(encoded)
        experiment_map = self.build_experiment_map(experiments)
        database.save_into_database(experiments, self.experiment_table)
        for schema in self.model_schemas.values():
            model_frame = self.build_model_frame(encoded, experiment_map, schema)
            if model_frame is None:
                continue
            database.save_into_database(model_frame, schema["table"])

    # -------------------------------------------------------------------------
    def load_fitting_results(self) -> pd.DataFrame:
        experiments = database.load_from_database(self.experiment_table)
        if experiments.empty:
            return experiments
        experiments = experiments.rename(columns={"id": "experiment_id"})
        experiments = self.convert_strings_to_lists(experiments)
        combined = experiments.copy()
        for schema in self.model_schemas.values():
            model_frame = database.load_from_database(schema["table"])
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
            database.save_into_database(empty_best, self.best_fit_table)
            return
        experiments = database.load_from_database(self.experiment_table)
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
        database.save_into_database(best, self.best_fit_table)

    # -------------------------------------------------------------------------
    def load_best_fit(self) -> pd.DataFrame:
        best = database.load_from_database(self.best_fit_table)
        if best.empty:
            return best
        experiments = database.load_from_database(self.experiment_table)
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
    def __init__(self) -> None:
        self.dataset_table = TextDataset.__tablename__
        self.stats_table = TextDatasetStatistics.__tablename__
        self.reports_table = TextDatasetReports.__tablename__

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
            'SELECT "dataset_name" as dataset_name, '
            'COUNT(*) as document_count '
            'FROM "TEXT_DATASET" GROUP BY "dataset_name" ORDER BY "dataset_name"'
        )
        with database.backend.engine.connect() as conn:
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
        return database.backend.get_distinct_values(self.dataset_table, "dataset_name")

    # -------------------------------------------------------------------------
    def dataset_exists(self, dataset_name: str) -> bool:
        query = sqlalchemy.text(
            'SELECT 1 FROM "TEXT_DATASET" WHERE "dataset_name" = :dataset LIMIT 1'
        )
        with database.backend.engine.connect() as conn:
            result = conn.execute(query, {"dataset": dataset_name})
            return result.first() is not None

    # -------------------------------------------------------------------------
    def count_dataset_documents(self, dataset_name: str) -> int:
        query = sqlalchemy.text(
            'SELECT COUNT(*) FROM "TEXT_DATASET" WHERE "dataset_name" = :dataset'
        )
        with database.backend.engine.connect() as conn:
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
                'SELECT "text" FROM "TEXT_DATASET" '
                'WHERE "dataset_name" = :dataset '
                "LIMIT :limit OFFSET :offset"
            )
            with database.backend.engine.connect() as conn:
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
    def delete_dataset(self, dataset_name: str) -> None:
        database.delete_by_key(self.dataset_table, "dataset_name", dataset_name)
        database.delete_by_key(self.stats_table, "dataset_name", dataset_name)
        database.delete_by_key(self.reports_table, "dataset_name", dataset_name)

    # -------------------------------------------------------------------------
    def delete_dataset_statistics(self, dataset_name: str) -> None:
        database.delete_by_key(self.stats_table, "dataset_name", dataset_name)

    # -------------------------------------------------------------------------
    def save_dataset_statistics_batch(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        df = pd.DataFrame(batch)
        database.insert_dataframe(df, self.stats_table)

    # -------------------------------------------------------------------------
    def build_report_storage(self, report: dict[str, Any]) -> dict[str, Any]:
        document_histogram = report.get("document_length_histogram", {})
        word_histogram = report.get("word_length_histogram", {})
        return {
            "dataset_name": report.get("dataset_name", ""),
            "document_count": report.get("document_count", 0),
            "document_bins": document_histogram.get("bins", []),
            "document_counts": document_histogram.get("counts", []),
            "document_bin_edges": document_histogram.get("bin_edges", []),
            "document_min_length": document_histogram.get("min_length", 0),
            "document_max_length": document_histogram.get("max_length", 0),
            "document_mean_length": document_histogram.get("mean_length", 0.0),
            "document_median_length": document_histogram.get("median_length", 0.0),
            "word_bins": word_histogram.get("bins", []),
            "word_counts": word_histogram.get("counts", []),
            "word_bin_edges": word_histogram.get("bin_edges", []),
            "word_min_length": word_histogram.get("min_length", 0),
            "word_max_length": word_histogram.get("max_length", 0),
            "word_mean_length": word_histogram.get("mean_length", 0.0),
            "word_median_length": word_histogram.get("median_length", 0.0),
            "most_common_words": report.get("most_common_words", []),
            "least_common_words": report.get("least_common_words", []),
        }

    # -------------------------------------------------------------------------
    def save_dataset_report(self, report: dict[str, Any]) -> None:
        storage = self.build_report_storage(report)
        df = pd.DataFrame([storage])
        df = self.serialize_json_columns(df)
        database.bulk_replace_by_key(df, self.reports_table, "dataset_name", storage["dataset_name"])

    # -------------------------------------------------------------------------
    def build_report_response(self, storage: dict[str, Any]) -> dict[str, Any]:
        document_histogram = {
            "bins": self.parse_json(storage.get("document_bins"), default=[]),
            "counts": self.parse_json(storage.get("document_counts"), default=[]),
            "bin_edges": self.parse_json(storage.get("document_bin_edges"), default=[]),
            "min_length": int(storage.get("document_min_length", 0) or 0),
            "max_length": int(storage.get("document_max_length", 0) or 0),
            "mean_length": float(storage.get("document_mean_length", 0.0) or 0.0),
            "median_length": float(storage.get("document_median_length", 0.0) or 0.0),
        }
        word_histogram = {
            "bins": self.parse_json(storage.get("word_bins"), default=[]),
            "counts": self.parse_json(storage.get("word_counts"), default=[]),
            "bin_edges": self.parse_json(storage.get("word_bin_edges"), default=[]),
            "min_length": int(storage.get("word_min_length", 0) or 0),
            "max_length": int(storage.get("word_max_length", 0) or 0),
            "mean_length": float(storage.get("word_mean_length", 0.0) or 0.0),
            "median_length": float(storage.get("word_median_length", 0.0) or 0.0),
        }
        return {
            "dataset_name": storage.get("dataset_name", ""),
            "document_count": int(storage.get("document_count", 0) or 0),
            "document_length_histogram": document_histogram,
            "word_length_histogram": word_histogram,
            "min_document_length": document_histogram["min_length"],
            "max_document_length": document_histogram["max_length"],
            "most_common_words": self.parse_json(storage.get("most_common_words"), default=[]),
            "least_common_words": self.parse_json(storage.get("least_common_words"), default=[]),
        }

    # -------------------------------------------------------------------------
    def load_dataset_report(self, dataset_name: str) -> dict[str, Any] | None:
        query = sqlalchemy.text(
            'SELECT * FROM "TEXT_DATASET_REPORTS" '
            'WHERE "dataset_name" = :dataset LIMIT 1'
        )
        with database.backend.engine.connect() as conn:
            result = conn.execute(query, {"dataset": dataset_name})
            row = result.first()

        if row is None:
            return None

        if hasattr(row, "_mapping"):
            storage = dict(row._mapping)
        else:
            storage = {key: value for key, value in zip(result.keys(), row, strict=False)}

        return self.build_report_response(storage)
