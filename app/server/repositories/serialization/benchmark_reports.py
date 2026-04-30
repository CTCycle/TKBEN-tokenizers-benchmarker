from __future__ import annotations

import json
from typing import Any

import pandas as pd

from TKBEN.server.domain.benchmarks import BenchmarkReportSummary, BenchmarkRunResponse
from TKBEN.server.repositories.benchmarks import BenchmarkRepository
from TKBEN.server.repositories.queries.data import DataRepositoryQueries
from TKBEN.server.repositories.schemas.models import BenchmarkReport


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
        normalized_payload["report_id"] = int(
            row.get("id") or normalized_payload.get("report_id") or 0
        )
        normalized_payload["report_version"] = int(
            row.get("report_version") or normalized_payload.get("report_version") or 2
        )
        normalized_payload["created_at"] = created_at_iso
        normalized_payload["run_name"] = row.get("run_name") or normalized_payload.get(
            "run_name"
        )
        normalized_payload["selected_metric_keys"] = selected_metric_keys
        normalized_payload["dataset_name"] = str(
            normalized_payload.get("dataset_name") or row.get("dataset_name") or ""
        )

        return BenchmarkRunResponse.model_validate(normalized_payload).model_dump(
            mode="json"
        )

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
            summaries.append(
                BenchmarkReportSummary.model_validate(normalized).model_dump(mode="json")
            )
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
