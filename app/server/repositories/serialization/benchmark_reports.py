from __future__ import annotations

from typing import Any, cast

import pandas as pd

from server.contracts.benchmarks import (
    BenchmarkReportListResponse,
    BenchmarkReportQuery,
    BenchmarkReportSummary,
    BenchmarkRunResponse,
)
from server.common.utils.logger import logger
from server.repositories.benchmarks import BenchmarkRepository
from server.common.constants import BENCHMARK_REPORT_VERSION, BENCHMARK_SCHEMA_VERSION

###############################################################################
def _parse_timestamp(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(cast(Any, value), utc=True, errors="coerce")
    return parsed if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed) else None

###############################################################################
class BenchmarkReportSerializer:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.repository = BenchmarkRepository()

    # -------------------------------------------------------------------------
    def save_benchmark_report(self, report_payload: dict[str, Any]) -> int:
        self._validate_current_contract(report_payload)
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

        created_at = _parse_timestamp(report_payload.get("created_at")) or pd.Timestamp.utcnow()
        created_at_value = created_at.to_pydatetime()

        run_name = report_payload.get("run_name")
        if isinstance(run_name, str):
            run_name = run_name.strip() or None
        else:
            run_name = None

        return self.repository.save_benchmark_report(
            dataset_id=int(dataset_id),
            report_version=BENCHMARK_REPORT_VERSION,
            created_at=created_at_value,
            run_name=run_name,
            selected_metric_keys=selected_metric_keys,
            payload=report_payload,
        )

    # -------------------------------------------------------------------------
    def _validate_current_contract(self, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
            raise ValueError("Benchmark report uses an incompatible schema version.")
        if "methodology_version" not in payload:
            raise ValueError(
                "Benchmark report is missing required methodology_version."
            )
        if payload.get("report_version") != BENCHMARK_REPORT_VERSION:
            raise ValueError("Benchmark report uses an incompatible report version.")

    # -------------------------------------------------------------------------
    def _normalize_report_row(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("Benchmark report payload must be a native JSON object.")
        created_at = _parse_timestamp(row.get("created_at"))
        created_at_iso = (
            created_at.isoformat().replace("+00:00", "Z")
            if created_at is not None
            else None
        )

        selected_metric_keys = row.get("selected_metric_keys")
        if selected_metric_keys is None:
            selected_metric_keys = []
        if not isinstance(selected_metric_keys, list):
            raise ValueError(
                "Benchmark report selected_metric_keys must be a native JSON array."
            )
        selected_metric_keys = [
            str(key) for key in selected_metric_keys if isinstance(key, str) and key
        ]

        normalized_payload = dict(payload)
        self._validate_current_contract(normalized_payload)
        normalized_payload["report_id"] = int(
            row.get("id") or normalized_payload.get("report_id") or 0
        )
        if int(row.get("report_version") or 0) != BENCHMARK_REPORT_VERSION:
            raise ValueError("Benchmark report uses an incompatible report version.")
        normalized_payload["report_version"] = BENCHMARK_REPORT_VERSION
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
    def list_benchmark_reports(
        self,
        query: BenchmarkReportQuery,
    ) -> BenchmarkReportListResponse:
        page = self.repository.list_benchmark_reports(
            search=query.search,
            sort=query.sort,
            offset=query.offset,
            limit=query.limit,
        )

        summaries: list[BenchmarkReportSummary] = []
        for row in page.rows:
            created_at = _parse_timestamp(row.get("created_at"))
            summaries.append(BenchmarkReportSummary.model_validate({
                "report_id": int(row["id"]),
                "report_version": int(row["report_version"]),
                "created_at": created_at.isoformat().replace("+00:00", "Z") if created_at is not None else None,
                "run_name": row.get("run_name"),
                "dataset_name": str(row["dataset_name"]),
                "documents_processed": int(row["documents_processed"]),
                "tokenizers_count": int(row["tokenizers_count"]),
                "tokenizers_processed": list(row.get("tokenizers_processed") or []),
                "selected_metric_keys": list(row.get("selected_metric_keys") or []),
            }))
        return BenchmarkReportListResponse(
            reports=summaries,
            total=page.total,
            offset=page.offset,
            limit=page.limit,
        )

    # -------------------------------------------------------------------------
    def delete_benchmark_report(self, report_id: int) -> bool:
        return self.repository.delete_benchmark_report(report_id)

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
        try:
            return self._normalize_report_row(mapped)
        except ValueError:
            logger.warning(
                "Benchmark report id=%s is incompatible with current schema",
                report_id,
            )
            return None
