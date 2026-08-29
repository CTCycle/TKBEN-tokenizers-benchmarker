from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from server.contracts.benchmarks import BenchmarkReportSort
from server.repositories.database.backend import TKBENDatabase, get_database
from server.repositories.schemas.models import (
    BenchmarkReport,
    Dataset,
    DatasetDocument,
    Tokenizer,
)

###############################################################################
@dataclass(frozen=True)
class BenchmarkReportPage:
    rows: list[dict[str, Any]]
    total: int
    offset: int
    limit: int

###############################################################################
class BenchmarkRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: TKBENDatabase | None = None) -> None:
        self.database = database or get_database()

    # -------------------------------------------------------------------------
    def _session(self) -> Session:
        return Session(bind=self.database.backend.engine)

    # -------------------------------------------------------------------------
    def get_dataset_document_count(self, dataset_name: str) -> int:
        stmt = (
            select(func.count(DatasetDocument.id))
            .join(Dataset, Dataset.id == DatasetDocument.dataset_id)
            .where(Dataset.name == dataset_name, Dataset.status == "ready")
        )
        with self._session() as session:
            value = session.execute(stmt).scalar_one_or_none() or 0
        return int(value)

    # -------------------------------------------------------------------------
    def get_missing_persisted_tokenizers(self, tokenizer_ids: list[str]) -> list[str]:
        if not tokenizer_ids:
            return []
        unique_requested = list(dict.fromkeys(tokenizer_ids))
        with self._session() as session:
            persisted_names = set(
                session.execute(
                    select(Tokenizer.name).where(Tokenizer.name.in_(unique_requested))
                ).scalars()
            )
        return [name for name in unique_requested if name not in persisted_names]

    # -------------------------------------------------------------------------
    def get_tokenizer_sources(self, tokenizer_ids: list[str]) -> dict[str, str]:
        if not tokenizer_ids:
            return {}
        with self._session() as session:
            rows = session.execute(
                select(Tokenizer.name, Tokenizer.source).where(
                    Tokenizer.name.in_(list(dict.fromkeys(tokenizer_ids)))
                )
            ).all()
        return {str(name): str(source) for name, source in rows}

    # -------------------------------------------------------------------------
    def list_benchmark_reports(
        self,
        *,
        search: str | None,
        sort: BenchmarkReportSort,
        offset: int,
        limit: int,
    ) -> BenchmarkReportPage:
        safe_offset = max(0, int(offset))
        safe_limit = max(1, min(100, int(limit)))
        filters = []
        if search:
            pattern = f"%{search}%"
            filters.append(
                or_(
                    BenchmarkReport.run_name.ilike(pattern),
                    Dataset.name.ilike(pattern),
                )
            )

        total_stmt = (
            select(func.count(BenchmarkReport.id))
            .join(Dataset, Dataset.id == BenchmarkReport.dataset_id)
            .where(*filters)
        )
        order = (
            (BenchmarkReport.created_at.asc(), BenchmarkReport.id.asc())
            if sort == BenchmarkReportSort.OLDEST
            else (BenchmarkReport.created_at.desc(), BenchmarkReport.id.desc())
        )
        stmt = (
            select(
                BenchmarkReport.id,
                BenchmarkReport.report_version,
                BenchmarkReport.schema_version,
                BenchmarkReport.created_at,
                BenchmarkReport.run_name,
                BenchmarkReport.documents_processed,
                BenchmarkReport.tokenizers_count,
                BenchmarkReport.tokenizers_processed,
                BenchmarkReport.selected_metric_keys,
                Dataset.name.label("dataset_name"),
            )
            .join(Dataset, Dataset.id == BenchmarkReport.dataset_id)
            .where(*filters)
            .order_by(*order)
            .offset(safe_offset)
            .limit(safe_limit)
        )
        with self._session() as session:
            total = int(session.execute(total_stmt).scalar_one() or 0)
            rows = [dict(row) for row in session.execute(stmt).mappings().all()]
        return BenchmarkReportPage(
            rows=rows,
            total=total,
            offset=safe_offset,
            limit=safe_limit,
        )

    # -------------------------------------------------------------------------
    def delete_benchmark_report(self, report_id: int) -> bool:
        with self._session() as session:
            row = session.execute(
                select(BenchmarkReport)
                .where(BenchmarkReport.id == int(report_id))
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return False
            session.delete(row)
            session.commit()
        return True

    # -------------------------------------------------------------------------
    def get_benchmark_report_by_id(
        self, report_id: int
    ) -> tuple[BenchmarkReport, str] | None:
        stmt = (
            select(BenchmarkReport, Dataset.name.label("dataset_name"))
            .join(Dataset, Dataset.id == BenchmarkReport.dataset_id)
            .where(BenchmarkReport.id == int(report_id))
            .limit(1)
        )
        with self._session() as session:
            row = session.execute(stmt).first()
        if row is None or row[0] is None:
            return None
        return row[0], str(row[1])

    # -------------------------------------------------------------------------
    def get_dataset_id(self, dataset_name: str) -> int | None:
        stmt = select(Dataset.id).where(Dataset.name == dataset_name).limit(1)
        with self._session() as session:
            dataset_id = session.execute(stmt).scalar_one_or_none()
        return int(dataset_id) if dataset_id is not None else None

    # -------------------------------------------------------------------------
    def get_tokenizer_ids(self, tokenizer_names: list[str]) -> dict[str, int]:
        if not tokenizer_names:
            return {}
        deduped_names = list(dict.fromkeys(tokenizer_names))
        with self._session() as session:
            mapping_rows = session.execute(
                select(Tokenizer.id, Tokenizer.name).where(
                    Tokenizer.name.in_(deduped_names)
                )
            ).all()
        mapping = {str(name): int(tokenizer_id) for tokenizer_id, name in mapping_rows}
        missing = [name for name in deduped_names if name not in mapping]
        if missing:
            raise ValueError(
                "Tokenizer records do not exist: " + ", ".join(missing)
            )
        return mapping

    # -------------------------------------------------------------------------
    def save_benchmark_report(
        self,
        dataset_id: int,
        report_version: int,
        created_at,
        run_name: str | None,
        selected_metric_keys: list[str],
        payload: dict[str, Any],
    ) -> int:
        required_fields = (
            "schema_version",
            "methodology_version",
            "status",
            "documents_processed",
            "tokenizers_processed",
        )
        if any(field not in payload for field in required_fields):
            raise ValueError("Benchmark report payload is missing canonical fields.")

        schema_version = int(payload["schema_version"])
        methodology_version = str(payload["methodology_version"])
        status = str(payload["status"])
        documents_processed = int(payload["documents_processed"])
        tokenizers_processed = payload["tokenizers_processed"]
        if not isinstance(tokenizers_processed, list):
            raise ValueError("Benchmark report tokenizers_processed must be a list.")
        tokenizers_processed = [str(name) for name in tokenizers_processed]
        payload_tokenizer_count = payload.get("tokenizers_count")
        if payload_tokenizer_count is not None and int(payload_tokenizer_count) != len(tokenizers_processed):
            raise ValueError("Benchmark report tokenizer count disagrees with its list.")
        payload_selected_metric_keys = payload.get("selected_metric_keys")
        if payload_selected_metric_keys is not None:
            if not isinstance(payload_selected_metric_keys, list):
                raise ValueError("Benchmark report selected_metric_keys must be a list.")
            if [str(key) for key in payload_selected_metric_keys] != selected_metric_keys:
                raise ValueError("Benchmark report selected metrics disagree with its summary.")
        detail_fields = {
            "report_id",
            "report_version",
            "schema_version",
            "methodology_version",
            "created_at",
            "run_name",
            "status",
            "documents_processed",
            "tokenizers_count",
            "tokenizers_processed",
            "selected_metric_keys",
            "dataset_name",
        }
        details_payload = {
            key: value for key, value in payload.items() if key not in detail_fields
        }
        report_row = BenchmarkReport(
            dataset_id=int(dataset_id),
            report_version=int(report_version),
            created_at=created_at,
            run_name=run_name,
            selected_metric_keys=selected_metric_keys,
            schema_version=schema_version,
            methodology_version=methodology_version,
            status=status,
            documents_processed=documents_processed,
            tokenizers_count=len(tokenizers_processed),
            tokenizers_processed=tokenizers_processed,
            payload=details_payload,
        )
        with self._session() as session:
            session.add(report_row)
            session.commit()
            session.refresh(report_row)
        if report_row.id is None:
            raise ValueError("Failed to resolve saved benchmark report id.")
        return int(report_row.id)
