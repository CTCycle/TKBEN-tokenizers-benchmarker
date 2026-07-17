from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, load_only

from server.repositories.database.backend import TKBENDatabase, get_database
from server.repositories.schemas.models import (
    BenchmarkReport,
    Dataset,
    DatasetDocument,
    Tokenizer,
)

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
    def list_benchmark_reports(
        self, limit: int = 200
    ) -> list[tuple[BenchmarkReport, str]]:
        capped_limit = max(1, min(1000, int(limit or 200)))
        stmt = (
            select(BenchmarkReport, Dataset.name.label("dataset_name"))
            .options(load_only(
                BenchmarkReport.id,
                BenchmarkReport.report_version,
                BenchmarkReport.created_at,
                BenchmarkReport.run_name,
                BenchmarkReport.status,
                BenchmarkReport.documents_processed,
                BenchmarkReport.tokenizers_count,
                BenchmarkReport.tokenizers_processed,
                BenchmarkReport.selected_metric_keys,
            ))
            .join(Dataset, Dataset.id == BenchmarkReport.dataset_id)
            .order_by(BenchmarkReport.id.desc())
            .limit(capped_limit)
        )
        with self._session() as session:
            rows = session.execute(stmt).all()
        return [(row[0], str(row[1])) for row in rows]

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
    def ensure_tokenizer_ids(self, tokenizer_names: list[str]) -> dict[str, int]:
        if not tokenizer_names:
            return {}
        deduped_names = list(dict.fromkeys(tokenizer_names))
        with self._session() as session:
            existing_rows = (
                session.execute(
                    select(Tokenizer).where(Tokenizer.name.in_(deduped_names))
                )
                .scalars()
                .all()
            )
            existing_names = {row.name for row in existing_rows}
            for name in deduped_names:
                if name not in existing_names:
                    session.add(Tokenizer(name=name, created_at=datetime.now(timezone.utc)))
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
            mapping_rows = session.execute(
                select(Tokenizer.id, Tokenizer.name).where(
                    Tokenizer.name.in_(deduped_names)
                )
            ).all()
        return {str(name): int(tokenizer_id) for tokenizer_id, name in mapping_rows}

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
        schema_version = int(payload.get("schema_version", 1) or 1)
        methodology_version = str(payload.get("methodology_version", "unknown") or "unknown")
        status = str(payload.get("status", "completed") or "completed")
        documents_processed = int(payload.get("documents_processed", payload.get("document_count", 0)) or 0)
        tokenizers_processed = payload.get("tokenizers_processed", payload.get("tokenizers", []))
        if not isinstance(tokenizers_processed, list):
            tokenizers_processed = []
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
            payload=payload,
        )
        with self._session() as session:
            session.add(report_row)
            session.commit()
            session.refresh(report_row)
        if report_row.id is None:
            raise ValueError("Failed to resolve saved benchmark report id.")
        return int(report_row.id)
