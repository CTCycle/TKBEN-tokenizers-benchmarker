from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from TKBEN.server.repositories.database.backend import database
from TKBEN.server.repositories.schemas.models import BenchmarkReport, Dataset, DatasetDocument, Tokenizer


###############################################################################
class BenchmarkRepository:
    def _session(self) -> Session:
        return Session(bind=database.backend.engine)

    # -------------------------------------------------------------------------
    def get_dataset_document_count(self, dataset_name: str) -> int:
        stmt = (
            select(func.count(DatasetDocument.id))
            .join(Dataset, Dataset.id == DatasetDocument.dataset_id)
            .where(Dataset.name == dataset_name)
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
    def list_benchmark_reports(self, limit: int = 200) -> list[tuple[BenchmarkReport, str]]:
        capped_limit = max(1, min(1000, int(limit or 200)))
        stmt = (
            select(BenchmarkReport, Dataset.name.label("dataset_name"))
            .join(Dataset, Dataset.id == BenchmarkReport.dataset_id)
            .order_by(BenchmarkReport.id.desc())
            .limit(capped_limit)
        )
        with self._session() as session:
            return list(session.execute(stmt).all())

    # -------------------------------------------------------------------------
    def get_benchmark_report_by_id(self, report_id: int) -> tuple[BenchmarkReport, str] | None:
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
        return row

    # -------------------------------------------------------------------------
    def get_dataset_id(self, dataset_name: str) -> int | None:
        stmt = select(Dataset.id).where(Dataset.name == dataset_name).limit(1)
        with self._session() as session:
            dataset_id = session.execute(stmt).scalar_one_or_none()
        return int(dataset_id) if dataset_id is not None else None

    # -------------------------------------------------------------------------
    def save_benchmark_report(
        self,
        dataset_id: int,
        report_version: int,
        created_at,
        run_name: str | None,
        selected_metric_keys: list[str],
        payload: dict,
    ) -> int:
        report_row = BenchmarkReport(
            dataset_id=int(dataset_id),
            report_version=int(report_version),
            created_at=created_at,
            run_name=run_name,
            selected_metric_keys=selected_metric_keys,
            payload=payload,
        )
        with self._session() as session:
            session.add(report_row)
            session.commit()
            session.refresh(report_row)
        if report_row.id is None:
            raise ValueError("Failed to resolve saved benchmark report id.")
        return int(report_row.id)
