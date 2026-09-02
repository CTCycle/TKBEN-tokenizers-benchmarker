from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from server.common.tokenizer_metrics import compute_vocabulary_shape_metrics
from server.repositories.queries.data import DataRepositoryQueries
from server.repositories.schemas.models import (
    Tokenizer,
    TokenizerReport,
    TokenizerVocabulary,
)


###############################################################################
def _parse_timestamp(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(cast(Any, value), utc=True, errors="coerce")
    return parsed if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed) else None


###############################################################################
class TokenizerReportRepository:
    REPORT_VERSION = 1

    # -------------------------------------------------------------------------
    def __init__(self, queries: DataRepositoryQueries | None = None) -> None:
        self.queries = queries or DataRepositoryQueries()
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
    def replace_report_and_vocabulary(
        self,
        tokenizer_name: str,
        report: dict[str, Any],
        vocabulary_rows: list[dict[str, Any]],
    ) -> int:
        name = str(tokenizer_name).strip()
        if not name:
            raise ValueError("Tokenizer name must be provided")
        now = datetime.now(timezone.utc)
        with self._session() as session:
            tokenizer_id = session.execute(
                select(Tokenizer.id).where(Tokenizer.name == name)
            ).scalar_one_or_none()
            if tokenizer_id is None:
                raise ValueError(
                    f"Tokenizer '{name}' must exist before storing a report."
                )
            session.execute(
                delete(TokenizerVocabulary).where(
                    TokenizerVocabulary.tokenizer_id == int(tokenizer_id)
                )
            )
            records = [
                {
                    "tokenizer_id": int(tokenizer_id),
                    "token_id": int(row["token_id"]),
                    "token": str(row.get("token", "")),
                    "decoded_token": row.get("decoded_token"),
                }
                for row in vocabulary_rows
            ]
            for start in range(0, len(records), 1000):
                session.execute(
                    cast(Any, TokenizerVocabulary.__table__).insert(),
                    records[start : start + 1000],
                )
            session.execute(
                delete(TokenizerReport).where(
                    TokenizerReport.tokenizer_id == int(tokenizer_id)
                )
            )
            vocabulary_shape_metrics = compute_vocabulary_shape_metrics(vocabulary_rows)
            global_stats = report.get("global_stats", {})
            metadata_payload = (
                dict(global_stats) if isinstance(global_stats, dict) else {}
            )
            vocabulary_stats = metadata_payload.get("vocabulary_stats")
            persisted_vocabulary_stats = (
                dict(vocabulary_stats) if isinstance(vocabulary_stats, dict) else {}
            )
            persisted_vocabulary_stats.update(vocabulary_shape_metrics)
            metadata_payload["vocabulary_stats"] = persisted_vocabulary_stats
            metadata_payload.setdefault(
                "huggingface_url", report.get("huggingface_url")
            )
            report_histogram = report.get("token_length_histogram", {})
            histogram_payload = (
                dict(report_histogram) if isinstance(report_histogram, dict) else {}
            )
            histogram_payload.update(vocabulary_shape_metrics)
            report_row = TokenizerReport(
                tokenizer_id=int(tokenizer_id),
                report_version=int(report["report_version"]),
                created_at=now,
                metadata_json=metadata_payload,
                token_length_histogram=histogram_payload,
                description=report.get("description"),
            )
            session.add(report_row)
            session.commit()
            session.refresh(report_row)
            return int(report_row.id)

    # -------------------------------------------------------------------------
    def _build_tokenizer_report_response(
        self, storage: dict[str, Any]
    ) -> dict[str, Any]:
        report_version = int(storage["report_version"])
        if report_version != self.REPORT_VERSION:
            raise ValueError(
                "Tokenizer report uses incompatible report version "
                f"{report_version}; expected {self.REPORT_VERSION}."
            )
        created_at = _parse_timestamp(storage.get("created_at"))
        created_at_iso = (
            created_at.isoformat().replace("+00:00", "Z")
            if created_at is not None
            else ""
        )
        metadata = storage.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("Tokenizer report metadata must be a native JSON object.")
        metadata_payload = dict(metadata)
        histogram = storage.get("token_length_histogram", {})
        if histogram is None:
            histogram = {}
        if not isinstance(histogram, dict):
            raise ValueError("Tokenizer report histogram must be a native JSON object.")
        histogram_payload = {
            "bins": list(histogram.get("bins", [])),
            "counts": list(histogram.get("counts", [])),
            "bin_edges": list(histogram.get("bin_edges", [])),
            "min_length": int(histogram.get("min_length", 0) or 0),
            "max_length": int(histogram.get("max_length", 0) or 0),
            "mean_length": float(histogram.get("mean_length", 0.0) or 0.0),
            "median_length": float(histogram.get("median_length", 0.0) or 0.0),
            "token_length_std": float(histogram.get("token_length_std", 0.0) or 0.0),
            "token_length_p90": float(histogram.get("token_length_p90", 0.0) or 0.0),
            "token_length_cv": float(histogram.get("token_length_cv", 0.0) or 0.0),
            "single_character_token_percentage": float(
                histogram.get("single_character_token_percentage", 0.0) or 0.0
            ),
        }
        huggingface_url = metadata_payload.pop("huggingface_url", None)
        if not isinstance(huggingface_url, str) or not huggingface_url.strip():
            huggingface_url = None
        return {
            "report_id": int(storage.get("id") or 0),
            "report_version": report_version,
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
            select(TokenizerVocabulary.token_id, TokenizerVocabulary.token)
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
