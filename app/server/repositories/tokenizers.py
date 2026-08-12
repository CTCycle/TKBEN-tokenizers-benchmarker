from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from server.repositories.database.backend import TKBENDatabase, get_database
from server.repositories.schemas.models import (
    Tokenizer,
    TokenizerReport,
)

###############################################################################
class TokenizerRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: TKBENDatabase | None = None) -> None:
        self.database = database or get_database()

    # -------------------------------------------------------------------------
    def _session(self) -> Session:
        return Session(bind=self.database.backend.engine)

    # -------------------------------------------------------------------------
    def list_downloaded_tokenizer_catalog(self) -> list[tuple[str, bool, object]]:
        stmt = (
            select(Tokenizer.name, TokenizerReport.id, TokenizerReport.metadata_json)
            .outerjoin(TokenizerReport, TokenizerReport.tokenizer_id == Tokenizer.id)
            .order_by(Tokenizer.name.asc())
        )
        with self._session() as session:
            rows = session.execute(stmt).all()
        return [(str(name), report_id is not None, metadata) for name, report_id, metadata in rows]

    # -------------------------------------------------------------------------
    def tokenizer_exists(self, tokenizer_id: str) -> bool:
        with self._session() as session:
            row = session.execute(
                select(Tokenizer.id).where(Tokenizer.name == tokenizer_id).limit(1)
            ).first()
        return row is not None

    # -------------------------------------------------------------------------
    def insert_if_missing(self, tokenizer_id: str) -> None:
        with self._session() as session:
            existing = session.execute(
                select(Tokenizer.id).where(Tokenizer.name == tokenizer_id).limit(1)
            ).scalar_one_or_none()
            if existing is None:
                session.add(Tokenizer(name=tokenizer_id, created_at=datetime.now(timezone.utc)))
                try:
                    session.commit()
                except IntegrityError:
                    session.rollback()

    # -------------------------------------------------------------------------
    def delete_tokenizer(self, tokenizer_id: str) -> bool:
        with self._session() as session:
            row = session.execute(
                select(Tokenizer).where(Tokenizer.name == tokenizer_id).limit(1)
            ).scalar_one_or_none()
            if row is None:
                return False
            session.delete(row)
            session.commit()
        return True

    # -------------------------------------------------------------------------
    def get_tokenizer_report_by_id(self, report_id: int) -> TokenizerReport | None:
        with self._session() as session:
            return session.execute(
                select(TokenizerReport)
                .where(TokenizerReport.id == int(report_id))
                .limit(1)
            ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    def get_latest_tokenizer_report(
        self, tokenizer_name: str
    ) -> TokenizerReport | None:
        stmt = (
            select(TokenizerReport)
            .join(Tokenizer, Tokenizer.id == TokenizerReport.tokenizer_id)
            .where(Tokenizer.name == tokenizer_name)
            .order_by(TokenizerReport.id.desc())
            .limit(1)
        )
        with self._session() as session:
            return session.execute(stmt).scalar_one_or_none()
