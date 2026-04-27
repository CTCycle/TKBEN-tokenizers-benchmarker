from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from TKBEN.server.repositories.database.backend import TKBENDatabase, get_database
from TKBEN.server.repositories.schemas.models import (
    Tokenizer, 
    TokenizerReport, 
    TokenizerVocabulary
)


###############################################################################
class TokenizerRepository:
    def __init__(self, database: TKBENDatabase | None = None) -> None:
        self.database = database or get_database()

    # -------------------------------------------------------------------------
    def _session(self) -> Session:
        return Session(bind=self.database.backend.engine)

    # -------------------------------------------------------------------------
    def list_downloaded_tokenizers(self) -> list[str]:
        stmt = select(Tokenizer.name).order_by(Tokenizer.name.asc())
        with self._session() as session:
            rows = session.execute(stmt).all()
        return [str(name) for (name,) in rows]

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
                session.add(Tokenizer(name=tokenizer_id))
                try:
                    session.commit()
                except IntegrityError:
                    session.rollback()

    # -------------------------------------------------------------------------
    def get_missing_tokenizers(self, tokenizer_ids: list[str]) -> list[str]:
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
    def get_tokenizer_report_by_id(self, report_id: int) -> TokenizerReport | None:
        with self._session() as session:
            return session.execute(
                select(TokenizerReport)
                .where(TokenizerReport.id == int(report_id))
                .limit(1)
            ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    def get_latest_tokenizer_report(self, tokenizer_name: str) -> TokenizerReport | None:
        stmt = (
            select(TokenizerReport)
            .join(Tokenizer, Tokenizer.id == TokenizerReport.tokenizer_id)
            .where(Tokenizer.name == tokenizer_name)
            .order_by(TokenizerReport.id.desc())
            .limit(1)
        )
        with self._session() as session:
            return session.execute(stmt).scalar_one_or_none()

    # -------------------------------------------------------------------------
    def get_tokenizer_name_by_id(self, tokenizer_id: int) -> str | None:
        with self._session() as session:
            value = session.execute(
                select(Tokenizer.name).where(Tokenizer.id == int(tokenizer_id)).limit(1)
            ).scalar_one_or_none()
        return str(value) if value is not None else None

    # -------------------------------------------------------------------------
    def get_tokenizer_id_by_name(self, tokenizer_name: str) -> int | None:
        with self._session() as session:
            value = session.execute(
                select(Tokenizer.id).where(Tokenizer.name == tokenizer_name).limit(1)
            ).scalar_one_or_none()
        return int(value) if value is not None else None

    # -------------------------------------------------------------------------
    def get_tokenizer_vocabulary_page(
        self,
        tokenizer_id: int,
        offset: int,
        limit: int,
    ) -> tuple[int, list[tuple[int, str]]]:
        count_stmt = select(func.count(TokenizerVocabulary.id)).where(
            TokenizerVocabulary.tokenizer_id == int(tokenizer_id)
        )
        page_stmt = (
            select(TokenizerVocabulary.token_id, TokenizerVocabulary.vocabulary_tokens)
            .where(TokenizerVocabulary.tokenizer_id == int(tokenizer_id))
            .order_by(TokenizerVocabulary.token_id.asc())
            .limit(int(limit))
            .offset(int(offset))
        )
        with self._session() as session:
            total = int(session.execute(count_stmt).scalar_one_or_none() or 0)
            rows = list(session.execute(page_stmt).all())
        items = [(int(token_id), str(token_value or "")) for token_id, token_value in rows]
        return total, items
