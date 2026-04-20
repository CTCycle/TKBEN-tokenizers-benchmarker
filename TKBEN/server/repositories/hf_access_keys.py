from __future__ import annotations

from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from TKBEN.server.repositories.database.backend import database
from TKBEN.server.repositories.schemas.models import HFAccessKey


###############################################################################
class HFAccessKeyRepository:
    def _session(self) -> Session:
        return Session(bind=database.backend.engine)

    # -------------------------------------------------------------------------
    def list_all(self) -> list[HFAccessKey]:
        stmt = select(HFAccessKey).order_by(
            HFAccessKey.created_at.desc(),
            HFAccessKey.id.desc(),
        )
        with self._session() as session:
            return list(session.execute(stmt).scalars().all())

    # -------------------------------------------------------------------------
    def list_encrypted_values(self) -> list[str]:
        with self._session() as session:
            rows = session.execute(select(HFAccessKey.key_value)).all()
        return [str(value) for (value,) in rows if value]

    # -------------------------------------------------------------------------
    def get_by_id(self, key_id: int) -> HFAccessKey | None:
        with self._session() as session:
            return session.execute(
                select(HFAccessKey).where(HFAccessKey.id == int(key_id)).limit(1)
            ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    def get_active(self) -> HFAccessKey | None:
        with self._session() as session:
            return session.execute(
                select(HFAccessKey)
                .where(HFAccessKey.is_active.is_(True))
                .order_by(HFAccessKey.id.desc())
                .limit(1)
            ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    def insert_key(self, encrypted_value: str, created_at: datetime) -> HFAccessKey:
        with self._session() as session:
            key_row = HFAccessKey(
                key_value=encrypted_value,
                created_at=created_at,
                is_active=False,
            )
            session.add(key_row)
            session.commit()
            session.refresh(key_row)
            return key_row

    # -------------------------------------------------------------------------
    def delete_by_id(self, key_id: int) -> bool:
        with self._session() as session:
            row = session.execute(
                select(HFAccessKey).where(HFAccessKey.id == int(key_id)).limit(1)
            ).scalar_one_or_none()
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    # -------------------------------------------------------------------------
    def clear_active_flags(self, except_key_id: int | None = None) -> None:
        stmt = update(HFAccessKey)
        if except_key_id is not None:
            stmt = stmt.where(HFAccessKey.id != int(except_key_id))
        stmt = stmt.values(is_active=False)
        with self._session() as session:
            session.execute(stmt)
            session.commit()

    # -------------------------------------------------------------------------
    def activate(self, key_id: int) -> bool:
        with self._session() as session:
            row = session.execute(
                select(HFAccessKey.id).where(HFAccessKey.id == int(key_id)).limit(1)
            ).scalar_one_or_none()
            if row is None:
                return False
            session.execute(
                update(HFAccessKey)
                .where(HFAccessKey.id != int(key_id))
                .values(is_active=False)
            )
            session.execute(
                update(HFAccessKey)
                .where(HFAccessKey.id == int(key_id))
                .values(is_active=True)
            )
            session.commit()
            return True

    # -------------------------------------------------------------------------
    def deactivate(self, key_id: int) -> bool:
        with self._session() as session:
            row = session.execute(
                select(HFAccessKey.id).where(HFAccessKey.id == int(key_id)).limit(1)
            ).scalar_one_or_none()
            if row is None:
                return False
            session.execute(
                update(HFAccessKey)
                .where(HFAccessKey.id == int(key_id))
                .values(is_active=False)
            )
            session.commit()
            return True
