from __future__ import annotations

from datetime import datetime, timezone
import threading
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from server.repositories.database.backend import TKBENDatabase, get_database
from server.repositories.schemas.models import ManagedJob

###############################################################################
class JobRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: TKBENDatabase | None = None) -> None:
        self.database = database or get_database()
        self.write_lock = threading.Lock()

    # -------------------------------------------------------------------------
    def _session(self) -> Session:
        return Session(bind=self.database.backend.engine)

    # -------------------------------------------------------------------------
    @staticmethod
    def _datetime(timestamp: float | None) -> datetime | None:
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp, timezone.utc)

    # -------------------------------------------------------------------------
    @classmethod
    def _values(cls, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": snapshot["job_id"],
            "job_type": snapshot["job_type"],
            "status": snapshot["status"],
            "progress": snapshot["progress"],
            "result": snapshot["result"],
            "error": snapshot["error"],
            "failure_reason": snapshot["failure_reason"],
            "created_at": cls._datetime(snapshot["created_at"]),
            "completed_at": cls._datetime(snapshot["completed_at"]),
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _snapshot(row: ManagedJob) -> dict[str, Any]:
        return {
            "job_id": row.job_id,
            "job_type": row.job_type,
            "status": row.status,
            "progress": row.progress,
            "result": row.result,
            "error": row.error,
            "failure_reason": row.failure_reason,
            "created_at": row.created_at.timestamp(),
            "completed_at": (
                row.completed_at.timestamp() if row.completed_at is not None else None
            ),
        }

    # -------------------------------------------------------------------------
    def save_job(self, snapshot: dict[str, Any]) -> None:
        values = self._values(snapshot)
        with self.write_lock:
            with self._session() as session:
                row = session.get(ManagedJob, values["job_id"])
                if row is None:
                    session.add(ManagedJob(**values))
                else:
                    for field_name, value in values.items():
                        setattr(row, field_name, value)
                session.commit()

    # -------------------------------------------------------------------------
    def list_jobs(self) -> list[dict[str, Any]]:
        with self._session() as session:
            rows = session.execute(
                select(ManagedJob).order_by(ManagedJob.created_at, ManagedJob.job_id)
            ).scalars()
            return [self._snapshot(row) for row in rows]

    # -------------------------------------------------------------------------
    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._session() as session:
            row = session.get(ManagedJob, job_id)
            return self._snapshot(row) if row is not None else None

    # -------------------------------------------------------------------------
    def delete_expired_jobs(self, now: float, retention_seconds: float) -> None:
        cutoff = self._datetime(now - max(0.0, retention_seconds))
        assert cutoff is not None
        with self.write_lock:
            with self._session() as session:
                session.execute(
                    delete(ManagedJob).where(
                        ManagedJob.status.in_(("completed", "failed", "cancelled")),
                        ManagedJob.completed_at < cutoff,
                    )
                )
                session.commit()
