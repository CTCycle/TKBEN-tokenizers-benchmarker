from __future__ import annotations

from time import time

from fastapi.testclient import TestClient

from server.app import app

###############################################################################
def _job_status(job_id: str, status: str) -> dict[str, object]:
    return {
        "job_id": job_id,
        "job_type": "benchmark",
        "status": status,
        "progress": 42.0,
        "result": None,
        "error": None,
    }

###############################################################################
def test_job_routes_distinguish_missing_terminal_and_running_jobs(monkeypatch) -> None:

    ###############################################################################
    class FakeJobManager:

        # -------------------------------------------------------------------------
        def get_job_status(self, job_id: str):
            if job_id == "missing":
                return None
            return _job_status(job_id, "completed" if job_id == "done" else "running")

        # -------------------------------------------------------------------------
        def request_stop(self, job_id: str):
            if job_id == "missing":
                return None
            if job_id == "done":
                return _job_status(job_id, "completed")
            return _job_status(job_id, "stopping")

    monkeypatch.setattr(app.state, "job_manager", FakeJobManager())
    client = TestClient(app)

    assert client.get("/api/jobs/missing").status_code == 404
    terminal = client.post("/api/jobs/done/cancel")
    running = client.post("/api/jobs/active/cancel")

    assert terminal.status_code == 409
    assert "already completed" in terminal.json()["detail"]
    assert running.status_code == 200
    assert running.json()["status"] == "stopping"

###############################################################################
def test_reconciled_job_remains_addressable_and_rejects_cancellation(
    monkeypatch,
) -> None:

    ###############################################################################
    class MemoryJobRepository:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            now = time()
            self.rows = {
                "restart1": {
                    "job_id": "restart1",
                    "job_type": "benchmark",
                    "status": "running",
                    "progress": 20.0,
                    "result": None,
                    "error": None,
                    "failure_reason": None,
                    "created_at": now - 1,
                    "completed_at": None,
                }
            }

        # -------------------------------------------------------------------------
        def delete_expired_jobs(self, now: float, retention_seconds: float) -> int:
            del now, retention_seconds
            return 0

        # -------------------------------------------------------------------------
        def list_jobs(self) -> list[dict[str, object]]:
            return list(self.rows.values())

        # -------------------------------------------------------------------------
        def save_job(self, snapshot: dict[str, object]) -> None:
            self.rows[snapshot["job_id"]] = snapshot

    from server.services.jobs import JobManager

    manager = JobManager(repository=MemoryJobRepository())
    manager.restore_persisted_jobs()
    monkeypatch.setattr(app.state, "job_manager", manager)
    client = TestClient(app)

    status = client.get("/api/jobs/restart1")
    cancellation = client.post("/api/jobs/restart1/cancel")

    assert status.status_code == 200
    assert status.json()["status"] == "failed"
    assert status.json()["error"] == (
        "Job interrupted because the application restarted while it was running."
    )
    assert "failure_reason" not in status.json()
    assert cancellation.status_code == 409
    assert "already failed" in cancellation.json()["detail"]
