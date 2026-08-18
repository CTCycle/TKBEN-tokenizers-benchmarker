from __future__ import annotations

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
