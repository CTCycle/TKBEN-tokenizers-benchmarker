from __future__ import annotations

import time

from server.services import jobs as jobs_module
from server.services.jobs import JobManager


###############################################################################
def _wait_for_status(
    manager: JobManager,
    job_id: str,
    expected_status: str,
    *,
    timeout_seconds: float = 2.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = manager.get_job_status(job_id)
        if status and status["status"] == expected_status:
            return status
        time.sleep(0.01)
    status = manager.get_job_status(job_id)
    raise AssertionError(f"Expected {expected_status}, got {status}")


###############################################################################
def test_completed_job_remains_visible_within_retention() -> None:
    manager = JobManager(terminal_retention_seconds=60.0)

    job_id = manager.start_job("sample", lambda: {"value": 1})

    status = _wait_for_status(manager, job_id, "completed")

    assert status["result"] == {"value": 1}
    assert manager.get_job_status(job_id) is not None


###############################################################################
def test_terminal_jobs_are_pruned_after_retention(monkeypatch) -> None:
    current_time = {"value": 100.0}
    monkeypatch.setattr(jobs_module, "monotonic", lambda: current_time["value"])
    manager = JobManager(terminal_retention_seconds=5.0)

    job_id = manager.start_job("sample", lambda: {"value": 1})
    status = _wait_for_status(manager, job_id, "completed")
    assert status["completed_at"] == 100.0

    current_time["value"] = 106.0

    assert manager.get_job_status(job_id) is None


###############################################################################
def test_running_job_can_be_stopped_cooperatively() -> None:
    manager = JobManager(terminal_retention_seconds=60.0)

    def runner(job_id: str) -> dict[str, object]:
        while not manager.should_stop(job_id):
            time.sleep(0.01)
        return {"ignored": True}

    job_id = manager.start_job("sample", runner)

    stop_status = manager.request_stop(job_id)
    assert stop_status is not None
    assert stop_status["status"] == "running"
    assert manager.should_stop(job_id) is True

    status = _wait_for_status(manager, job_id, "cancelled")
    assert status["result"] is None


###############################################################################
def test_stopping_unknown_job_returns_none() -> None:
    assert JobManager().request_stop("missing") is None
