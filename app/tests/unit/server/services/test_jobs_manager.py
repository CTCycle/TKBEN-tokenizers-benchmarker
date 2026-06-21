from __future__ import annotations

import threading
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

    assert manager.list_jobs() == []
    assert manager.get_job_status(job_id) is None


###############################################################################
def test_cancellation_reports_cancelled_state() -> None:
    manager = JobManager(terminal_retention_seconds=60.0)
    started = threading.Event()

    def cancellable_runner(job_manager: JobManager, job_id: str) -> dict[str, object]:
        started.set()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if job_manager.should_stop(job_id):
                return {"observed_stop": True}
            time.sleep(0.01)
        return {"observed_stop": False}

    job_id = manager.start_job(
        "sample",
        cancellable_runner,
        kwargs={"job_manager": manager},
    )
    assert started.wait(timeout=1.0)

    cancelled = manager.cancel_job(job_id)
    status = _wait_for_status(manager, job_id, "cancelled")

    assert cancelled is True
    assert status["status"] == "cancelled"
    assert status["error"] is None
