from __future__ import annotations

import time
from pathlib import Path

from server.services import jobs as jobs_module
from server.configurations import DatabaseSettings
from server.repositories.database.backend import TKBENDatabase
from server.repositories.database.initializer import initialize_database
from server.repositories.jobs import JobRepository
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
def test_terminal_jobs_are_pruned_after_retention() -> None:
    current_time = {"value": 100.0}
    manager = JobManager(
        terminal_retention_seconds=5.0,
        clock=lambda: current_time["value"],
    )

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

###############################################################################
def _job_repository(tmp_path: Path) -> tuple[JobRepository, TKBENDatabase]:
    settings = DatabaseSettings(
        embedded_database=True,
        sqlite_path=tmp_path / "jobs.db",
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=5,
        insert_batch_size=1000,
    )
    database = TKBENDatabase(settings)
    initialize_database(settings=settings)
    return JobRepository(database), database

###############################################################################
def test_terminal_job_history_survives_manager_recreation(tmp_path: Path) -> None:
    repository, database = _job_repository(tmp_path)
    manager = JobManager(repository=repository)
    try:
        completed_id = manager.start_job("sample", lambda: {"value": "saved"})

        def fail(job_id: str) -> dict[str, object]:
            manager.update_result(job_id, {"phase": "started"})
            raise ValueError("runner failed")

        failed_id = manager.start_job("sample", fail)

        def wait_for_stop(job_id: str) -> dict[str, object]:
            manager.update_result(job_id, {"phase": "waiting"})
            while not manager.should_stop(job_id):
                time.sleep(0.01)
            return {"discarded": True}

        cancelled_id = manager.start_job("sample", wait_for_stop)
        manager.request_stop(cancelled_id)
        _wait_for_status(manager, completed_id, "completed")
        _wait_for_status(manager, failed_id, "failed")
        _wait_for_status(manager, cancelled_id, "cancelled")

        restored_manager = JobManager(repository=repository)
        restored_manager.restore_persisted_jobs()

        completed = restored_manager.get_job_status(completed_id)
        failed = restored_manager.get_job_status(failed_id)
        cancelled = restored_manager.get_job_status(cancelled_id)
        assert completed is not None
        assert completed["result"] == {"value": "saved"}
        assert completed["completed_at"] is not None
        assert failed is not None
        assert failed["error"] == "runner failed"
        assert failed["failure_reason"] == "runner_error"
        assert failed["result"] == {"phase": "started"}
        assert cancelled is not None
        assert cancelled["status"] == "cancelled"
        assert cancelled["result"] == {"phase": "waiting"}
    finally:
        database.backend.engine.dispose()

###############################################################################
def test_restart_reconciliation_fails_pending_and_running_jobs(tmp_path: Path) -> None:
    repository, database = _job_repository(tmp_path)
    now = time.time()
    active_rows = (
        ("pending1", "pending", 0.0),
        ("running1", "running", 37.5),
    )
    try:
        for job_id, status, progress in active_rows:
            repository.save_job(
                {
                    "job_id": job_id,
                    "job_type": "dataset_analysis",
                    "status": status,
                    "progress": progress,
                    "result": {"partial": True},
                    "error": None,
                    "failure_reason": None,
                    "created_at": now - 10,
                    "completed_at": None,
                }
            )

        manager = JobManager(repository=repository)
        manager.restore_persisted_jobs()

        for job_id, status, progress in active_rows:
            snapshot = manager.get_job_status(job_id)
            assert snapshot is not None
            assert snapshot["status"] == "failed"
            assert snapshot["progress"] == progress
            assert snapshot["result"] == {"partial": True}
            assert snapshot["error"] == jobs_module.RESTART_INTERRUPTION_ERROR
            assert snapshot["failure_reason"] == "application_restart"
            assert snapshot["completed_at"] is not None
            stored = repository.get_job(job_id)
            assert stored is not None
            assert stored["failure_reason"] == "application_restart"
    finally:
        database.backend.engine.dispose()

###############################################################################
def test_restart_prunes_expired_persisted_terminal_jobs(tmp_path: Path) -> None:
    repository, database = _job_repository(tmp_path)
    now = time.time()
    try:
        repository.save_job(
            {
                "job_id": "expired1",
                "job_type": "benchmark",
                "status": "completed",
                "progress": 100.0,
                "result": {"report_id": "old"},
                "error": None,
                "failure_reason": None,
                "created_at": now - 30,
                "completed_at": now - 10,
            }
        )
        repository.save_job(
            {
                "job_id": "recent01",
                "job_type": "benchmark",
                "status": "failed",
                "progress": 15.0,
                "result": None,
                "error": "recent failure",
                "failure_reason": "runner_error",
                "created_at": now - 10,
                "completed_at": now - 1,
            }
        )

        manager = JobManager(
            terminal_retention_seconds=5.0,
            repository=repository,
            clock=lambda: now,
        )
        manager.restore_persisted_jobs()

        assert repository.get_job("expired1") is None
        assert manager.get_job_status("expired1") is None
        recent = manager.get_job_status("recent01")
        assert recent is not None
        assert recent["error"] == "recent failure"
    finally:
        database.backend.engine.dispose()
