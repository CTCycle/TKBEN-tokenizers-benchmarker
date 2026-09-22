from __future__ import annotations

import inspect
import threading
import uuid
from dataclasses import dataclass, field
from time import time as wall_time
from typing import Any

from collections.abc import Callable

from server.common.utils.logger import logger
from server.repositories.jobs import JobRepository

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
ACTIVE_STATUSES = {"pending", "running"}
RESTART_INTERRUPTION_ERROR = (
    "Job interrupted because the application restarted while it was running."
)
RESTART_INTERRUPTION_REASON = "application_restart"
REPOSITORY_PRUNE_INTERVAL_SECONDS = 60.0
PROGRESS_PERSIST_INTERVAL_SECONDS = 1.0
PROGRESS_PERSIST_STEP = 1.0

###############################################################################
@dataclass
class JobState:
    job_id: str
    job_type: str
    status: str
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    failure_reason: str | None = None
    created_at: float = field(default_factory=wall_time)
    completed_at: float | None = None
    stop_requested: bool = False
    persisted_progress: float = field(default=0.0, init=False, repr=False)
    last_progress_persisted_at: float = field(
        default_factory=wall_time, init=False, repr=False
    )
    lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    persistence_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    # -------------------------------------------------------------------------
    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    # -------------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.job_id,
                "job_type": self.job_type,
                "status": self.status,
                "progress": self.progress,
                "result": self.result,
                "error": self.error,
                "failure_reason": self.failure_reason,
                "created_at": self.created_at,
                "completed_at": self.completed_at,
            }

###############################################################################
class JobProgressReporter:

    # -------------------------------------------------------------------------
    def __init__(self, job_manager: JobManager, job_id: str) -> None:
        self.job_manager = job_manager
        self.job_id = job_id

    # -------------------------------------------------------------------------
    def __call__(self, value: float) -> None:
        self.job_manager.update_progress(self.job_id, value)

###############################################################################
class JobStopChecker:

    # -------------------------------------------------------------------------
    def __init__(self, job_manager: JobManager, job_id: str) -> None:
        self.job_manager = job_manager
        self.job_id = job_id

    # -------------------------------------------------------------------------
    def __call__(self) -> bool:
        return self.job_manager.should_stop(self.job_id)

###############################################################################
class JobManager:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        terminal_retention_seconds: float = 3600.0,
        repository: JobRepository | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.jobs: dict[str, JobState] = {}
        self.threads: dict[str, threading.Thread] = {}
        self.lock = threading.Lock()
        self.terminal_retention_seconds = max(0.0, float(terminal_retention_seconds))
        self.repository = repository
        self.clock = clock or wall_time
        self._last_repository_prune_at: float | None = None

    # -------------------------------------------------------------------------
    def restore_persisted_jobs(self) -> None:
        """Restore job history and fail work that cannot resume after restart."""
        if self.repository is None:
            return

        now = self.clock()
        self.repository.delete_expired_jobs(now, self.terminal_retention_seconds)
        restored_jobs: dict[str, JobState] = {}
        for snapshot in self.repository.list_jobs():
            state = JobState(
                job_id=snapshot["job_id"],
                job_type=snapshot["job_type"],
                status=snapshot["status"],
                progress=snapshot["progress"],
                result=snapshot["result"],
                error=snapshot["error"],
                failure_reason=snapshot["failure_reason"],
                created_at=snapshot["created_at"],
                completed_at=snapshot["completed_at"],
            )
            state.persisted_progress = state.progress
            state.last_progress_persisted_at = now
            if state.status in ACTIVE_STATUSES:
                state.update(
                    status="failed",
                    error=RESTART_INTERRUPTION_ERROR,
                    failure_reason=RESTART_INTERRUPTION_REASON,
                    completed_at=now,
                )
                self.repository.save_job(state.snapshot())
            restored_jobs[state.job_id] = state

        with self.lock:
            self.jobs.update(restored_jobs)
            self._last_repository_prune_at = now

    # -------------------------------------------------------------------------
    def start_job(
        self,
        job_type: str,
        runner: Callable[..., dict[str, Any]],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> str:
        while True:
            job_id = str(uuid.uuid4())[:8]
            with self.lock:
                in_memory = job_id in self.jobs
            if not in_memory and (
                self.repository is None or self.repository.get_job(job_id) is None
            ):
                break
        state = JobState(
            job_id=job_id,
            job_type=job_type,
            status="pending",
            created_at=self.clock(),
        )
        runner_kwargs = kwargs.copy() if kwargs else {}

        if self._runner_accepts_job_id(runner):
            runner_kwargs["job_id"] = job_id

        with self.lock:
            self._prune_terminal_jobs_locked(self.clock())
            self.jobs[job_id] = state

        try:
            self._persist_state(state)
            with state.persistence_lock:
                state.update(status="running")
                self._persist_state_locked(state)
        except Exception:
            with self.lock:
                self.jobs.pop(job_id, None)
            raise

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, runner, args, runner_kwargs),
            daemon=True,
        )

        with self.lock:
            self.threads[job_id] = thread

        thread.start()

        logger.info("Started job %s (type=%s)", job_id, job_type)
        return job_id

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            self._prune_terminal_jobs_locked(self.clock())
            state = self.jobs.get(job_id)
        if state is None:
            return None
        with state.persistence_lock:
            return state.snapshot()

    # -------------------------------------------------------------------------
    def is_job_running(self, job_type: str | None = None) -> bool:
        with self.lock:
            self._prune_terminal_jobs_locked(self.clock())
            for state in self.jobs.values():
                if state.status in ("pending", "running"):
                    if job_type is None or state.job_type == job_type:
                        return True
        return False

    # -------------------------------------------------------------------------
    def should_stop(self, job_id: str) -> bool:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return True
        return state.stop_requested

    # -------------------------------------------------------------------------
    def request_stop(self, job_id: str) -> dict[str, Any] | None:
        """Request cooperative cancellation for an active job."""
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return None

        with state.lock:
            if state.status not in TERMINAL_STATUSES:
                state.stop_requested = True
        return state.snapshot()

    # -------------------------------------------------------------------------
    def update_progress(self, job_id: str, progress: float) -> None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state:
            with state.persistence_lock:
                normalized = min(100.0, max(0.0, progress))
                state.update(progress=normalized)
                now = self.clock()
                if self.repository is not None and (
                    abs(normalized - state.persisted_progress) >= PROGRESS_PERSIST_STEP
                    or normalized >= 100.0
                    or now - state.last_progress_persisted_at
                    >= PROGRESS_PERSIST_INTERVAL_SECONDS
                ):
                    self._persist_state_locked(state)
                    state.persisted_progress = normalized
                    state.last_progress_persisted_at = now

    # -------------------------------------------------------------------------
    def update_result(self, job_id: str, patch: dict[str, Any]) -> None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return
        with state.persistence_lock:
            with state.lock:
                existing = state.result or {}
                merged = {**existing, **patch}
                state.result = merged
            self._persist_state_locked(state)

    # -------------------------------------------------------------------------
    def _prune_terminal_jobs_locked(self, now: float) -> None:
        expired_job_ids: list[str] = []
        for job_id, state in self.jobs.items():
            completed_at = state.completed_at
            if (
                state.status in TERMINAL_STATUSES
                and completed_at is not None
                and now - completed_at > self.terminal_retention_seconds
            ):
                expired_job_ids.append(job_id)

        for job_id in expired_job_ids:
            self.jobs.pop(job_id, None)
            self.threads.pop(job_id, None)

        if (
            self.repository is not None
            and (
                self._last_repository_prune_at is None
                or now - self._last_repository_prune_at
                >= REPOSITORY_PRUNE_INTERVAL_SECONDS
            )
        ):
            self.repository.delete_expired_jobs(now, self.terminal_retention_seconds)
            self._last_repository_prune_at = now

    # -------------------------------------------------------------------------
    def _persist_state(self, state: JobState) -> None:
        if self.repository is not None:
            with state.persistence_lock:
                self._persist_state_locked(state)

    # -------------------------------------------------------------------------
    def _persist_state_locked(self, state: JobState) -> None:
        if self.repository is not None:
            self.repository.save_job(state.snapshot())

    # -------------------------------------------------------------------------
    def _finish_job(
        self,
        state: JobState,
        *,
        status: str,
        completed_at: float,
        result: dict[str, Any] | None = None,
        set_result: bool = False,
        error: str | None = None,
        failure_reason: str | None = None,
    ) -> None:
        with state.persistence_lock:
            updates: dict[str, Any] = {
                "status": status,
                "completed_at": completed_at,
                "error": error,
                "failure_reason": failure_reason,
                "progress": 100.0 if status == "completed" else state.progress,
            }
            if set_result:
                updates["result"] = result
            state.update(**updates)
            try:
                self._persist_state_locked(state)
                state.persisted_progress = state.progress
                state.last_progress_persisted_at = self.clock()
            except Exception:
                logger.exception("Unable to persist terminal state for job %s", state.job_id)

    # -------------------------------------------------------------------------
    def _run_job(
        self,
        job_id: str,
        runner: Callable[..., dict[str, Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return

        try:
            result = runner(*args, **kwargs)
            if state.stop_requested:
                self._finish_job(
                    state, status="cancelled", completed_at=self.clock()
                )
            else:
                result_payload = result or {}
                with state.lock:
                    merged = {**(state.result or {}), **result_payload}
                self._finish_job(
                    state,
                    status="completed",
                    result=merged if merged else None,
                    set_result=True,
                    completed_at=self.clock(),
                )
                logger.info("Job %s completed successfully", job_id)
        except Exception as exc:  # noqa: BLE001
            if state.stop_requested:
                self._finish_job(
                    state, status="cancelled", completed_at=self.clock()
                )
                logger.info("Job %s cancelled during execution", job_id)
            else:
                error_msg = str(exc).split("\n")[0][:200]
                self._finish_job(
                    state,
                    status="failed",
                    completed_at=self.clock(),
                    error=error_msg,
                    failure_reason="runner_error",
                )
                logger.error("Job %s failed: %s", job_id, error_msg)
                logger.debug("Job %s error details", job_id, exc_info=True)

    # -------------------------------------------------------------------------
    def _runner_accepts_job_id(self, runner: Callable[..., dict[str, Any]]) -> bool:
        try:
            signature = inspect.signature(runner)
        except (TypeError, ValueError):
            return False
        for param in signature.parameters.values():
            if param.kind == param.VAR_KEYWORD:
                return True
        return "job_id" in signature.parameters
