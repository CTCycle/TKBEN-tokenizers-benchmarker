# Background Jobs (TKBEN)
Last updated: 2026-04-08


## 1. Purpose
TKBEN uses an in-process job manager for long-running operations so API calls can return quickly and clients can poll progress.

## 2. Core Components
- `TKBEN/server/services/jobs.py`
  - `JobManager`
  - `JobState`
  - `JobProgressReporter`
  - `JobStopChecker`

## 3. Job Lifecycle
`JobManager.start_job(...)`:
1. creates a short `job_id`
2. stores `JobState` as `pending`
3. starts a daemon thread
4. marks job `running`

Terminal states:
- `completed`
- `failed`
- `cancelled`

## 4. API Contract
Jobs API (`TKBEN/server/api/jobs.py`):
- `GET /jobs` list jobs (optional `job_type`)
- `GET /jobs/{job_id}` fetch current status
- `DELETE /jobs/{job_id}` request cancellation

Long-running routes return `JobStartResponse` with:
- `job_id`
- `status`
- `message`
- `poll_interval`

## 5. Progress and Results
- Runners can report progress via `JobProgressReporter` (0 to 100).
- Result payload is stored in `state.result`.
- On failure, a truncated error string is stored in `state.error`.

## 6. Cancellation Model
- `cancel_job` marks `stop_requested=true` and status `cancelled`.
- Cooperative cancellation is handled by runner code via `JobStopChecker`.
- If runner ignores stop checks, terminal state may still arrive from runner completion/failure path.

## 7. Current Constraints
- In-memory only: jobs are not persisted across process restart.
- Thread-based execution: no external queue or distributed workers.
- No per-user authorization boundaries on job listing/cancellation endpoints.
