# Python Guidelines (TKBEN)
Last updated: 2026-04-08


## 1. Scope
Apply these rules to:
- `TKBEN/server`
- Python scripts in `TKBEN/scripts`
- Python tests in `tests`

Target runtime:
- Python `>=3.14` (see `pyproject.toml`)

Execution preference:
- If present, use `runtimes/.venv`:
  - `.\runtimes\.venv\Scripts\python.exe -m ...`

## 2. Structure and Responsibilities
- Keep HTTP orchestration in `TKBEN/server/api`.
- Keep business logic in `TKBEN/server/services`.
- Keep persistence concerns in `TKBEN/server/repositories`.
- Reuse request/response/domain models from `TKBEN/server/domain`.

## 3. Style and Readability
- Follow PEP 8 and explicit naming.
- Keep modules small and focused.
- Prefer pure helper functions for metric/transform logic.
- Use classes when methods share state/scope.
- Preserve existing separator/comment conventions where already used.

## 4. Typing
- Keep type hints on public functions and non-trivial internals.
- Prefer built-in generics (`list[str]`, `dict[str, Any]`) and `X | None`.
- Avoid `Any` unless unavoidable at boundaries.
- Keep typing compatible with existing conventions in `domain` and `configurations`.

## 5. FastAPI Conventions
- Use route/path constants from `TKBEN/server/common/constants.py`.
- Validate inputs early and return explicit `HTTPException` messages.
- For long-running operations, preserve job-based flow:
  - return `JobStartResponse`
  - poll via `/jobs/{job_id}`
- Use `asyncio.to_thread(...)` when sync service code is called from async routes.

## 6. Data and Persistence
- Do not bypass repository/database abstractions for new persistence logic.
- Preserve SQLite/PostgreSQL compatibility (`repositories/database/*`).
- Keep schema changes aligned with `repositories/schemas/models.py`.

## 7. Security and Secrets
- Never hardcode secrets.
- Keep HF key encryption compatible with `HF_KEYS_ENCRYPTION_KEY`.
- Validate uploads and external input before processing.

## 8. Testing
- Use `pytest`.
- Update tests in:
  - `tests/unit` for logic/contracts
  - `tests/e2e` for user/API workflows
- Keep tests deterministic and avoid real network dependency unless explicitly intended.
