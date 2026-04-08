# Testing Guidelines (TKBEN)
Last updated: 2026-04-08


## 1. Test Stack
- Runner: `pytest`
- UI/API E2E support: `pytest-playwright`
- Shared fixtures/helpers: `tests/conftest.py`

## 2. Test Layout
- `tests/unit`: backend unit/contract tests
- `tests/e2e`: end-to-end API/UI workflow tests
- `tests/run_tests.bat`: Windows orchestration entrypoint

## 3. Execution
Recommended on Windows:
```bat
tests\run_tests.bat
```

Manual (prefer runtime venv):
```bat
runtimes\.venv\Scripts\python.exe -m pytest tests\unit -q
runtimes\.venv\Scripts\python.exe -m pytest tests\e2e -q
```

## 4. Environment Contracts
Tests read these variables (direct URL vars take priority):
- `APP_TEST_FRONTEND_URL`, `APP_TEST_BACKEND_URL`
- `UI_BASE_URL`, `UI_URL`
- `API_BASE_URL`
- `UI_HOST`, `UI_PORT`
- `FASTAPI_HOST`, `FASTAPI_PORT`

## 5. Async Job API Testing
For async operations (`/datasets/*`, `/tokenizers/*`, `/benchmarks/run`):
- assert `202` + `job_id` on start
- poll `/jobs/{job_id}` until terminal state
- assert final payload shape and required fields

Terminal states:
- `completed`
- `failed`
- `cancelled`

## 6. Authoring Rules
- Keep tests deterministic, isolated, and small.
- Prefer local fixtures and small payloads.
- Update unit tests when service/repository/domain behavior changes.
- Update e2e tests when route contracts or user-visible workflows change.
