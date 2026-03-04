# Testing Guidelines (TKBEN)

## 1. Test Stack
- Runner: `pytest`
- E2E browser/API: `pytest-playwright`

## 2. Current Test Layout
- `tests/unit`: unit and contract-focused backend tests
- `tests/e2e`: end-to-end API/UI flows
- `tests/conftest.py`: shared fixtures and job polling helpers
- `tests/run_tests.bat`: Windows orchestration script

## 3. Execution
Recommended on Windows:
```bat
tests\run_tests.bat
```

Manual:
```bat
uv run pytest tests\unit -q
uv run pytest tests\e2e -q
```

## 4. Environment Contracts
Tests consume host/port or full URL settings:
- `APP_TEST_FRONTEND_URL`, `APP_TEST_BACKEND_URL`
- `UI_BASE_URL` / `UI_URL`
- `API_BASE_URL`
- `UI_HOST`, `UI_PORT`
- `FASTAPI_HOST`, `FASTAPI_PORT`

## 5. Job-Based API Tests
- For async operations (`/datasets/*`, `/tokenizers/*`, `/benchmarks/run`), assert:
  - `202` + `job_id` returned
  - polling via `/jobs/{job_id}` reaches `completed`
  - result payload shape is valid

## 6. Authoring Rules
- Keep tests small, deterministic, and isolated.
- Prefer small payloads and local fixtures for speed.
- Add unit tests for metric logic and repository contracts when backend behavior changes.
- Add/update e2e tests when user-visible workflows or endpoint contracts change.

