# Python Guidelines (TKBEN)

## 1. Scope
Apply to:
- `TKBEN/server`
- Python scripts in `TKBEN/scripts`
- Python tests in `tests`

Target runtime:
- Python `>=3.14` (see `pyproject.toml`)

## 2. Style and Structure
- Follow PEP 8 and keep modules focused.
- Keep business logic in `services`, HTTP orchestration in `api`, persistence in `repositories`.
- Reuse existing domain models in `TKBEN/server/domain` for request/response payloads.
- Prefer small pure helper functions for metric/transform logic.
- Leverage classes to group methods with similar scope
- Enforce the use of cosmetic separators (series of # and - symbols) for class and functions

## 3. Typing
- Keep type hints on public functions and non-trivial internals.
- Prefer built-in generics (`list[str]`, `dict[str, Any]`) and `X | None`.
- Maintain compatibility with current model typing conventions in `entities` and `configurations`.

## 4. API Layer Conventions
- Use FastAPI routers with constants from `TKBEN/server/common/constants.py`.
- Validate inputs early and return explicit `HTTPException` messages.
- For long-running CPU/IO tasks, keep job-based flow (`JobStartResponse` + `/jobs/{job_id}` polling).
- Use `asyncio.to_thread(...)` where synchronous service logic is invoked from async routes.

## 5. Data and Repository Rules
- Do not bypass repository/database abstractions for new persistence logic.
- Keep SQLite/Postgres compatibility (`repositories/database/*`).
- Preserve schema consistency with `repositories/schemas/models.py`.

## 6. Security and Secrets
- Never hardcode secrets.
- Keep Hugging Face key encryption behavior compatible with `HF_KEYS_ENCRYPTION_KEY`.
- Validate file uploads and external inputs before processing.

## 7. Testing
- Use `pytest`.
- Add/adjust tests in:
  - `tests/unit` for logic and contracts
  - `tests/e2e` for API/UI flows
- Keep tests deterministic; avoid network-heavy paths unless explicitly opt-in.

