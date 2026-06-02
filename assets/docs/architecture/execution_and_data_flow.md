# Execution and Data Flow
Last updated: 2026-06-02

## Layered Architecture
Primary backend flow:
`endpoint (api/*) -> service (services/*) -> repository/serializer (repositories/*) -> DB/filesystem`

## Key Module Responsibilities
- `server/app.py`
  - FastAPI app factory, router registration, and SPA serving in Tauri mode.
- `server/api/*`
  - HTTP contracts, status codes, request/response models, and job dispatch.
- `server/domain/*`
  - Pydantic and dataclass domain models plus settings schemas.
- `server/services/*`
  - Business logic, long-running operations, and orchestration.
- `server/repositories/database/*`
  - Backend selection and database adapter implementations.
- `server/repositories/schemas/*`
  - SQLAlchemy models and types.
- `server/repositories/serialization/*`
  - Persistence serialization and report materialization.
- `server/common/*`
  - Constants, logging, and helper utilities.

## Service Notes
- `services/tokenizer_storage.py`
  - Tokenizer identifier validation, cache path resolution, and Hugging Face URL construction shared by tokenizer workflows.
- `services/dashboard_export_helpers.py`
  - Dashboard export payload parsing and value formatting helpers used by the PDF export service.
- `services/benchmark_engine.py`
  - Warmup and timed trial batch runner with per-batch observations and cancellation checks.
- `repositories/database/backend.py`
  - `get_database()` is the single cached accessor for the configured backend; repositories receive or resolve this dependency instead of importing module-level database state.
- `repositories/serialization/benchmark_reports.py`
  - Benchmark report persistence serialization and Pydantic response normalization.
- `repositories/frequencies.py`
  - Temporary SQLite-backed frequency persistence used by metrics services for large vocabularies.

## Async and Sync Behavior
- FastAPI endpoints are mostly `async def`.
- Blocking logic is intentionally offloaded with `await asyncio.to_thread(...)`.
- Long-running operations such as download, analysis, benchmark, and report generation run in background threads via `JobManager`.
- Job polling and cancel operations are synchronous handler functions over in-memory job state.
- Repository and database operations are synchronous SQLAlchemy session usage.

## Constraint
Async handlers must not execute CPU-heavy or blocking I/O inline. They should offload to threads or the job system.
