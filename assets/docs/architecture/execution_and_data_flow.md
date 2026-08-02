# Execution and Data Flow
Last updated: 2026-08-02

## Layered Architecture
Primary backend flow:
`endpoint (api/*) -> service (services/*) -> repository/serializer (repositories/*) -> DB/filesystem`

## Key Module Responsibilities
- `server/app.py`
  - FastAPI app factory, router registration, and SPA serving support.
- `server/api/*`
  - HTTP contracts, status codes, request/response models, multipart handling, and HTTP error mapping.
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
- `services/tokenizer_reporting.py`
  - Tokenizer metadata resolution, cached metadata loading, vocabulary analysis, histogram generation, report persistence, and report retrieval.
- `services/dataset_statistics.py`
  - Deterministic dataset length statistics and histogram construction used by `DatasetService`.
- `app/client/src/features/dataset/datasetDashboardData.ts`
  - Shared frontend normalization for numeric metrics, histograms, word frequencies, Zipf curves, and word-cloud terms before chart rendering.
- `app/client/src/features/benchmark-dashboard/benchmarkDashboardChartUtils.ts`
  - Pure classification of normalized dashboard widgets as point, bucket, or distribution data shapes.
- `app/client/src/hooks/useAvailableDatasets.ts`, `useBodyScrollLock.ts`, and `useCompactChart.ts`
  - Shared frontend lifecycle state for catalog loading, modal scroll restoration, and the `700px` compact-chart breakpoint.
- `services/dashboard_export_helpers.py`
  - Dashboard export payload parsing and value formatting helpers used by the PDF export service.
- `services/benchmark_engine.py`
  - Warmup and timed trial batch runner with per-batch observations and cancellation checks.
- `services/managed_jobs.py`
  - Shared typed dispatcher for conflict checks, job start, initial status verification, polling metadata, and initialization failures; `server.api.helpers.ManagedJobHttpAdapter` maps its errors to HTTP responses.
- `repositories/database/backend.py`
  - `get_database()` is the single cached accessor for the configured backend; repositories receive or resolve this dependency instead of importing module-level database state.
- `repositories/serialization/benchmark_reports.py`
  - Benchmark report persistence serialization and Pydantic response normalization.
- `repositories/serialization/data.py` and `repositories/serialization/tokenizer_reports.py`
  - Canonical dataset and tokenizer-report persistence serialization boundaries.
- `repositories/frequencies.py`
  - Temporary SQLite-backed frequency persistence used by metrics services for large vocabularies.

## Catalog Filtering Flow
- Dataset and tokenizer catalog controls are rendered by the shared frontend `CatalogFilterToolbar`.
- Pages keep filter input state locally and debounce changes by 250 ms before calling the typed API service.
- The backend applies search, source, and numeric comparison filters in the catalog service and returns the filtered items plus a result count.
- Dataset/tokenizer catalog refreshes use request sequence guards so an older response cannot replace a newer filter result; the filter state only changes catalog visibility, not benchmark selection.
- Dataset page rendering uses shared pure data helpers so missing, string-encoded, malformed, or non-finite optional chart payloads degrade to empty states without duplicating parsing logic in the page component.

## Async and Sync Behavior
- FastAPI endpoints are mostly `async def`.
- Blocking logic is intentionally offloaded with `await asyncio.to_thread(...)`.
- Long-running operations such as download, analysis, benchmark, and report generation run in background threads via `JobManager`.
- Job polling and cancel operations are synchronous handler functions over in-memory job state.
- Repository and database operations are synchronous SQLAlchemy session usage.

Tokenizer scan failures from Hugging Face are allowed to propagate through the service so the API can return its sanitized HTTP 500 response; an outage is never represented as a successful empty catalog. No legacy service aliases are retained after the reporting extraction.

## Constraint
Async handlers must not execute CPU-heavy or blocking I/O inline. They should offload to threads or the job system.
