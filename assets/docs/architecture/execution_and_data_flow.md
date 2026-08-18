# Execution and Data Flow
Last updated: 2026-08-18

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
- `app/client/angular/app/core/utils/dataset-dashboard-data.ts`
  - Shared frontend normalization for numeric metrics, histograms, word frequencies, Zipf curves, and word-cloud terms before chart rendering.
- `app/client/angular/app/core/utils/benchmark-dashboard-data.ts`
  - Pure classification of normalized dashboard widgets as point, bucket, or distribution data shapes.
- `app/client/angular/app/core/state/*.store.ts` and the shared chart components
  - Signal-based catalog, report, persistence, and modal state plus the shared chart rendering contracts.
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
- `repositories/serialization/datasets.py` and `repositories/serialization/tokenizer_reports.py`
  - Canonical dataset and tokenizer-report persistence serialization boundaries.
- `repositories/frequencies.py`
  - Temporary SQLite-backed frequency persistence used by metrics services for large vocabularies.

## Catalog Filtering Flow
- Dataset and tokenizer catalog controls are rendered by the Angular page templates and typed reactive forms.
- Pages keep filter input state locally and debounce changes by 250 ms before calling the typed API service.
- The backend applies search, source, and numeric comparison filters in the catalog service and returns the filtered items plus a result count.
- Dataset/tokenizer catalog refreshes use request sequence guards so an older response cannot replace a newer filter result; the filter state only changes catalog visibility, not benchmark selection.
- Hugging Face tokenizer discovery builds one provider query from the validated domain contract. Hub-native filters are sent to `HfApi.list_models` with expanded `siblings` metadata; bounded candidates are locally checked for root-level tokenizer artifacts, supported any-text-task categories, excluded tags, and optional vocabulary metadata/order. Artifact validation accepts standard `tokenizer.json`, SentencePiece, BPE (`vocab.json` plus `merges.txt`), and WordPiece (`vocab.txt`) combinations with the required metadata, and rejects weight-only, metadata-only, and nested-only repositories before presentation. Vocabulary enrichment expands `config` only when needed and reads only top-level `config.vocab_size`; discovery never loads model weights.
- Benchmark report management requests one server page at a time. Search and sort are debounced in the Angular store, `switchMap` prevents stale responses from replacing newer pages, and deletion updates the visible page immediately before refreshing persisted state.
- Dataset page rendering uses shared pure data helpers so missing, malformed, or non-finite optional chart payloads degrade to empty states without duplicating parsing logic in the page component.

## Async and Sync Behavior
- FastAPI endpoints are mostly `async def`.
- Blocking logic is intentionally offloaded with `await asyncio.to_thread(...)`.
- Long-running operations such as download, analysis, benchmark, and report generation run in background threads via `JobManager`.
- Job polling and cancel operations are synchronous handler functions over in-memory job state.
- Repository and database operations are synchronous SQLAlchemy session usage.

Hugging Face discovery failures are allowed to propagate through the service so the API can return its sanitized HTTP 500 response; an outage is never represented as a successful empty catalog. Benchmark report list/load/save/delete operations remain on the API -> service -> serializer -> repository path.

## Constraint
Async handlers must not execute CPU-heavy or blocking I/O inline. They should offload to threads or the job system.
