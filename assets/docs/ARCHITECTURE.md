# ARCHITECTURE
Last updated: 2026-06-02

## System Summary
TKBEN is a tokenizer benchmarking platform with:
- FastAPI backend (`app/server`)
- React + Vite frontend (`app/client`)
- Optional Tauri desktop packaging (`app/client/src-tauri`)
- Shared local resources and settings (`app/resources`, `settings`)

Backend APIs are mounted under `/api/*`. Frontend calls `/api` and relies on Vite proxy in dev/preview.

## Repository Structure
Source-level structure (generated folders like `node_modules`, `dist`, caches omitted):

```text
.
├─ runtimes/
│  ├─ .venv/                # Runtime virtualenv used by scripts
│  └─ uv.lock               # Runtime lockfile used by launcher/packaging
├─ assets/
│  ├─ docs/
│  └─ figures/
├─ start_on_windows.bat
├─ setup_and_maintenance.bat
├─ settings/
│  ├─ .env
│  ├─ .env.example
│  └─ configurations.json
├─ app/
│  ├─ client/
│  │  ├─ package.json
│  │  ├─ vite.config.ts
│  │  ├─ src/
│  │  │  ├─ main.tsx
│  │  │  ├─ App.tsx
│  │  │  ├─ App.css
│  │  │  ├─ index.css
│  │  │  ├─ components/
│  │  │  ├─ common/
│  │  │  │  └─ constants/
│  │  │  ├─ contexts/
│  │  │  ├─ hooks/
│  │  │  ├─ pages/
│  │  │  ├─ services/
│  │  │  ├─ types/
│  │  │  └─ workers/
│  │  └─ src-tauri/
│  │     ├─ tauri.conf.json
│  │     └─ src/main.rs
│  ├─ server/
│  │  ├─ pyproject.toml
│  │  ├─ app.py
│  │  ├─ api/
│  │  ├─ configurations/
│  │  ├─ domain/
│  │  ├─ services/
│  │  ├─ repositories/
│  │  │  ├─ database/
│  │  │  ├─ schemas/
│  │  │  ├─ serialization/
│  │  │  └─ frequencies.py
│  │  └─ common/
│  ├─ scripts/
│  │  └─ initialize_database.py
│  ├─ tests/
│  │  ├─ run_tests.bat
│  │  ├─ conftest.py
│  │  ├─ unit/
│  │  └─ e2e/
│  └─ resources/
│     ├─ database.db
│     ├─ logs/
│     ├─ templates/
│     └─ sources/
├─ release/
│  ├─ tauri/
│  │  ├─ build_with_tauri.bat
│  │  └─ scripts/
│  └─ windows/
```

## Application Entry Points
- Backend app factory/module:
  - `server.app:create_app` constructs the FastAPI app and registers API/frontend routes.
  - `server.app:app` is the canonical ASGI entry point. Repository-root tooling must provide `app/` on `PYTHONPATH` rather than using alternate module paths.
- Frontend entry:
  - `app/client/src/main.tsx`
- Frontend routing root:
  - `app/client/src/App.tsx`
- Desktop runtime entry:
  - `app/client/src-tauri/src/main.rs`
- Windows local launcher:
  - `start_on_windows.bat`

## Backend API Endpoints
All routers are included with `prefix="/api"` in backend app startup.

### Datasets
- `GET /api/datasets/list`
- `GET /api/datasets/metrics/catalog`
- `POST /api/datasets/download`
- `POST /api/datasets/upload`
- `POST /api/datasets/analyze`
- `GET /api/datasets/reports/latest`
- `GET /api/datasets/reports/{report_id}`
- `DELETE /api/datasets/delete`

### Tokenizers
- `GET /api/tokenizers/settings`
- `GET /api/tokenizers/scan`
- `GET /api/tokenizers/list`
- `POST /api/tokenizers/download`
- `POST /api/tokenizers/reports/generate`
- `GET /api/tokenizers/reports/latest`
- `GET /api/tokenizers/reports/{report_id}`
- `GET /api/tokenizers/reports/{report_id}/vocabulary`
- `POST /api/tokenizers/upload`
- `DELETE /api/tokenizers/custom`

### Benchmarks
- `POST /api/benchmarks/run`
- `GET /api/benchmarks/reports`
- `GET /api/benchmarks/reports/{report_id}`
- `GET /api/benchmarks/metrics/catalog`

### Jobs
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `DELETE /api/jobs/{job_id}`

### Hugging Face Keys
- `POST /api/keys`
- `GET /api/keys`
- `DELETE /api/keys/{key_id}`
- `POST /api/keys/{key_id}/activate`
- `POST /api/keys/{key_id}/deactivate`
- `POST /api/keys/{key_id}/reveal`

### Exports
- `POST /api/exports/dashboard/pdf`

## Layered Architecture
Primary backend flow:
`endpoint (api/*) -> service (services/*) -> repository/serializer (repositories/*) -> DB/filesystem`

Examples:
- Dataset flow:
  - `api/datasets.py` validates request and starts jobs
  - `services/dataset_jobs.py` + `services/datasets.py` handle orchestration/analysis/download
  - `repositories/serialization/data.py` persists and retrieves dataset/report records
- Benchmark flow:
  - `api/benchmarks.py` receives run/list/report requests
  - `services/benchmarks.py` coordinates tokenizer loading + metrics
  - `repositories/benchmarks.py` handles SQL reads/writes
  - `services/benchmark_execution.py` performs trial execution, metric computation, and response assembly
  - `services/benchmark_metric_plan.py` maps selected metric keys to execution-time compute dependencies
  - `services/benchmark_streams.py` normalizes and limits row streams and builds benchmark batches
  - `services/benchmark_spool.py` provides replayable on-disk row spooling for multi-trial/tokenizer runs without full in-memory dataset materialization

### Key Module Responsibilities
- `server/app.py`: FastAPI app factory, router registration, SPA serving in Tauri mode.
- `server/api/*`: HTTP contracts, status codes, request/response models, job dispatch.
- `server/domain/*`: Pydantic/dataclass domain models and settings schemas.
- `server/services/*`: Business logic, long-running operations, orchestration.
  - `services/tokenizer_storage.py`: tokenizer identifier validation, cache path resolution, and Hugging Face URL construction shared by tokenizer workflows.
  - `services/dashboard_export_helpers.py`: dashboard export payload parsing and value formatting helpers used by the PDF export service.
  - `services/benchmark_engine.py`: warmup/timed trial batch runner with per-batch observations and cancellation checks.
- `server/repositories/database/*`: Backend selection and DB adapter implementations.
  - `repositories/database/backend.py`: `get_database()` is the single cached accessor for the configured backend; repositories receive or resolve this dependency instead of importing module-level database state.
- `server/repositories/schemas/*`: SQLAlchemy models and types.
- `server/repositories/serialization/*`: Persistence serialization and report materialization.
  - `repositories/serialization/benchmark_reports.py`: benchmark report persistence serialization and Pydantic response normalization.
- `server/repositories/frequencies.py`: temporary SQLite-backed frequency persistence used by metrics services for large vocabularies.
- `server/common/*`: constants, logging, type/util/security helpers.

Frontend structure:
- `src/pages/*`: page-level flows (`/dataset`, `/tokenizers`, `/cross-benchmark`)
- `src/components/*`: reusable UI blocks (wizards, banners, export modal, shell)
- `src/common/constants/*`: shared frontend constants (API endpoints, timeouts)
- `src/contexts/*`: state and operations for dataset/tokenizer workspaces
- `src/services/*`: API calls and response guards
- `src/workers/*`: background client-side word cloud layout worker

## Data Persistence
- Default embedded persistence:
  - SQLite file: `app/resources/database.db`
- Optional external persistence:
  - PostgreSQL via `postgresql+psycopg` when `DATABASE_EMBEDDED=false` in `settings/.env`
- Non-DB persisted artifacts:
  - `app/resources/sources/datasets` (download caches/uploads)
  - `app/resources/sources/tokenizers` (tokenizer caches/custom uploads)
  - `app/resources/logs` (runtime logs)

## Async vs Sync Behavior
- FastAPI endpoints are mostly `async def`.
- Blocking logic is intentionally offloaded with `await asyncio.to_thread(...)`.
- Long-running operations (download/analysis/benchmark/report generation) run in background threads via `JobManager`.
- Job polling/cancel operations are synchronous handler functions over in-memory job state.
- Repository/DB operations are synchronous SQLAlchemy session usage.
- Constraint:
  - Async handlers must not execute CPU-heavy or blocking I/O inline; they should offload to threads or job system.

## Runtime Interaction Topology
- Local webapp mode:
  - Browser -> Vite preview (`UI_HOST:UI_PORT`) -> proxied `/api` -> FastAPI (`FASTAPI_HOST:FASTAPI_PORT`)
- Desktop mode:
  - Tauri webview boots local backend process and loads the local app URL; backend can serve packaged SPA when `TKBEN_TAURI_MODE=true`.

## Benchmark Contract Notes
- Benchmark run request config now includes tokenizer behavior flags and per-document controls:
  - `add_special_tokens`, `padding`, `truncation`, `max_length`
  - `store_per_document_stats`, `per_document_sample_size`
- Each tokenizer result includes status and optional error details for failure isolation:
  - `status`, `error_type`, `error_message`
- Runtime metadata includes benchmark config echo and dataset scope details:
  - `dataset_total_documents_available`, `dataset_documents_benchmarked`, `benchmark_config`
  - `metric_availability` indicates whether metric families are measured/available for this run payload
- Benchmark efficiency payload includes boundary-separated timing fields:
  - `encode_only_wall_time_seconds`
  - `dataset_stream_wall_time_seconds`
  - `postprocess_wall_time_seconds`
- Chart aggregations are derived from successful tokenizer results only (`status="success"`), so failed tokenizers do not appear as misleading zero-value bars.
