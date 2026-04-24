# ARCHITECTURE
Last updated: 2026-04-24

## System Summary
TKBEN is a tokenizer benchmarking platform with:
- FastAPI backend (`TKBEN/server`)
- React + Vite frontend (`TKBEN/client`)
- Optional Tauri desktop packaging (`TKBEN/client/src-tauri`)
- Shared local resources and settings (`TKBEN/resources`, `TKBEN/settings`)

Backend APIs are mounted under `/api/*`. Frontend calls `/api` and relies on Vite proxy in dev/preview.

## Repository Structure
Source-level structure (generated folders like `node_modules`, `dist`, caches omitted):

```text
.
├─ pyproject.toml
├─ runtimes/
│  ├─ .venv/                # Runtime virtualenv used by scripts
│  └─ uv.lock               # Runtime lockfile used by launcher/packaging
├─ assets/
│  ├─ docs/
│  └─ figures/
├─ TKBEN/
│  ├─ start_on_windows.bat
│  ├─ setup_and_maintenance.bat
│  ├─ client/
│  │  ├─ package.json
│  │  ├─ vite.config.ts
│  │  ├─ src/
│  │  │  ├─ main.tsx
│  │  │  ├─ App.tsx
│  │  │  ├─ App.css
│  │  │  ├─ index.css
│  │  │  ├─ components/
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
│  │  ├─ app.py
│  │  ├─ api/
│  │  ├─ configurations/
│  │  ├─ domain/
│  │  ├─ services/
│  │  ├─ repositories/
│  │  │  ├─ database/
│  │  │  ├─ schemas/
│  │  │  └─ serialization/
│  │  └─ common/
│  ├─ scripts/
│  │  └─ initialize_database.py
│  ├─ settings/
│  │  ├─ .env
│  │  ├─ .env.example
│  │  └─ configurations.json
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
└─ tests/
   ├─ run_tests.bat
   ├─ conftest.py
   ├─ unit/
   └─ e2e/
```

## Application Entry Points
- Backend app factory/module:
  - `TKBEN.server.app:app` in [app.py](/G:/Projects/Repositories/Active%20projects/TKBEN%20Benchmarker/TKBEN/server/app.py)
- Frontend entry:
  - [main.tsx](/G:/Projects/Repositories/Active%20projects/TKBEN%20Benchmarker/TKBEN/client/src/main.tsx)
- Frontend routing root:
  - [App.tsx](/G:/Projects/Repositories/Active%20projects/TKBEN%20Benchmarker/TKBEN/client/src/App.tsx)
- Desktop runtime entry:
  - [main.rs](/G:/Projects/Repositories/Active%20projects/TKBEN%20Benchmarker/TKBEN/client/src-tauri/src/main.rs)
- Windows local launcher:
  - [start_on_windows.bat](/G:/Projects/Repositories/Active%20projects/TKBEN%20Benchmarker/TKBEN/start_on_windows.bat)

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

### Key Module Responsibilities
- `server/app.py`: FastAPI app construction, router registration, SPA serving in Tauri mode.
- `server/api/*`: HTTP contracts, status codes, request/response models, job dispatch.
- `server/domain/*`: Pydantic/dataclass domain models and settings schemas.
- `server/services/*`: Business logic, long-running operations, orchestration.
  - `services/tokenizer_storage.py`: tokenizer identifier validation, cache path resolution, and Hugging Face URL construction shared by tokenizer workflows.
  - `services/dashboard_export_helpers.py`: dashboard export payload parsing and value formatting helpers used by the PDF export service.
- `server/repositories/database/*`: Backend selection and DB adapter implementations.
- `server/repositories/schemas/*`: SQLAlchemy models and types.
- `server/repositories/serialization/*`: Persistence serialization and report materialization.
  - `repositories/serialization/benchmark_reports.py`: benchmark report persistence serialization and Pydantic response normalization.
- `server/common/*`: constants, logging, type/util/security helpers.

Frontend structure:
- `src/pages/*`: page-level flows (`/dataset`, `/tokenizers`, `/cross-benchmark`)
- `src/components/*`: reusable UI blocks (wizards, banners, export modal, shell)
- `src/contexts/*`: state and operations for dataset/tokenizer workspaces
- `src/services/*`: API calls and response guards
- `src/workers/*`: background client-side word cloud layout worker

## Data Persistence
- Default embedded persistence:
  - SQLite file: `TKBEN/resources/database.db`
- Optional external persistence:
  - PostgreSQL via `postgresql+psycopg` when `embedded_database=false` in `settings/configurations.json`
- Non-DB persisted artifacts:
  - `TKBEN/resources/sources/datasets` (download caches/uploads)
  - `TKBEN/resources/sources/tokenizers` (tokenizer caches/custom uploads)
  - `TKBEN/resources/logs` (runtime logs)

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
