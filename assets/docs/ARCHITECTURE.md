# TKBEN Architecture
Last updated: 2026-04-20

## 1. Purpose
TKBEN is a tokenizer benchmarking application that supports:
- dataset ingestion (Hugging Face download + local upload)
- dataset validation/analysis and persisted reports
- tokenizer discovery/download/reporting
- cross-tokenizer benchmark execution with persisted results
- dashboard PDF export for benchmark/report views

## 2. High-Level System
- Frontend: React + TypeScript SPA in `TKBEN/client`
- Backend: FastAPI service in `TKBEN/server`
- Persistence:
  - default embedded SQLite (`TKBEN/resources/database.db`)
  - optional PostgreSQL (`TKBEN/settings/configurations.json` -> `database.embedded_database=false`)
- External services: Hugging Face datasets/tokenizers APIs

## 3. Frontend Structure
- App shell and routing:
  - `TKBEN/client/src/App.tsx`
  - `TKBEN/client/src/components/AppShell.tsx`
  - routes: `/dataset`, `/tokenizers`, `/cross-benchmark`
- State contexts:
  - `TKBEN/client/src/contexts/DatasetContext.tsx`
  - `TKBEN/client/src/contexts/TokenizersContext.tsx`
- API client modules:
  - `TKBEN/client/src/services/*`
  - default API base: `/api`

## 4. Backend Structure
- App entrypoint: `TKBEN/server/app.py`
- API routers: `TKBEN/server/api`
  - `datasets.py`
  - `tokenizers.py`
  - `benchmarks.py`
  - `jobs.py`
  - `keys.py`
  - `exports.py`
- Domain models: `TKBEN/server/domain`
- Services: `TKBEN/server/services`
  - dataset service split:
    - `datasets.py` (download resolution, stream/stat helpers)
    - `dataset_operations.py` (persist/upload/analysis workflows)
  - benchmark service split:
    - `benchmarks.py` (tokenizer tools + core service wiring)
    - `benchmark_execution.py` (benchmark execution/persistence)
    - `benchmark_plotting.py` (plot generation helpers)
  - `custom_tokenizers.py` (thread-safe in-memory registry used by tokenizer upload + benchmark run flows)
- Persistence:
  - `TKBEN/server/repositories/database`
  - `TKBEN/server/repositories/schemas/models.py`

## 5. Routing Model
All backend API routes are exposed only under `/api`.

## 6. API Surface

Root behavior:
- `GET /`
  - local mode: redirects to `/docs`
  - packaged Tauri mode with built frontend: serves SPA entrypoint

Datasets:
- `GET /api/datasets/list`
- `GET /api/datasets/metrics/catalog`
- `POST /api/datasets/download` (async job start)
- `POST /api/datasets/upload` (async job start)
- `POST /api/datasets/analyze` (async job start)
- `GET /api/datasets/reports/latest`
- `GET /api/datasets/reports/{report_id}`
- `DELETE /api/datasets/delete`

Tokenizers:
- `GET /api/tokenizers/settings`
- `GET /api/tokenizers/scan`
- `GET /api/tokenizers/list`
- `POST /api/tokenizers/download` (async job start)
- `POST /api/tokenizers/reports/generate` (async job start)
- `GET /api/tokenizers/reports/latest`
- `GET /api/tokenizers/reports/{report_id}`
- `GET /api/tokenizers/reports/{report_id}/vocabulary`
- `POST /api/tokenizers/upload`
- `DELETE /api/tokenizers/custom`

Benchmarks:
- `POST /api/benchmarks/run` (async job start)
- `GET /api/benchmarks/reports`
- `GET /api/benchmarks/reports/{report_id}`
- `GET /api/benchmarks/metrics/catalog`

Benchmark payload contract:
- benchmark API responses use `report_version=2` payload fields.
- benchmark persistence table currently stores report metadata with default ORM `report_version=1` and JSON payload.
- active payload centers on:
  - `config` (warmup/timed trials, batch size, seed, parallelism)
  - `hardware_profile`
  - `trial_summary`
  - `tokenizer_results` grouped by efficiency/latency/fidelity/fragmentation/resources
  - `chart_data` with V2 series groups for efficiency/fidelity/vocabulary/fragmentation/distribution

Exports:
- `POST /api/exports/dashboard/pdf` (returns generated PDF bytes)

Jobs:
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `DELETE /api/jobs/{job_id}`

HF Keys:
- `POST /api/keys`
- `GET /api/keys`
- `DELETE /api/keys/{key_id}`
- `POST /api/keys/{key_id}/activate`
- `POST /api/keys/{key_id}/deactivate`
- `POST /api/keys/{key_id}/reveal` (guarded by `ALLOW_KEY_REVEAL`)

## 7. Async Job Pattern
Long-running operations return `JobStartResponse` and are polled via `/api/jobs/{job_id}`.

Job terminal states:
- `completed`
- `failed`
- `cancelled`

Covered workflows:
- dataset download/upload/analyze
- tokenizer download/report generation
- benchmark run

## 8. Database Tables (Current)
Main ORM tables in `TKBEN/server/repositories/schemas/models.py`:
- `dataset`, `dataset_document`
- `analysis_session`, `metric_type`, `metric_value`, `histogram_artifact`
- `dataset_validation_report`
- `tokenizer`, `tokenizer_report`, `tokenizer_vocabulary`, `tokenizer_vocabulary_statistics`
- `tokenization_document_stats`, `tokenization_dataset_stats`, `tokenization_dataset_stats_detail`
- `benchmark_report`
- `hf_access_keys`

## 9. Runtime and Packaging Anchors
- Local bootstrap/launcher: `TKBEN/start_on_windows.bat`
- Maintenance utility: `TKBEN/setup_and_maintenance.bat`
- Desktop build entrypoint: `release/tauri/build_with_tauri.bat`
- Packaged artifacts output: `release/windows/installers`, `release/windows/portable`

## 10. Known Constraints
- API endpoints do not implement user/session auth.
- HF keys are encrypted at rest but API access itself is not user-authenticated.
- Uploaded custom tokenizers are in-memory only and are not persisted across backend restart.
