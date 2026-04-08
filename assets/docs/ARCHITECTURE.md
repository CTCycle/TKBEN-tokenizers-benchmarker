# TKBEN Architecture
Last updated: 2026-04-08

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
- Persistence:
  - `TKBEN/server/repositories/database`
  - `TKBEN/server/repositories/schemas/models.py`

## 5. Routing Model
Every API router is registered twice:
- direct path (for example `/datasets/list`)
- `/api`-prefixed alias (for example `/api/datasets/list`)

This keeps compatibility across local web mode, tests, and packaged desktop mode.

## 6. API Surface

Root behavior:
- `GET /`
  - local mode: redirects to `/docs`
  - packaged Tauri mode with built frontend: serves SPA entrypoint

Datasets:
- `GET /datasets/list`
- `GET /datasets/metrics/catalog`
- `POST /datasets/download` (async job start)
- `POST /datasets/upload` (async job start)
- `POST /datasets/analyze` (async job start)
- `GET /datasets/reports/latest`
- `GET /datasets/reports/{report_id}`
- `DELETE /datasets/delete`

Tokenizers:
- `GET /tokenizers/settings`
- `GET /tokenizers/scan`
- `GET /tokenizers/list`
- `POST /tokenizers/download` (async job start)
- `POST /tokenizers/reports/generate` (async job start)
- `GET /tokenizers/reports/latest`
- `GET /tokenizers/reports/{report_id}`
- `GET /tokenizers/reports/{report_id}/vocabulary`
- `POST /tokenizers/upload`
- `DELETE /tokenizers/custom`

Benchmarks:
- `POST /benchmarks/run` (async job start)
- `GET /benchmarks/reports`
- `GET /benchmarks/reports/{report_id}`
- `GET /benchmarks/metrics/catalog`

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
- `POST /exports/dashboard/pdf` (returns generated PDF bytes)

Jobs:
- `GET /jobs`
- `GET /jobs/{job_id}`
- `DELETE /jobs/{job_id}`

HF Keys:
- `POST /keys`
- `GET /keys`
- `DELETE /keys/{key_id}`
- `POST /keys/{key_id}/activate`
- `POST /keys/{key_id}/deactivate`
- `POST /keys/{key_id}/reveal` (guarded by `ALLOW_KEY_REVEAL`)

## 7. Async Job Pattern
Long-running operations return `JobStartResponse` and are polled via `/jobs/{job_id}`.

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
