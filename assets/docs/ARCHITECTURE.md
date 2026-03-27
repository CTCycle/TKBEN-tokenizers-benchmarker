# TKBEN Architecture

## 1. Purpose
TKBEN is a tokenizer benchmarking web app. It supports:
- dataset ingestion and validation
- tokenizer discovery/download/reporting
- cross-tokenizer benchmark runs with persisted reports

## 2. High-Level System
- Frontend: React + TypeScript SPA (`TKBEN/client`)
- Backend: FastAPI service (`TKBEN/server`)
- Persistence: SQLite by default (`TKBEN/resources/database.db`), optional PostgreSQL
- External integrations: Hugging Face datasets/models APIs

## 3. Frontend Structure
- App shell and routing:
  - `TKBEN/client/src/App.tsx`
  - routes: `/dataset`, `/tokenizers`, `/cross-benchmark`
- State:
  - `TKBEN/client/src/contexts/DatasetContext.tsx`
  - `TKBEN/client/src/contexts/TokenizersContext.tsx`
- API clients:
  - `TKBEN/client/src/services/*`
  - all calls use `/api` base path (proxied in local dev/preview and preserved in desktop packaging)
- Key pages:
  - `DatasetPage.tsx`
  - `TokenizerExaminationPage.tsx` (includes `TokenizersPage.tsx`)
  - `CrossBenchmarkPage.tsx`

## 4. Backend Structure
- App entrypoint: `TKBEN/server/app.py`
- Routers: `TKBEN/server/routes`
  - `datasets.py`
  - `tokenizers.py`
  - `benchmarks.py`
  - `jobs.py`
  - `keys.py`
- Services: `TKBEN/server/services`
  - dataset processing, tokenizer handling, benchmark execution, key management, job orchestration
- Entities (request/response models): `TKBEN/server/entities`
- Repositories and schema:
  - `TKBEN/server/repositories/database`
  - `TKBEN/server/repositories/schemas/models.py`

## 5. API Surface
Root:
- `GET /` -> redirects to `/docs`

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
- `POST /keys/{key_id}/reveal`

## 6. Async Job Pattern
Long-running operations return `JobStartResponse` and are polled via `/jobs/{job_id}` until:
- `completed`
- `failed`
- `cancelled`

This pattern is used for dataset load/analysis, tokenizer download/report generation, and benchmark execution.

## 6.1 Dataset Download Resilience
Dataset download jobs apply bounded resilience controls from `TKBEN/settings/configurations.json`:
- `datasets.download_timeout_seconds`: per-attempt timeout for external dataset loading.
- `datasets.download_retry_attempts`: maximum retry attempts for transient/network failures.
- `datasets.download_retry_backoff_seconds`: exponential backoff base delay between retry attempts.

## 7. Database Model (Current)
Main tables in `models.py`:
- `dataset`, `dataset_document`
- `analysis_session`, `metric_type`, `metric_value`, `histogram_artifact`
- `tokenizer`, `tokenizer_report`, `tokenizer_vocabulary`, `tokenizer_vocabulary_statistics`
- `tokenization_document_stats`, `tokenization_dataset_stats`, `tokenization_dataset_stats_detail`
- `benchmark_report`
- `hf_access_keys`

## 8. Runtime and Packaging
- Local launcher: `TKBEN/start_on_windows.bat`
- Maintenance utilities: `TKBEN/setup_and_maintenance.bat`
- Desktop packaging build: `release/tauri/build_with_tauri.bat`
- Desktop runtime serves API routes under original paths and `/api`

## 9. Current Constraints
- No application-level auth for datasets/tokenizers/benchmarks endpoints.
- HF access keys are managed server-side, but API access itself is not user-authenticated.
- Uploaded custom tokenizers are kept in process memory and are not persisted across server restart.
