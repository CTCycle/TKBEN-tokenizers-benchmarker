# Backend API
Last updated: 2026-06-02

## API Prefix
All routers are included with `prefix="/api"` during backend startup.

## Datasets
- `GET /api/datasets/list`
- `GET /api/datasets/metrics/catalog`
- `POST /api/datasets/download`
- `POST /api/datasets/upload`
- `POST /api/datasets/analyze`
- `GET /api/datasets/reports/latest`
- `GET /api/datasets/reports/{report_id}`
- `DELETE /api/datasets/delete`

## Tokenizers
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

## Benchmarks
- `POST /api/benchmarks/run`
- `GET /api/benchmarks/reports`
- `GET /api/benchmarks/reports/{report_id}`
- `GET /api/benchmarks/metrics/catalog`

## Jobs
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `DELETE /api/jobs/{job_id}`

## Hugging Face Keys
- `POST /api/keys`
- `GET /api/keys`
- `DELETE /api/keys/{key_id}`
- `POST /api/keys/{key_id}/activate`
- `POST /api/keys/{key_id}/deactivate`
- `POST /api/keys/{key_id}/reveal`

## Exports
- `POST /api/exports/dashboard/pdf`
