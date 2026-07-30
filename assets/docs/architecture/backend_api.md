# Backend API
Last updated: 2026-07-29

## API Prefix
All routers are included with `prefix="/api"` during backend startup.

## Health
- `GET /api/health`

## Datasets
- `GET /api/datasets/list` — returns `{datasets, count}`; optional `search` (trimmed, max 160 characters), `source=all|public|custom`, `document_count_operator=at_least|at_most`, and non-negative `document_count` filters. Filtering is applied server-side to the catalog.
- `GET /api/datasets/metrics/catalog`
- `POST /api/datasets/download`
- `POST /api/datasets/upload`
- `POST /api/datasets/analyze`
- `GET /api/datasets/reports/latest`
- `GET /api/datasets/reports/{report_id}`
- `DELETE /api/datasets/delete`

## Tokenizers
- `GET /api/tokenizers/settings`
- `GET /api/tokenizers/scan` — returns Hugging Face models with supported text pipeline tags only; multimodal, image, audio, and untyped repositories are excluded.
- `GET /api/tokenizers/list` — returns `{tokenizers, count}`; each item includes `tokenizer_name`, `source=huggingface|custom`, `has_report`, and nullable `vocabulary_size`. Optional `search` (trimmed, max 160 characters), `source=all|huggingface|custom`, `vocabulary_size_operator=at_least|at_most`, and non-negative `vocabulary_size` filters are applied server-side.
- `POST /api/tokenizers/download` — background result includes `failed_details` with sanitized exception summaries; failed downloads remove incomplete cache artifacts.
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
  - dashboard widgets use report-v5/schema-3 `default_visualization`, ordered `compatible_visualizations`, and persisted `histogram_bins`; older reports are not listed or loaded

## Jobs
- `GET /api/jobs/{job_id}`
- `POST /api/jobs/{job_id}/cancel` — requests cooperative cancellation for an active job; status polling reports `cancelled` once the runner exits.

## Hugging Face Keys
- `POST /api/keys`
- `GET /api/keys`
- `DELETE /api/keys/{key_id}`
- `POST /api/keys/{key_id}/activate`
- `POST /api/keys/{key_id}/deactivate`
- `POST /api/keys/{key_id}/reveal`

## Exports
- `POST /api/exports/dashboard/pdf` — benchmark payloads include `visualization_by_widget_id`; the server rejects unknown or incompatible visualization overrides
