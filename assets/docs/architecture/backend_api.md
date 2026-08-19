# Backend API
Last updated: 2026-08-18

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

The predefined `c4` dataset maps to `allenai/c4`, configuration `en`, and the
streaming `train` split capped at the first 10,000 documents so the preset is
usable in the local SQLite workflow. The full repository remains available
through the manual Hugging Face dataset-ID workflow.

## Tokenizers
- `GET /api/tokenizers/settings`
- `GET /api/tokenizers/discover` — performs bounded Hugging Face repository discovery. `search`, `author`, `pipeline_tag`, required `include_tags`, `access=all|public|gated`, `sort`, and `limit` are passed to the installed Hub API where supported. The query requests `siblings` metadata and locally requires a usable root-level tokenizer artifact: `tokenizer.json` or a SentencePiece file with tokenizer/config metadata, `vocab.json` plus `merges.txt`, or `vocab.txt` with tokenizer/config metadata. Weight-only, metadata-only, nested-only, and artifact-less repositories are discarded before the response is built. Because artifact validation and existing `exclude_tags`, any-text-task, and vocabulary comparison/order filters are local, the provider query uses bounded overfetch and the configured candidate cap. Discovery remains metadata-only; tokenizer weights are never downloaded or model-loaded. The existing download loader remains the final `AutoTokenizer` compatibility check and removes failed cache artifacts. The response is `{items, count, fetched_count}` with structured repository metadata. Hugging Face failures return a sanitized HTTP 500 rather than a false successful empty result.
- `GET /api/tokenizers/settings` — returns discovery and metadata candidate limits plus the tokenizer upload limit.
- `GET /api/tokenizers/list` — returns `{tokenizers, count}`; each item includes `tokenizer_name`, `source=huggingface|custom`, `has_report`, and nullable `vocabulary_size`. Optional `search` (trimmed, max 160 characters), `source=all|huggingface|custom`, `vocabulary_size_operator=at_least|at_most`, and non-negative `vocabulary_size` filters are applied server-side.
- `POST /api/tokenizers/download` — background result includes `failed_details` with sanitized exception summaries; failed downloads remove incomplete cache artifacts.
- `POST /api/tokenizers/reports/generate`
- `GET /api/tokenizers/reports/latest`
- `GET /api/tokenizers/reports/{report_id}`
- `GET /api/tokenizers/reports/{report_id}/vocabulary`
- `POST /api/tokenizers/upload`
- `DELETE /api/tokenizers/custom`
- `DELETE /api/tokenizers/delete?tokenizer_name={name}` — removes a downloaded tokenizer and its cached artifacts

## Benchmarks
- `POST /api/benchmarks/run`
- `GET /api/benchmarks/reports` — returns `{reports, total, offset, limit}`. Optional `search` is applied server-side to `run_name` and `Dataset.name`; `sort=newest|oldest`, `offset`, and `limit` provide deterministic server pagination. List queries fetch summary columns only and never select the JSON payload.
- `GET /api/benchmarks/reports/{report_id}`
- `DELETE /api/benchmarks/reports/{report_id}` — physically deletes the persisted report and returns `204`; nonexistent reports return `404`.
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

Service ownership is explicit: tokenizer discovery/catalog/download/cache and custom-tokenizer workflows remain in `TokenizersService`, while report analysis/generation/retrieval is owned by `TokenizerReportingService`. No legacy service aliases or forwarding methods are part of the API architecture.

Frontend API services keep endpoint construction centralized and pass responses
through typed guards before page-level rendering. Invalid or non-finite metric
values remain unavailable rather than being converted into synthetic zeroes.
