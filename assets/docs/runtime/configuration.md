# Configuration
Last updated: 2026-09-13

## Environment File
Primary launcher runtime env file:
- `settings/.env`
- Created from the versioned `settings/.env.example` template when missing
- Owns environment-specific runtime values, including ports, database mode and
  connection fields, filesystem roots, logging, and boolean controls.
- The backend bootstrap loads this file before importing configuration and
  database modules that can resolve environment-derived paths or settings.

## Core Variables
- `FASTAPI_HOST`
- `FASTAPI_PORT`
- `UI_HOST`
- `UI_PORT`
- `VITE_API_BASE_URL` (default `/api`)
- `RELOAD`
- `BACKEND_LOGS_VISIBLE` (accepts only `true` or `false`; shows backend logs in a dedicated terminal when `true`, and defaults to `true` when absent)
- `ALLOW_KEY_REVEAL`
- `HF_KEYS_ENCRYPTION_MATERIAL_FILE`
- `TKBEN_DATA_DIR` (resource root for the embedded database, datasets, tokenizers, and exports; defaults to `app/resources`)
- `TKBEN_LOG_DIR` (runtime logs)
- `DATABASE_EMBEDDED`
- `DATABASE_HOST`
- `DATABASE_PORT`
- `DATABASE_NAME`
- `DATABASE_USERNAME`
- `DATABASE_PASSWORD`
- `DATABASE_SSL`
- `DATABASE_SSL_CA`
- `DATABASE_CONNECT_TIMEOUT`
- `DATABASE_INSERT_BATCH_SIZE`
- `jobs.terminal_retention_seconds` in `settings/configurations.json`

## Structured Settings
- `settings/configurations.json`
  - `datasets`, `tokenizers`, `benchmarks`, and `jobs`
  - Unknown top-level blocks, including a `database` block, are rejected.
- Every required structured block and field must be present. The validated
  configuration is resolved into one immutable `ServerSettings` snapshot for
  application startup; it is not a second source for environment values.

## Ownership and validation

Do not move values between the two configuration files to work around a
validation error. `.env` is canonical for operational and environment-specific
values; `configurations.json` is canonical for structured application tuning.
Pydantic validation rejects missing or unknown structured settings, and runtime
booleans accept only `true` or `false`. Invalid input fails before the
application exposes its health endpoint.

## Configuration Differences
### Dev and Local Webapp
- Angular serves and proxies `/api` to the FastAPI host and port from the environment.
- `RELOAD=true` enables Uvicorn reload behavior.

### Persistence Toggle
- Database mode and connection fields always come from `settings/.env` through the `DATABASE_*` variables. This is the same source used by Alembic and the database initializer; `configurations.json` cannot override it.
- `DATABASE_EMBEDDED=true` uses SQLite (`<TKBEN_DATA_DIR>/database.db`; defaults to `app/resources/database.db`).
- `DATABASE_EMBEDDED=false` uses PostgreSQL with the explicit `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_NAME`, `DATABASE_USERNAME`, `DATABASE_PASSWORD`, `DATABASE_SSL`, and `DATABASE_SSL_CA` fields. The engine is fixed to `postgresql+psycopg`.

### Job Retention
- `jobs.polling_interval` controls frontend polling guidance for async job status.
- `jobs.terminal_retention_seconds` controls how long completed, failed, and cancelled in-memory jobs remain visible before pruning.

### Upload Limits
- `datasets.max_upload_bytes` and `tokenizers.max_upload_bytes` are enforced while reading upload streams.
- Uploads that exceed the configured limit return HTTP 413 before dispatching a job or tokenizer import workflow.

### Security Controls
- `ALLOW_KEY_REVEAL=false` keeps plaintext Hugging Face key reveal disabled by default.
- Hugging Face access-key encryption material is generated and persisted in the external JSON file configured by `HF_KEYS_ENCRYPTION_MATERIAL_FILE` (default `app/resources/hf-key-material.json`). Keep this file private and do not copy it into database backups.
- `FASTAPI_HOST` controls the interface on which the backend listens. Network exposure requires appropriate deployment-level access controls.

Boolean launcher settings are validated as `true` or `false`; invalid values
fail fast. The versioned template defaults are `FASTAPI_PORT=5000` and
`UI_PORT=8000`, and the same values are used by the test harness.

After changing `settings/.env` or `settings/configurations.json`, restart the
backend (or relaunch TKBEN) so the new process resolves a fresh settings
snapshot. Existing database rows, reports, and canonical tokenizer artifacts
remain in the configured resource root unless an explicit data-removal action
is selected.
