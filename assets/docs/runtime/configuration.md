# Configuration
Last updated: 2026-07-12

## Environment File
Primary launcher runtime env file:
- `settings/.env`
- Created from the versioned `settings/.env.example` template when missing

## Core Variables
- `FASTAPI_HOST`
- `FASTAPI_PORT`
- `TKBEN_ALLOW_UNAUTHENTICATED_NETWORK_BIND`
- `UI_HOST`
- `UI_PORT`
- `VITE_API_BASE_URL` (default `/api`)
- `RELOAD`
- `OPTIONAL_DEPENDENCIES`
- `BACKEND_LOGS_VISIBLE` (accepts only `true` or `false`; shows backend logs in a dedicated terminal when `true`, and defaults to `true` when absent)
- `ALLOW_KEY_REVEAL`
- `HF_KEYS_ENCRYPTION_KEY`
- `TKBEN_DATA_DIR` (mutable database, datasets, tokenizers, and exports)
- `TKBEN_LOG_DIR` (runtime logs)
- `TKBEN_CONFIG_DIR` (active `.env` and `configurations.json`)
- `DATABASE_EMBEDDED`
- `DATABASE_URL`
- `DATABASE_ENGINE`
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
- `TKBEN/settings/configurations.json`
  - `datasets`, `tokenizers`, `benchmarks`, `jobs`, and optional `database` overrides

## Configuration Differences
### Dev and Local Webapp
- Vite serves and proxies `/api` to the FastAPI host and port from the environment.
- `RELOAD=true` enables Uvicorn reload behavior.

### Persistence Toggle
- If `database` is present in `settings/configurations.json`, that block is authoritative for database mode and connection fields.
- Otherwise the backend falls back to `DATABASE_*` environment variables.
- `DATABASE_EMBEDDED=true` uses SQLite (`resources/database.db`).
- `DATABASE_EMBEDDED=false` with `DATABASE_ENGINE=postgresql+psycopg` uses PostgreSQL.
- `DATABASE_URL` may seed engine, host, port, name, user, and password values when no structured database block is supplied.

### Job Retention
- `jobs.polling_interval` controls frontend polling guidance for async job status.
- `jobs.terminal_retention_seconds` controls how long completed, failed, and cancelled in-memory jobs remain visible before pruning.

### Upload Limits
- `datasets.max_upload_bytes` and `tokenizers.max_upload_bytes` are enforced while reading upload streams.
- Uploads that exceed the configured limit return HTTP 413 before dispatching a job or tokenizer import workflow.

### Security Controls
- `ALLOW_KEY_REVEAL=false` keeps plaintext Hugging Face key reveal disabled by default.
- `FASTAPI_HOST` should remain loopback (`127.0.0.1`, `localhost`, or `::1`) for the built-in unauthenticated local runtime.
- Setting `TKBEN_ALLOW_UNAUTHENTICATED_NETWORK_BIND=true` bypasses the loopback startup guard and should only be used behind an external authentication boundary.
