# Configuration
Last updated: 2026-09-16

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
- `BACKEND_LOGS_VISIBLE` (accepts only `true` or `false`; an interactive launcher session shows backend logs in a dedicated terminal when `true`, while redirected `-Launch` runs capture stdout and stderr under `app/resources/logs`; defaults to `true` when absent)
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

These values are startup and infrastructure configuration. They are not
returned by the Settings API, cannot be edited from the Settings page, and are
resolved once for the running process.

## Application Runtime Settings
Typed Pydantic models own the defaults for application behavior. The optional
application-managed file `<TKBEN_DATA_DIR>/runtime-settings.json` stores only
sparse user overrides. It is created under `app/resources` by default, is
ignored by Git, and is independent of whether TKBEN uses SQLite or PostgreSQL.
The file contains a schema version, revision, and nested overrides; defaults
are never duplicated in it.

The Settings page and `/api/settings` endpoints expose exactly these 13 fields:

- Tokenizers: default discovery limit, maximum discovery limit, discovery
  candidate cap, metadata candidate multiplier, and tokenizer upload limit.
- Data: histogram bins, dataset streaming batch size, dataset upload limit,
  download timeout, retry attempts, and retry backoff.
- Runtime: benchmark streaming batch size and new-job polling interval.

`datasets.log_interval`, `benchmarks.log_interval`, and
`jobs.terminal_retention_seconds` remain backend-only operational settings.
Dataset capability values such as allowed file extensions remain outside the
runtime Settings surface.

## Ownership and validation

Do not move values between startup configuration and runtime settings to work
around a validation error. `.env` is canonical for operational and
environment-specific values. Runtime requests are strict typed contracts with
backend-authoritative validation, including tokenizer cross-field constraints,
finite numeric values, and revision checks. Invalid startup input fails before
the application exposes its health endpoint; invalid runtime input is rejected
without changing the current snapshot.

## Configuration Differences
### Dev and Local Webapp
- Angular serves and proxies `/api` to the FastAPI host and port from the environment.
- `RELOAD=true` enables Uvicorn reload behavior.

### Persistence Toggle
- Database mode and connection fields always come from `settings/.env` through the `DATABASE_*` variables. This is the same source used by Alembic and the database initializer; runtime settings cannot override it.
- `DATABASE_EMBEDDED=true` uses SQLite (`<TKBEN_DATA_DIR>/database.db`; defaults to `app/resources/database.db`).
- `DATABASE_EMBEDDED=false` uses PostgreSQL with the explicit `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_NAME`, `DATABASE_USERNAME`, `DATABASE_PASSWORD`, `DATABASE_SSL`, and `DATABASE_SSL_CA` fields. The engine is fixed to `postgresql+psycopg`.

### Job Lifecycle
- `jobs.polling_interval` controls frontend polling guidance for async job status for newly created jobs. Existing jobs keep their captured interval.
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

After changing `settings/.env`, restart the backend (or relaunch TKBEN) so the
new process resolves a fresh startup snapshot. Changes saved from **Settings**
are applied to subsequently started operations without a restart. Existing
database rows, reports, and canonical tokenizer artifacts remain in the
configured resource root unless an explicit data-removal action is selected.
