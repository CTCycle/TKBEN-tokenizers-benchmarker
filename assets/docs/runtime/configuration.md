# Configuration
Last updated: 2026-10-21

## Environment File
Primary launcher runtime env file:
- `settings/.env`
- Created from the versioned `settings/.env.example` template when missing
- Owns bootstrap-only values, including listeners, filesystem roots, logging,
  security controls, and process behavior. Database settings are never loaded
  from this file or the process environment.
- The backend bootstrap loads this file before importing configuration and
  database modules that can resolve environment-derived paths or settings.

## Core Variables
- `FASTAPI_HOST` and `FASTAPI_PORT` (production FastAPI listener and same-origin
  SPA/API URL)
- `UI_HOST` and `UI_PORT` (manual Angular development server only; production
  launch does not bind or check a UI port)
- `VITE_API_BASE_URL` (default `/api`)
- `CORS_ORIGINS` (optional comma-separated HTTP(S) origins without paths or
  credentials; blank uses origins derived from `UI_HOST` and `UI_PORT` for
  direct cross-origin development or API access)
- `ALLOW_KEY_REVEAL`
- `HF_KEYS_ENCRYPTION_MATERIAL_FILE`
- `TKBEN_DATA_DIR` (persistent application data root and default SQLite database directory; downloads, tokenizer artifacts, and exports also use it; defaults to `data`; set it to a repository-relative or absolute path to use a different root; it is separate from the disposable `runtimes/cache` root)
- `TKBEN_LOG_DIR` (runtime logs)

These values are startup and infrastructure configuration. Database
connection settings are managed separately in
`<TKBEN_DATA_DIR>/database-config.json` through Settings → Database.
An explicit `CORS_ORIGINS` list replaces the derived origins. Invalid entries
fail bootstrap; restart the backend after changing this value.

## Application Runtime Settings
Typed Pydantic models own the defaults for application behavior. The active
database stores sparse user overrides and their revision in its
`application_runtime_settings` table. Each database has independent settings;
switching reloads that database's values and does not copy data. Runtime
settings are never read from a file.

Dashboard widget order, visibility, chart choices, and report-specific baseline
selections use the same database row with an independent revision. These
preferences save as the user changes them and follow the active database.
UI-managed application settings exclude network bootstrap values such as API
base paths, listener addresses and ports, and CORS origins.

The Settings page and `/api/settings` endpoints expose exactly these 16 fields,
grouped in the UI by workflow:

- Data: histogram bins, dataset upload limit, download timeout, retry attempts,
  and retry backoff.
- Tokenizers: default discovery limit, maximum discovery limit, discovery
  candidate cap, metadata candidate multiplier, and tokenizer upload limit.
- Benchmarks: default document cap, default tokenizer batch size, default
  parallelism, and benchmark streaming batch size.
- Runtime: dataset streaming batch size and new-job polling interval.

The three benchmark defaults prefill newly opened benchmark-run forms and can
still be changed for an individual run. They do not alter persisted reports or
already-open operations. Benchmark methodology choices such as warmup trials,
timed trials, seed, tokenization flags, and per-document statistics remain
explicit per-run controls rather than global application settings.

`datasets.log_interval`, `benchmarks.log_interval`, and
`jobs.terminal_retention_seconds` remain backend-only operational settings.
Dataset capability values such as allowed file extensions remain outside the
runtime Settings surface.

## Ownership and validation

Do not move values between startup configuration and runtime settings to work
around a validation error. `.env` is canonical for bootstrap and
environment-specific values. UI-managed application runtime settings are not
read from `.env`. Runtime requests are strict typed contracts with
backend-authoritative validation, including tokenizer cross-field constraints,
finite numeric values, and revision checks. Invalid bootstrap input fails
before the application exposes its health endpoint; an unavailable selected
database instead starts in repair mode. Invalid runtime input is rejected
without changing the current snapshot.

## Configuration Differences
### Dev and Local Webapp
- In production, FastAPI serves the built Angular SPA and `/api` on
  `FASTAPI_HOST:FASTAPI_PORT`; the browser uses that single origin.
- In manual development, Angular CLI serves the frontend at `UI_HOST:UI_PORT`
  and proxies `/api` to `FASTAPI_HOST:FASTAPI_PORT`. Its proxy defaults to
  `127.0.0.1:5000` and reads the FastAPI values from the frontend process
  environment.
- Start development Uvicorn with `--reload`. The production launcher always
  runs FastAPI without reload mode.

### Database Settings
- Settings → Database supports SQLite and PostgreSQL. SQLite paths, PostgreSQL host/port/name/user/password/SSL, connection timeout, and insert batch size are persisted in `<TKBEN_DATA_DIR>/database-config.json`; PostgreSQL passwords are encrypted with a dedicated Fernet purpose key in the existing local key-material file.
- The UI requires a successful connection test for the exact current configuration before the user can confirm a switch. The switch API revalidates and initializes the target schema, drains active requests, refuses while managed jobs are pending/running, and restores the previous backend and file configuration if activation fails.
- Switching reloads the selected database's settings and records. It never copies data between databases. The browser reload clears cached feature state after switching.
- If the selected database is unavailable at startup, the API remains available in a limited mode so the Database section can test or replace the connection; other database-backed routes return HTTP 503 until a database is activated.
- Restart is still required after changing bootstrap settings such as listener ports, data/log paths, CORS origins, or the key-material-file path. Database switching and saved runtime settings apply in the current process when their validation and drain checks succeed.

### Job Lifecycle
- `jobs.polling_interval` controls frontend polling guidance for async job status for newly created jobs. Existing jobs keep their captured interval.
- `jobs.terminal_retention_seconds` controls how long completed, failed, and cancelled in-memory jobs remain visible before pruning.

### Upload Limits
- `datasets.max_upload_bytes` and `tokenizers.max_upload_bytes` are enforced while reading upload streams.
- Uploads that exceed the configured limit return HTTP 413 before dispatching a job or tokenizer import workflow.

### Security Controls
- `ALLOW_KEY_REVEAL=false` keeps plaintext Hugging Face key reveal disabled by default.
- Hugging Face access-key encryption material is generated and persisted in the external JSON file configured by `HF_KEYS_ENCRYPTION_MATERIAL_FILE` (default `data/hf-key-material.json`). Keep this file private and do not copy it into database backups.
- `FASTAPI_HOST` controls the interface on which the backend listens. Network exposure requires appropriate deployment-level access controls.

Boolean launcher settings are validated as `true` or `false`; invalid values
fail fast. The versioned template defaults are `FASTAPI_PORT=5000` and
`UI_PORT=8000`. `UI_PORT` remains available for development and test workflows.

After changing bootstrap values in `settings/.env`, restart the backend (or
relaunch TKBEN) so the new process resolves a fresh startup snapshot. Changes
saved from **Settings** apply to subsequently started operations without a
restart. Database switches activate live when request-drain and job checks
succeed. Existing database rows, reports, and canonical tokenizer artifacts
remain in their selected database unless an explicit data-removal action is
selected.
