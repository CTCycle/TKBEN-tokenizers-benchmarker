# Configuration
Last updated: 2026-06-02

## Environment File
Primary runtime env file:
- `TKBEN/settings/.env`
- Seed from `TKBEN/settings/.env.example`

## Core Variables
- `FASTAPI_HOST`
- `FASTAPI_PORT`
- `UI_HOST`
- `UI_PORT`
- `VITE_API_BASE_URL` (default `/api`)
- `RELOAD`
- `OPTIONAL_DEPENDENCIES`
- `ALLOW_KEY_REVEAL`
- `HF_KEYS_ENCRYPTION_KEY`
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

## Structured Settings
- `TKBEN/settings/configurations.json`
  - `datasets`, `tokenizers`, `benchmarks`, and `jobs` tunables

## Configuration Differences
### Dev and Local Webapp
- Vite serves and proxies `/api` to the FastAPI host and port from the environment.
- `RELOAD=true` enables Uvicorn reload behavior.

### Desktop Packaged
- Tauri launches the backend process locally and sets `TKBEN_TAURI_MODE=true`.
- Backend may serve packaged SPA assets when packaged client dist is available.
- Runtime environment may be prepared into a writable runtime path for packaged execution.

### Persistence Toggle
- `DATABASE_EMBEDDED=true` uses SQLite (`resources/database.db`).
- `DATABASE_EMBEDDED=false` with `DATABASE_ENGINE=postgresql+psycopg` uses PostgreSQL.
- `DATABASE_URL` may seed engine, host, port, name, user, and password values, with explicit `DATABASE_*` values taking precedence.
