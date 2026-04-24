# RUNTIME_MODES
Last updated: 2026-04-24

## Supported Modes
### 1. Local webapp mode (default)
- Backend: FastAPI (`TKBEN.server.app:app`)
- Frontend: Vite preview build (`TKBEN/client/dist`)
- Primary launcher: `TKBEN/start_on_windows.bat`

### 2. Desktop packaged mode (Tauri)
- Windows packaged desktop app using `TKBEN/client/src-tauri`.
- Bundles backend/frontend/resources/runtimes as Tauri resources.
- Build helper: `release/tauri/build_with_tauri.bat`

### 3. Test runtime mode
- Uses existing `runtimes/.venv` + local backend/frontend servers for pytest suites.
- Entry script: `tests/run_tests.bat`

### 4. Containerized mode
- Not implemented in current repository state (no active Docker runtime configuration in root).

## Startup Procedures
### Local webapp mode (Windows recommended)
```bat
.\TKBEN\start_on_windows.bat
```
What it does:
- installs/uses portable Python, uv, Node.js under `runtimes`
- syncs Python deps (`uv sync`) using `runtimes/uv.lock`
- installs frontend deps and builds frontend
- starts backend + frontend

### Manual local mode (cross-platform)
```bash
uv sync
uv run python -m uvicorn TKBEN.server.app:app --host 127.0.0.1 --port 5000
cd TKBEN/client
npm ci
npm run preview -- --host 127.0.0.1 --port 8000 --strictPort
```

### Desktop packaging mode (Windows)
```bat
copy /Y TKBEN\settings\.env.example TKBEN\settings\.env
.\release\tauri\build_with_tauri.bat
```

### Test mode
```bat
.\tests\run_tests.bat
```

## Environment Variables and Config Requirements
Primary runtime env file:
- `TKBEN/settings/.env` (seed from `.env.example`)

Core variables:
- `FASTAPI_HOST`
- `FASTAPI_PORT`
- `UI_HOST`
- `UI_PORT`
- `VITE_API_BASE_URL` (default `/api`)
- `RELOAD`
- `OPTIONAL_DEPENDENCIES`
- `ALLOW_KEY_REVEAL`
- `HF_KEYS_ENCRYPTION_KEY`

Structured settings:
- `TKBEN/settings/configurations.json`
  - `database` (embedded SQLite vs PostgreSQL)
  - `datasets`, `tokenizers`, `benchmarks`, `jobs` tunables

## Dependency Prerequisites
From project/runtime scripts and metadata:
- Python `>=3.14` (launcher currently installs 3.14.2 embedded on Windows)
- Node.js `>=22` (launcher currently uses 22.12.0)
- `uv`
- Rust/Cargo required only for Tauri packaging

## Configuration Differences
### Dev/local webapp
- Vite serves/proxies `/api` to FastAPI host/port from env.
- `RELOAD=true` enables Uvicorn reload behavior.

### Desktop packaged
- Tauri launches backend process locally and sets `TKBEN_TAURI_MODE=true`.
- Backend may serve packaged SPA assets when packaged client dist is available.
- Runtime environment may be prepared into writable runtime path for packaged execution.

### Persistence mode toggle
- `configurations.json`:
  - `database.embedded_database=true` -> SQLite (`resources/database.db`)
  - `database.embedded_database=false` + `engine=postgresql+psycopg` -> PostgreSQL

## Interoperability
- Frontend and backend communicate through HTTP JSON APIs under `/api/*`.
- In local webapp mode, Vite proxy rewrites `/api/*` to backend root.
- In desktop mode, Tauri manages backend process lifecycle and points UI to local backend URL.
- Shared resources:
  - database (`resources/database.db`)
  - downloaded datasets/tokenizers (`resources/sources/*`)
  - templates/logs (`resources/templates`, `resources/logs`)

## Limitations and Constraints
- Desktop local backend bootstrap in Tauri is Windows-only in current Rust implementation.
- Long-running operations are asynchronous jobs and require polling via `/api/jobs/{job_id}`.
- Large download/processing operations depend on local network and disk throughput.
- `ALLOW_KEY_REVEAL` controls whether HF keys can be revealed via API (`/api/keys/{id}/reveal`).

## Deployment and Packaging Notes
### Desktop build outputs
- `release/windows/installers`
- `release/windows/portable`

### Packaging flow summary
- Build frontend (`npm run build`)
- Build Tauri app (`npm run tauri:build:release`)
- Export artifacts using `release/tauri/scripts/export-windows-artifacts.ps1`

### Local distribution strategy
- For non-packaged use, repository + `start_on_windows.bat` is the operational deployment path.
