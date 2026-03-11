# Packaging and Runtime Modes (TKBEN)

## 1. Runtime Profiles
Active profile:
- `TKBEN/settings/.env`

Profile templates:
- `TKBEN/settings/.env.local.example`
- `TKBEN/settings/.env.local.tauri.example`
- `TKBEN/settings/.env.cloud.example`

Non-env defaults:
- `TKBEN/settings/configurations.json`

## 2. Local Mode (Default)
Entry point:
- `TKBEN/start_on_windows.bat`

Behavior:
- installs portable Python, uv, and Node runtimes in `TKBEN/resources/runtimes`
- syncs backend dependencies with `uv sync`
- installs/builds frontend
- starts backend (uvicorn) and frontend preview server

Default local targets from examples:
- backend: `127.0.0.1:5000`
- frontend: `127.0.0.1:8000`

## 3. Cloud Mode (Docker)
Use:
```bat
docker compose --env-file TKBEN/settings/.env build
docker compose --env-file TKBEN/settings/.env up -d
```

Topology:
- `backend`: FastAPI/Uvicorn on container `8000`
- `frontend`: nginx on container `80`
- frontend `/api/*` reverse-proxies to `backend:8000`

## 4. Packaged Desktop Mode (Tauri)
Prepare desktop profile:
```bat
copy /Y TKBEN\settings\.env.local.tauri.example TKBEN\settings\.env
```

Provision portable runtimes if needed:
```bat
TKBEN\start_on_windows.bat
```

Build packaged desktop artifacts:
```bat
release\tauri\build_with_tauri.bat
```

Behavior:
- the Tauri app starts on `about:blank` and renders an in-window splash
- Rust resolves a packaged workspace containing `pyproject.toml` and `TKBEN/server/app.py`
- bundled `uv` and embedded Python prepare or reuse `.venv`
- FastAPI serves the packaged SPA from `TKBEN/client/dist`
- API routes remain available at their original paths and under `/api`
- exported user-facing artifacts land in `release/windows/installers` and `release/windows/portable`

## 5. Database Mode
- Embedded mode (`DB_EMBEDDED=true`):
  - SQLite at `TKBEN/resources/database.db`
- External mode (`DB_EMBEDDED=false`):
  - PostgreSQL connection via `DB_*` settings

## 6. Critical Env Keys
- `FASTAPI_HOST`, `FASTAPI_PORT`
- `UI_HOST`, `UI_PORT`
- `VITE_API_BASE_URL` (expected `/api`)
- `RELOAD`
- `DB_EMBEDDED`, `DB_ENGINE`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `DB_SSL`, `DB_SSL_CA`, `DB_CONNECT_TIMEOUT`, `DB_INSERT_BATCH_SIZE`
- `OPTIONAL_DEPENDENCIES`
- `HF_KEYS_ENCRYPTION_KEY`

Desktop packaging profile:
- `TKBEN/settings/.env.local.tauri.example`
- expected packaged defaults: loopback host, `/api`, `RELOAD=true`, `OPTIONAL_DEPENDENCIES=false`

## 7. Determinism
- Backend lockfile: `uv.lock` + `uv sync --frozen` in Docker image build
- Frontend lockfile: `TKBEN/client/package-lock.json` + `npm ci`
- Desktop build entrypoint: `release/tauri/build_with_tauri.bat`
