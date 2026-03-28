# Packaging and Runtime Modes (TKBEN)

## 1. Runtime Profiles
Active runtime profile:
- `TKBEN/settings/.env`

Profile templates:
- `TKBEN/settings/.env.local.example`
- `TKBEN/settings/.env.local.tauri.example`

Non-env defaults:
- `TKBEN/settings/configurations.json`

## 2. Local Webapp Mode (Default)
Entrypoint:
- `TKBEN/start_on_windows.bat`

Behavior:
- provisions portable Python, uv, and Node in `runtimes/`
- syncs backend dependencies with `uv sync`
- installs/builds frontend assets as needed
- starts backend (`uvicorn`) and frontend preview server

Local profile defaults (`.env.local.example`):
- backend: `127.0.0.1:5000`
- frontend: `127.0.0.1:8000`

Launcher hardcoded fallback if `.env` is missing:
- backend: `127.0.0.1:5000`
- frontend: `127.0.0.1:8001`

## 3. Packaged Desktop Mode (Tauri)
Prepare desktop profile:
```bat
copy /Y TKBEN\settings\.env.local.tauri.example TKBEN\settings\.env
```

Provision runtimes if needed:
```bat
TKBEN\start_on_windows.bat
```

Build desktop artifacts:
```bat
release\tauri\build_with_tauri.bat
```

Rust prerequisite:
- install Rust and configure default toolchain:
  - `rustup toolchain install stable`
  - `rustup default stable`
  - `rustup target add x86_64-pc-windows-msvc`

Packaged behavior:
- Tauri app starts at `about:blank`, then loads splash/UI in-window.
- Backend serves packaged SPA from `TKBEN/client/dist`.
- API routes remain available both as original paths and `/api/*` aliases.
- Exported artifacts land in `release/windows/installers` and `release/windows/portable`.

## 4. Database Modes
- Embedded mode (`DB_EMBEDDED=true`):
  - SQLite at `TKBEN/resources/database.db`
- External mode (`DB_EMBEDDED=false`):
  - PostgreSQL via `DB_*` environment variables

## 5. Critical Environment Keys
- `FASTAPI_HOST`, `FASTAPI_PORT`
- `UI_HOST`, `UI_PORT`
- `VITE_API_BASE_URL` (expected `/api`)
- `RELOAD`
- `DB_EMBEDDED`, `DB_ENGINE`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `DB_SSL`, `DB_SSL_CA`, `DB_CONNECT_TIMEOUT`, `DB_INSERT_BATCH_SIZE`
- `OPTIONAL_DEPENDENCIES`
- `ALLOW_KEY_REVEAL`
- `HF_KEYS_ENCRYPTION_KEY`

Tauri profile baseline (`.env.local.tauri.example`):
- loopback hosts
- `VITE_API_BASE_URL=/api`
- `RELOAD=false`
- `OPTIONAL_DEPENDENCIES=false`

## 6. Reproducibility Anchors
- Backend dependency lock: `runtimes/uv.lock`
  - copied to root `uv.lock` before `uv sync`
  - copied back to `runtimes/uv.lock` on successful sync
- Frontend lock: `TKBEN/client/package-lock.json` with `npm ci` when available
- Desktop build entrypoint: `release/tauri/build_with_tauri.bat`
