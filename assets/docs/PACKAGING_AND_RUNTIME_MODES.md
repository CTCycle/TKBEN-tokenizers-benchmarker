# Packaging and Runtime Modes (TKBEN)
Last updated: 2026-04-08


## 1. Runtime Profiles
Active runtime profile:
- `TKBEN/settings/.env`

Single template:
- `TKBEN/settings/.env.example`

Structured backend configuration:
- `TKBEN/settings/configurations.json`

## 2. Local Webapp Mode (Default)
Entrypoint:
- `TKBEN/start_on_windows.bat`

Behavior:
- provisions portable Python, uv, and Node in `runtimes/`
- syncs backend dependencies with `uv sync`
- installs/builds frontend assets as needed
- starts backend (`uvicorn`) and frontend preview server

Default runtime values (`.env.example`):
- backend: `127.0.0.1:5000`
- frontend: `127.0.0.1:8000`

Launcher hardcoded fallback if `.env` is missing:
- backend: `127.0.0.1:5000`
- frontend: `127.0.0.1:8000`

## 3. Packaged Desktop Mode (Tauri)
Prepare runtime env file:
```bat
copy /Y TKBEN\settings\.env.example TKBEN\settings\.env
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
- API routes are exposed under `/api/*` only.
- Exported artifacts land in `release/windows/installers` and `release/windows/portable`.

## 4. Database Modes
- Embedded mode (`database.embedded_database=true` in `configurations.json`):
  - SQLite at `TKBEN/resources/database.db`
- External mode (`database.embedded_database=false`):
  - PostgreSQL via `database.*` fields in `configurations.json`

## 5. Critical Environment Keys
- `FASTAPI_HOST`, `FASTAPI_PORT`
- `UI_HOST`, `UI_PORT`
- `VITE_API_BASE_URL` (expected `/api`)
- `RELOAD`
- `KERAS_BACKEND`, `MPLBACKEND`
- `OPTIONAL_DEPENDENCIES`
- `ALLOW_KEY_REVEAL`
- `HF_KEYS_ENCRYPTION_KEY`

## 6. Reproducibility Anchors
- Backend dependency lock: `runtimes/uv.lock`
  - copied to root `uv.lock` before `uv sync`
  - copied back to `runtimes/uv.lock` on successful sync
- Frontend lock: `TKBEN/client/package-lock.json` with `npm ci` when available
- Desktop build entrypoint: `release/tauri/build_with_tauri.bat`
