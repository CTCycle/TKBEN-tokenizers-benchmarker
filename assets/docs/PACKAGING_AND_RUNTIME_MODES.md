# TKBEN Packaging and Runtime Modes

## 1. Strategy

TKBEN uses one active runtime file: `TKBEN/settings/.env`.

- Local mode: run directly on host (default, no Docker required).
- Cloud mode: run with Docker (`backend` + `frontend`).
- Mode switching: replace values in `TKBEN/settings/.env` only.
- Runtime mode changes do not require business-logic branching.

## 2. Runtime Profiles

- `TKBEN/settings/.env.local.example`: local defaults (loopback hosts, embedded DB).
- `TKBEN/settings/.env.cloud.example`: cloud defaults (bind hosts, external DB).
- `TKBEN/settings/.env`: active profile used by launcher, tests, and Docker runtime env loading.
- `TKBEN/settings/configurations.json`: non-runtime defaults; runtime DB mode can be overridden by `DB_EMBEDDED`.

## 3. Required Runtime Keys

| Key | Purpose | Local Example | Cloud Example |
|---|---|---|---|
| `FASTAPI_HOST`, `FASTAPI_PORT` | Backend bind/runtime port | `127.0.0.1`, `5000` | `0.0.0.0`, `5000` |
| `UI_HOST`, `UI_PORT` | Frontend host/port (local) and host-published UI port (cloud) | `127.0.0.1`, `8000` | `0.0.0.0`, `8000` |
| `VITE_API_BASE_URL` | Frontend API base path | `/api` | `/api` |
| `RELOAD` | Backend auto-reload toggle | `false` | `false` |
| `DB_EMBEDDED` | Runtime DB mode switch | `true` | `false` |
| `DB_ENGINE`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | External DB connection fields | optional in local | required in cloud |
| `DB_SSL`, `DB_SSL_CA` | External DB TLS behavior | `false`, empty | typically `true`, CA path |
| `DB_CONNECT_TIMEOUT`, `DB_INSERT_BATCH_SIZE` | DB runtime tuning | `30`, `1000` | `30`, `1000` |
| `OPTIONAL_DEPENDENCIES` | Optional local test extras install toggle | `true` | `false` |
| `MPLBACKEND`, `KERAS_BACKEND` | Runtime library backend controls | `Agg`, `tensorflow` | `Agg`, `tensorflow` |

## 4. Local Mode (Default)

1. Activate local profile:
   - `copy /Y TKBEN\settings\.env.local.example TKBEN\settings\.env`
2. Start app:
   - `.\TKBEN\start_on_windows.bat`
3. Run tests (optional):
   - `tests\run_tests.bat`

Local mode does not require Docker.

## 5. Cloud Mode (Docker)

1. Activate cloud profile:
   - `copy /Y TKBEN\settings\.env.cloud.example TKBEN\settings\.env`
2. Build images:
   - `docker compose --env-file TKBEN/settings/.env build --no-cache`
3. Start stack:
   - `docker compose --env-file TKBEN/settings/.env up -d`
4. Stop stack:
   - `docker compose --env-file TKBEN/settings/.env down`

Cloud topology:
- `frontend` serves the built SPA via Nginx.
- `/api` on the frontend origin is reverse-proxied to `backend:8000`.
- `backend` runs FastAPI/Uvicorn on container port `8000`.

## 6. Build Determinism

- Backend dependency resolution is lockfile-backed with `uv.lock`.
- Backend image install path uses `uv sync --frozen`.
- Frontend dependency resolution is lockfile-backed with committed `TKBEN/client/package-lock.json`.
- Frontend image install path uses `npm ci`.
- Docker images are pinned to explicit base image tags in `docker/backend.Dockerfile` and `docker/frontend.Dockerfile`.
