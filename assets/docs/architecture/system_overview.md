# System Overview
Last updated: 2026-07-11

## System Summary
TKBEN is a tokenizer benchmarking platform with:
- FastAPI backend (`app/server`)
- React + Vite frontend (`app/client`)
- Optional Tauri desktop packaging (`app/src-tauri`)
- Shared local resources and settings (`app/resources`, `settings`)

Backend APIs are mounted under `/api/*`. Frontend calls `/api` and relies on the Vite proxy in dev and preview modes.

## Repository Structure
Source-level structure, with generated folders omitted:

```text
.
├─ runtimes/
│  ├─ .venv/
│  └─ uv.lock
├─ assets/
│  ├─ docs/
│  └─ figures/
├─ start_on_windows.ps1
├─ settings/
│  ├─ .env
│  ├─ .env.example
│  └─ configurations.json
├─ app/
│  ├─ client/
│  │  ├─ package.json
│  │  ├─ vite.config.ts
│  │  ├─ src/
│  │  └─ dist/
│  ├─ src-tauri/
│  ├─ server/
│  │  ├─ pyproject.toml
│  │  ├─ app.py
│  │  ├─ api/
│  │  ├─ configurations/
│  │  ├─ domain/
│  │  ├─ services/
│  │  ├─ repositories/
│  │  └─ common/
│  ├─ scripts/
│  ├─ tests/
│  └─ resources/
├─ release/
│  ├─ tauri/
│  └─ windows/
```

## Application Entry Points
- Backend app factory/module:
  - `server.app:create_app` constructs the FastAPI app and registers API and frontend routes.
  - `server.app:app` is the canonical ASGI entry point.
- Frontend entry:
  - `app/client/src/main.tsx`
- Frontend routing root:
  - `app/client/src/App.tsx`
- Desktop runtime entry:
  - `app/src-tauri/src/main.rs`
- Windows launcher:
  - `start_on_windows.ps1` is the single user-facing root entry point for the combined launch and maintenance menu.

## Runtime Interaction Topology
- Local webapp mode:
  - Browser -> Vite preview (`UI_HOST:UI_PORT`) -> proxied `/api` -> FastAPI (`FASTAPI_HOST:FASTAPI_PORT`)
- Desktop mode:
  - Tauri webview boots the bundled Python/backend from immutable resources and loads its dynamically allocated loopback URL.
  - Mutable configuration, data, logs, and caches are redirected to `%LOCALAPPDATA%\TKBEN`.
