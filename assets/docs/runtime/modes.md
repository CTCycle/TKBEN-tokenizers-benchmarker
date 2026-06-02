# Runtime Modes
Last updated: 2026-06-02

## Supported Modes
### 1. Local webapp mode
- Backend: FastAPI (`TKBEN.server.app:app`)
- Frontend: Vite preview build (`TKBEN/client/dist`)
- Primary launcher: `TKBEN/start_on_windows.bat`

### 2. Desktop packaged mode
- Windows packaged desktop app using `TKBEN/client/src-tauri`.
- Bundles backend, frontend, resources, and runtimes as Tauri resources.
- Build helper: `release/tauri/build_with_tauri.bat`

### 3. Test runtime mode
- Uses the existing `runtimes/.venv` and local backend and frontend servers for pytest suites.
- Entry script: `app/tests/run_tests.bat`

### 4. Containerized mode
- Not implemented in the current repository state.

## Interoperability
- Frontend and backend communicate through HTTP JSON APIs under `/api/*`.
- In local webapp mode, Vite proxy rewrites `/api/*` to the backend root.
- In desktop mode, Tauri manages backend process lifecycle and points the UI to the local backend URL.
- Shared resources:
  - database (`resources/database.db`)
  - downloaded datasets and tokenizers (`resources/sources/*`)
  - templates and logs (`resources/templates`, `resources/logs`)

## Limitations and Constraints
- Desktop local backend bootstrap in Tauri is Windows-only in the current Rust implementation.
- Long-running operations are asynchronous jobs and require polling via `/api/jobs/{job_id}`.
- Large download and processing operations depend on local network and disk throughput.
- `ALLOW_KEY_REVEAL` controls whether Hugging Face keys can be revealed via API.
