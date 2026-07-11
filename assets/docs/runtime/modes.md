# Runtime Modes
Last updated: 2026-07-11

## Supported Modes
### 1. Local webapp mode
- Backend: FastAPI (`TKBEN.server.app:app`)
- Frontend: Vite preview build (`TKBEN/client/dist`)
- Canonical and sole root launcher: `TKBEN/start_on_windows.ps1`.
- Uses Vite development mode and FastAPI without rebuilding Tauri.

### 2. Desktop packaged mode
- Windows packaged desktop app using `TKBEN/client/src-tauri`.
- Windows 10/11 x64 only.
- Bundles frontend, backend, embedded Python, and locked production dependencies as Tauri resources.
- Performs no dependency installation on the target system.
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
- Launcher-managed and packaged modes store mutable content under `%LOCALAPPDATA%\TKBEN`.

## Limitations and Constraints
- Desktop local backend bootstrap in Tauri is Windows-only in the current Rust implementation.
- Long-running operations are asynchronous jobs and require polling via `/api/jobs/{job_id}`.
- Large download and processing operations depend on local network and disk throughput.
- `ALLOW_KEY_REVEAL` controls whether Hugging Face keys can be revealed via API.
