# Runtime Modes
Last updated: 2026-07-11

## Supported Modes
### 1. Local webapp mode
- Backend: FastAPI (`TKBEN.server.app:app`)
- Frontend: Vite preview build (`TKBEN/client/dist`)
- Canonical and sole root launcher: `TKBEN/start_on_windows.ps1`.
- Uses a Vite preview build and FastAPI with portable Windows runtimes.

### 2. Test runtime mode
- Uses the local backend and frontend test environments managed by `app/tests/run_tests.bat`.
- Entry script: `app/tests/run_tests.bat`

### 3. Containerized mode
- Not implemented in the current repository state.

## Interoperability
- Frontend and backend communicate through HTTP JSON APIs under `/api/*`.
- In local webapp mode, Vite proxy rewrites `/api/*` to the backend root.
- The launcher starts the backend and frontend as separate local processes and points the browser to the configured UI URL.

## Limitations and Constraints
- The automatic portable-runtime bootstrap is Windows-only.
- Long-running operations are asynchronous jobs and require polling via `/api/jobs/{job_id}`.
- Large download and processing operations depend on local network and disk throughput.
- `ALLOW_KEY_REVEAL` controls whether Hugging Face keys can be revealed via API.
