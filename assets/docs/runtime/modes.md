# Runtime Modes
Last updated: 2026-10-21

## Supported Modes

### 1. Local production webapp

- FastAPI (`server.app:app` from `app/`) serves both the built Angular SPA and
  API routes under `/api/*` from `FASTAPI_HOST:FASTAPI_PORT`.
- The canonical Windows launcher is `start_on_windows.ps1`. It reuses or
  rebuilds the production Angular output at
  `app/client/dist/tkben-angular/browser`, starts one FastAPI process, waits
  for API and SPA readiness, opens the same-origin URL, and supervises that
  process until Ctrl+C or process exit.
- `-Launch` runs the same lifecycle without the maintenance menu. Ctrl+C
  requests graceful shutdown; launcher cleanup is limited to its owned
  FastAPI process tree. `-KillAll` is a separate explicit cleanup action.
- `UI_HOST` and `UI_PORT` configure development only and do not reserve a
  production listener.

### 2. Manual development

- Run Uvicorn with reload in one terminal and Angular CLI's `npm run dev` in a
  second terminal.
- Angular dev server proxies `/api/*` to the FastAPI host and port configured
  in its process environment (`127.0.0.1:5000` by default).
- `UI_HOST` and `UI_PORT` can be passed as Angular CLI `--host` and `--port`
  options. The production launcher does not use the Angular development or
  preview server.

### 3. Test runtime mode

- Uses the local backend and frontend test environments managed by
  `app/tests/run_tests.bat`. The `npm run preview` helper remains available to
  the test harness.

### 4. Containerized mode

- Not implemented in the current repository state.

## Interoperability

- Frontend and backend communicate through relative `/api/*` requests.
- In production, the browser sends SPA and API requests to FastAPI on the same
  origin. FastAPI returns built files, `index.html` for client-side routes,
  and 404 for unknown API paths or missing file-like assets.
- In development, Angular CLI serves the frontend and proxies `/api/*` to
  FastAPI.
- The repository is a source-only folder distribution. It has no Tauri or
  other desktop package.

## Limitations and Constraints

- The automatic portable-runtime bootstrap is Windows-only.
- Long-running operations are asynchronous jobs and require polling via
  `/api/jobs/{job_id}`.
- Large download and processing operations depend on local network and disk
  throughput.
- `ALLOW_KEY_REVEAL` controls whether Hugging Face keys can be revealed via
  API.
