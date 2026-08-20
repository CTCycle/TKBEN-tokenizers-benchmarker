# Startup
Last updated: 2026-08-20

## Local Webapp Mode
Windows recommended:

```powershell
.\start_on_windows.ps1
```

For redirected or automated validation, launch the application directly while
still using the same launcher:

```powershell
.\start_on_windows.ps1 -Launch
```

What it does:
- opens the single combined launch-and-maintenance menu
- installs pinned portable Python, uv, and Node.js on first use
- creates `settings/.env` from the versioned example when missing
- synchronizes Python dependencies and reuses the frontend dependency tree on application launch when the package manifests and portable Node.js version are unchanged; it performs a clean frontend install when dependencies are missing or stale. The dependency maintenance option prompts for `Development` (including Ruff, BasedPyright, and pytest extras) or `Standard` (runtime dependencies only), then rebuilds the frontend and synchronizes the database to the latest Alembic head. Use menu option 3 to rebuild only the Angular frontend; it reuses valid frontend dependencies and installs them when missing or stale without synchronizing Python dependencies. Application launch reuses the existing production output and does not rebuild it; run menu option 2 after dependency changes or option 3 after frontend changes. Development tool caches and temporary test artifacts are stored under `assets/cache`; if the managed uv cache causes a sync failure, the launcher clears that cache and retries once. Before frontend synchronization, it stops any listener on the configured UI port so a prior Angular preview cannot keep `node_modules` binaries locked. On Windows, portable `npm.cmd` calls are routed through `cmd.exe` so repository paths containing spaces work reliably. The launcher verifies the portable Node.js version and replaces an older runtime when required by the frontend dependency engines.
- starts FastAPI and the Angular preview server, captures preview output under `app/resources/logs`, waits for health checks, and prints ports and process IDs; browser auto-open is best-effort and reports the URL when Windows blocks it
- optionally shows backend logs in a dedicated terminal when `BACKEND_LOGS_VISIBLE=true` (the default when absent; the only accepted values are `true` and `false`)
- keeps the maintenance menu usable when stdin/stdout are redirected by skipping cursor-only screen repaint operations while preserving normal interactive clearing and window-title behavior
- supports the direct `-Launch` path for automation while retaining the same dependency, health-check, and process-start logic as menu option 1

## Manual Local Mode
Cross-platform manual startup:

```bash
cd app/server
uv sync
uv run python -m uvicorn server.app:app --app-dir .. --host 127.0.0.1 --port 5000
cd ../client
npm ci
npm run preview -- --host 127.0.0.1 --port 8000 --strictPort
```

## Maintenance Menu
Use `.\start_on_windows.ps1` for dependency installation/update, database initialization, tests, log removal, cache cleanup, and uninstall operations.

### Database initialization

- Every application startup checks the Alembic version and automatically
  applies pending revisions before FastAPI exposes application state.
- SQLite creates `<TKBEN_DATA_DIR>/database.db` when missing and serializes
  migration writers. PostgreSQL uses the `DATABASE_*` values in
  `settings/.env`; a missing target is created only when the configured role
  has `CREATEDB` permission. PostgreSQL migrations are serialized with an
  advisory lock.
- Launcher menu option 2 (dependency installation/update) and option 4
  (database initialization) invoke the same initializer. Repeating either
  operation is safe and preserves existing data.
- A supported unversioned TKBEN database is adopted only after a complete
  schema signature check. Unknown, partial, conflicting, ahead-of-application,
  or multi-head databases fail without being stamped or downgraded. Migration
  errors return a nonzero command exit and prevent startup health checks from
  succeeding.

## Test Mode
```bat
.\app\tests\run_tests.bat
```
