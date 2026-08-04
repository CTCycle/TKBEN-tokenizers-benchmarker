# Startup
Last updated: 2026-08-04

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
- synchronizes Python dependencies and reuses the frontend dependency tree on application launch when the package manifests and portable Node.js version are unchanged; it performs a clean frontend install when dependencies are missing or stale. The frontend is rebuilt when `ALWAYS_REBUILD=true`; the launcher uses `true` when the setting is absent, while the generated `settings/.env` copied from the template sets it to `false`. If the managed uv cache causes a sync failure, the launcher clears that cache and retries once. Before frontend synchronization, it stops any listener on the configured UI port so a prior Vite preview cannot keep `node_modules` binaries such as esbuild locked. On Windows, portable `npm.cmd` calls are routed through `cmd.exe` so repository paths containing spaces work reliably. The launcher verifies the portable Node.js version and replaces an older runtime when required by the frontend dependency engines.
- starts FastAPI and the Vite preview server, waits for health checks, opens the browser, and prints ports and process IDs
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

- SQLite is initialized automatically only when `app/resources/database.db` is
  missing. Existing SQLite files are not schema-validated, recreated, reset, or
  reseeded during startup.
- PostgreSQL is never created or initialized by normal startup. Set
  `DATABASE_EMBEDDED=false` and the PostgreSQL connection fields in
  `settings/.env`, then choose menu option 3 (`Initialize database`). The
  command creates the configured database and schema and seeds the persisted
  metric catalog.
- After PostgreSQL initialization, later launches perform only a connection
  readiness check. Invalid or unavailable PostgreSQL settings stop the launch
  or initialization command with a nonzero failure.

## Test Mode
```bat
.\app\tests\run_tests.bat
```
