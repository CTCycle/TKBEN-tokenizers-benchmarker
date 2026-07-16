# Startup
Last updated: 2026-07-12

## Local Webapp Mode
Windows recommended:

```powershell
.\start_on_windows.ps1
```

What it does:
- opens the single combined launch-and-maintenance menu
- installs pinned portable Python, uv, and Node.js on first use
- creates `settings/.env` from the versioned example when missing
- synchronizes Python and frontend dependencies and rebuilds the frontend when `always_rebuild=true` (the default); set it to `false` to skip the frontend build at application start
- starts FastAPI and the Vite preview server, waits for health checks, opens the browser, and prints ports and process IDs
- optionally shows backend logs in a dedicated terminal when `BACKEND_LOGS_VISIBLE=true` (the default when absent; the only accepted values are `true` and `false`)

## Manual Local Mode
Cross-platform manual startup:

```bash
uv sync
uv run python -m uvicorn server.app:app --app-dir app --host 127.0.0.1 --port 5000
cd app/client
npm ci
npm run preview -- --host 127.0.0.1 --port 8000 --strictPort
```

## Maintenance Menu
Use `.\start_on_windows.ps1` for dependency installation/update, database initialization, tests, log removal, cache cleanup, and uninstall operations.

## Test Mode
```bat
.\app\tests\run_tests.bat
```
