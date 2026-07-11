# Startup
Last updated: 2026-07-11

## Local Webapp Mode
Windows recommended:

```powershell
.\start_on_windows.ps1
```

What it does:
- opens the single combined launch-and-maintenance menu
- installs pinned portable Python, uv, and Node.js on first use
- restores committed `app/server/uv.lock` and `app/client/package-lock.json`
- seeds `%LOCALAPPDATA%\TKBEN` configuration/data directories
- starts FastAPI and Vite development mode with owned PID tracking

## Manual Local Mode
Cross-platform manual startup:

```bash
uv sync
uv run python -m uvicorn server.app:app --app-dir app --host 127.0.0.1 --port 5000
cd app/client
npm ci
npm run preview -- --host 127.0.0.1 --port 8000 --strictPort
```

## Desktop Packaging Mode
Windows packaging flow:

```bat
.\release\tauri\build_with_tauri.bat
```

Use `.\start_on_windows.ps1` for the interactive menu. Non-interactive callers pass `-Action <Action>` through that same entry point.

## Test Mode
```bat
.\app\tests\run_tests.bat
```
