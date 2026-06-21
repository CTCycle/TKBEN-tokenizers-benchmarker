# Startup
Last updated: 2026-06-21

## Local Webapp Mode
Windows recommended:

```bat
.\start_on_windows.bat
```

What it does:
- installs or uses portable Python, uv, and Node.js under `runtimes`
- syncs Python dependencies with `uv sync` using `runtimes/uv.lock`
- installs frontend dependencies and builds the frontend
- starts backend and frontend

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
copy /Y settings\.env.example settings\.env
.\release\tauri\build_with_tauri.bat
```

## Test Mode
```bat
.\app\tests\run_tests.bat
```
