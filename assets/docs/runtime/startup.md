# Startup
Last updated: 2026-10-25

## Production Local Webapp

The supported Windows release path uses the root launcher:

```powershell
.\start_on_windows.ps1
```

Choose **Launch application**. For direct or automated startup, run:

```powershell
.\start_on_windows.ps1 -Launch
```

Both launch paths use one FastAPI process. It serves the production Angular
build from `app/client/dist/tkben-angular/browser` and keeps API and frontend
requests on `FASTAPI_HOST:FASTAPI_PORT`. Production does not start Angular's
development or preview server. When a build is absent, `/` redirects to `/docs`
until a build is available.

The launcher:

- installs pinned portable Python 3.14.7, Node.js 22.23.1, and uv 0.12.17 on
  first use and creates `settings/.env` from the versioned example when needed
- during production launch, reuses valid backend dependencies, frontend
  dependencies, and the stamped production build; repairs or rebuilds only
  missing, stale, or malformed inputs. The **Install dependencies** maintenance
  action and explicit **Rebuild frontend** action still build when selected
- checks `FASTAPI_PORT` before setup and again before starting FastAPI. An
  occupied backend port is reported by listener PID and process name; an
  interactive launch asks before terminating those exact listener processes.
  A declined prompt, redirected `-Launch`, failed termination, or remaining
  listener aborts without starting the app. `UI_PORT` is not checked during
  production launch because no UI server uses it
- starts FastAPI, waits for `/api/health` and `/`, then opens the backend URL
  in the browser. A wildcard bind address maps to a loopback URL for local use
- keeps the backend attached to the launcher console for visible logs. The app
  also writes its normal `TKBEN_*.log` under `<TKBEN_LOG_DIR>` (default
  `data/logs`); redirected launches save stdout and stderr there as well
- remains open as the lifecycle owner after readiness. Press Ctrl+C in that
  window to request FastAPI's normal shutdown. It waits up to 10 seconds, then
  stops only the FastAPI process tree it started if necessary. Startup errors,
  unexpected process exits, and cleanup failures include diagnostics. If the
  launcher exits unexpectedly, its cleanup also targets only that owned
  process tree
- keeps the portable Node.js version compatible with the frontend dependency
  engines and disables Angular's build progress renderer for non-interactive
  builds, where the portable console renderer can fail

`-Launch` stays in the foreground until Ctrl+C or FastAPI exits. The menu's
**Launch application** option returns to the maintenance menu after the app is
stopped. `-KillAll` and menu option 12 remain explicit cleanup actions for
TKBEN backend process trees rooted in this repository; they do not stop
unrelated Python or Node processes.

FastAPI loads bootstrap values from `settings/.env` and then
`database-config.json` before initializing the selected database. If the
database cannot initialize, the backend still serves the SPA and exposes
`/api/health` and `/api/database` so Settings → Database can repair the
connection. Other API requests return HTTP 503 until the database is available.
Database switching applies in the current process; listener, data/log path,
CORS, and key-material changes require a restart.

Jobs marked `pending` or `running` when the backend restarts become `failed`
with an application-restart error and remain available at their original job
IDs. Interrupted dataset analyses, downloads, and benchmarks are not resumed.

Every application start checks the Alembic schema and applies pending
migrations before database-backed state is exposed. SQLite serializes migration
writers; PostgreSQL uses an advisory lock. A failed database initialization
keeps FastAPI available in repair mode. Menu options 2 and 4 use the same
initializer.

## Manual Development Workflow

Development is a two-terminal workflow. The frontend development server is
used here only; its `/api` proxy forwards to FastAPI. The proxy defaults to
`127.0.0.1:5000` and reads `FASTAPI_HOST` and `FASTAPI_PORT` from the frontend
process environment when set.

Terminal 1, from the repository root:

```powershell
Set-Location app/server
uv sync
uv run python -m uvicorn server.app:app --app-dir .. --reload --host 127.0.0.1 --port 5000
```

Terminal 2, from the repository root:

```powershell
Set-Location app/client
npm ci
npm run dev -- --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Stop each development process with Ctrl+C in its
own terminal. Set `FASTAPI_HOST` and `FASTAPI_PORT` in Terminal 2 if the API
uses a different address. `UI_HOST` and `UI_PORT` are development-server
settings; they do not affect the production listener.

The `npm run preview` helper remains available to the test harness. It is not
used by the production launcher.

## Maintenance Menu

Use `.\start_on_windows.ps1` for dependency installation, application updates,
update checks, database initialization, tests, log removal, cache cleanup,
user-data removal, process cleanup, and uninstall operations.

Use menu option 12, **Kill all application processes**, or run:

```powershell
.\start_on_windows.ps1 -KillAll
```

This explicit cleanup action is separate from launch. Launch port checks never
infer ownership from a port; they ask before releasing the exact listener PID
on the configured backend port. Process cleanup matches TKBEN's backend
command rooted in this checkout and does not stop unrelated services.

### Application updates

- `Update` updates source only from a non-detached, clean checkout of `main`
  with `git pull --ff-only origin main`. It never switches branches or modifies
  local changes; rerun dependency or frontend setup options when pulled changes
  require local rebuilds.
- `Check for Updates` reads the remote `origin/main` revision with `git
  ls-remote` and reports the status without fetching, pulling, or applying
  source changes.
- `Remove All Data` requires an affirmative response at a `[y/N]` confirmation
  prompt and removes mutable local data while preserving tracked application
  files and templates. It does not drop an external PostgreSQL database.

## Test Mode

Run the repository's test workflows through:

```powershell
.\app\tests\run_tests.bat
```

The runner may use `npm run preview` for frontend integration checks. This is
separate from the production launch. Its live mode now performs a locked
frontend install and production build of the current checkout, starts only
missing services, and records the exact spawned PIDs, executable paths, and
creation times. Cleanup stops only those verified test-owned processes.
The preview server uses strict port binding instead of selecting an
unexpected fallback port.

The release-specific Windows production launcher still requires a manual
standard-console startup and Ctrl+C shutdown smoke test. The hosted Windows
CI checks script parsing and owned preview lifecycle, but does not substitute
for the interactive production launcher gate.
