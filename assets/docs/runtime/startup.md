# Startup
Last updated: 2026-09-21

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
- installs pinned portable Python and Node.js on first use, and downloads uv from the current uv release when it is missing
- creates `settings/.env` from the versioned example when missing
- synchronizes Python dependencies and reuses the frontend dependency tree on application launch when the package manifests and portable Node.js version are unchanged; it performs `npm ci` when dependencies are missing or stale. The dependency maintenance option prompts for `Development` (including Ruff, BasedPyright, and pytest extras) or `Standard` (runtime dependencies only), then rebuilds the frontend and synchronizes the database to the latest Alembic head. Use menu option 3 to rebuild only the Angular frontend; it reuses valid frontend dependencies and runs `npm ci` when missing or stale without synchronizing Python dependencies. Application launch reuses an existing production build only when `dist/tkben-angular/browser/index.html` and the source-fingerprint stamp `dist/tkben-angular/.tkben-build.json` are present and match the Angular source tree, frontend manifests, and portable Node.js version; a missing or stale stamp triggers a rebuild before preview. All disposable runtime, package-manager, pytest, and development-tool caches are stored under the single canonical root `runtimes/cache`. Menu option 9 removes that root and legacy Python bytecode outside it; locked or admin-only files are reported and skipped so cleanup continues. Downloaded datasets and Hugging Face tokenizer artifacts remain persistent application data under `<TKBEN_DATA_DIR>/sources/...`; menu option 10 is the separate operation that permanently removes the embedded database, downloaded/uploaded sources, logs, and Hugging Face key material after confirmation while preserving application files, templates, and `.gitkeep` sentinels. If `DATABASE_EMBEDDED=false`, the external database is not modified. If the managed uv cache causes a sync failure, the launcher clears that cache on a best-effort basis and retries once. Before frontend synchronization and application startup, it stops listeners on the configured ports; a failed stop or remaining listener aborts the operation so an old process cannot make a new launch appear healthy. On Windows, maintenance and build commands invoke the portable `npm.cmd` from PowerShell (Windows dispatches `.cmd` through `cmd.exe`), while the preview process is launched through `cmd.exe` so repository paths containing spaces work reliably. Non-interactive Angular builds disable the progress renderer because the portable console renderer can terminate with a native access violation; the same production configuration is used. The launcher verifies the portable Node.js version and replaces an older runtime when required by the frontend dependency engines.
- starts FastAPI and the Angular preview server, captures preview output under `<TKBEN_LOG_DIR>` (defaulting to `app/resources/logs`), waits for preview readiness before opening the browser, and prints ports and process IDs; backend readiness continues under launcher and Angular supervision while browser auto-open remains best-effort
- loads `settings/.env` before backend configuration and database modules are imported; startup/environment values are validated before readiness, while the optional application-managed runtime override file is loaded after typed defaults and falls back to those defaults with a warning if malformed
- opens a dedicated terminal showing backend logs for interactive sessions; redirected `-Launch` runs capture backend stdout and stderr under `<TKBEN_LOG_DIR>` (defaulting to `app/resources/logs`) so health-check failures include actionable diagnostics
- keeps the maintenance menu usable when stdin/stdout are redirected by skipping cursor-only screen repaint operations while preserving normal interactive clearing and window-title behavior
- supports the direct `-Launch` path for automation while retaining the same dependency, health-check, and process-start logic as menu option 1

## Startup lifecycle

The launcher keeps process supervision separate from the browser-facing
readiness experience:

1. The launcher prepares dependencies, stops owned listeners, and starts the
   backend process.
2. The Angular preview starts immediately after the backend process is
   spawned; the launcher waits only for the preview root to respond before
   opening the browser.
3. Angular initially renders the tokenizer-focused startup screen and polls
   /api/health once per second through the preview proxy.
4. The shell, router outlet, and settings initialization are revealed only
   after the health response reports status ok.
5. The launcher continues its existing backend supervision after the browser
   opens. A backend process exit or 60-attempt timeout includes the redirected
   backend log tail, while the browser shows a retryable startup error.

Temporary connection failures during backend initialization are expected and
are kept out of the normal user-facing error state. The browser reports a
slow-start notice after 15 seconds and a retryable failure after the same
60-second readiness window used by the launcher. Refreshing during startup
restarts the readiness gate without changing the requested route; refreshing
after readiness loads the normal application directly.

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
Use `.\start_on_windows.ps1` for dependency installation, application updates, update checks, database initialization, tests, log removal, cache cleanup, user-data removal, process cleanup, and uninstall operations.

Use menu option 12, **Kill all application processes**, to stop TKBEN's
backend and frontend process trees. The equivalent direct command is:

```powershell
.\start_on_windows.ps1 -KillAll
```

The cleanup targets TKBEN launch commands rooted in this repository. It does
not stop unrelated Python or Node.js processes, even when they use the same
ports.

### Application updates

- `Update` updates source only from a non-detached, clean checkout of `main` with `git pull --ff-only origin main`. It never switches branches or modifies local changes; rerun the dependency or frontend setup options when the pulled changes require local rebuilds.
- `Check for Updates` reads the remote `origin/main` revision with `git ls-remote` and reports the status without fetching, pulling, or applying source changes.
- `Remove All Data` requires an affirmative response at a `[y/N]` confirmation prompt and removes mutable local data while preserving tracked application files and templates. It does not drop an external PostgreSQL database.
- Cache and data removal use deterministic deepest-first individual deletion. Preserved sentinels remain in place, while locked or inaccessible entries are reported and skipped so cleanup can continue.

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
- An empty database upgrades through the Alembic graph to its single current
  head. A non-empty database without an Alembic version row is rejected; so are
  unknown, partial, ahead-of-application, or multi-head states. Migration
  errors return a nonzero command exit and prevent startup health checks from
  succeeding.

## Test Mode
```bat
.\app\tests\run_tests.bat
```
