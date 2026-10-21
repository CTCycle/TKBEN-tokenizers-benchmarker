# FastAPI Static SPA Hosting QA

Date: 2026-10-21

Status: PARTIAL. Backend serving and same-origin browser behavior passed. The
normal Windows launcher did not pass readiness in this host's nested
PowerShell environment.

## Passed checks

- `app/tests/unit/server/test_app_startup.py`: 7 passed.
- Ruff passed for the changed Python files. The PowerShell parser accepted
  `start_on_windows.ps1`; `git diff --check` passed.
- Direct Uvicorn startup against isolated SQLite data served `/`, `/settings`,
  the Angular JavaScript bundle, `/api/health`, `/api/settings`, and dataset
  API requests. The root and client route returned the built `index.html`;
  unknown `/api` paths and a missing `.js` asset returned 404.
- During database repair mode, a valid database-backed API route returned 503
  while an unknown API path, the SPA, and a missing asset retained their
  expected 404/200 responses.
- The in-app browser rendered the built Angular application. Its network log
  showed the hashed JavaScript/CSS chunks and same-origin health, settings, and
  dataset API requests returning 200.
- A synthetic PowerShell parent/child tree was stopped by the revised
  `Stop-ProcessTree` helper. Both owned PIDs exited and an unrelated listener
  on port 8000 remained active until deliberate test cleanup.
- With that port-8000 sentinel active, the launcher proceeded past port checks
  and attempted FastAPI startup; production does not depend on `UI_PORT`.

## Launcher limitation

Two `start_on_windows.ps1 -Launch` attempts did not reach `/api/health` within
the 60-attempt readiness window. A temporary `sitecustomize.py` diagnostic
captured Python 3.14 blocked in `socket._fallback_socketpair()` while
initializing the Windows Proactor event loop. The temporary hook was removed.
Direct Uvicorn startup with the same isolated data and production build worked,
so this run does not establish whether the stall is specific to the nested
PowerShell host or the launcher path. Standard Windows-console launcher
readiness and its normal Ctrl+C shutdown remain unverified.

The original ignored `settings/.env` was restored byte-for-byte (SHA-256
`B31B2C3C79F9F69A49F7790B3C4013C4E8012664F128A2296B25DBE379D4880F`). Scratch
data and the sentinel were removed; ports 64076, 64077, and 8000 were clear.
