# Deployment
Last updated: 2026-10-21

## Dependency Prerequisites

- The Windows launcher bootstraps pinned Python 3.14.7, Node.js 22.23.1, and
  uv 0.12.17 locally, replacing managed runtimes when their versions differ.
- Downloaded portable runtimes live under the ignored `runtimes/` directory.
- Windows x64 is the supported release target. Manual execution on other
  operating systems is unsupported and outside release validation.
- The Angular production build is static output at
  `app/client/dist/tkben-angular/browser`. The launcher builds it when missing
  or stale, then FastAPI serves it alongside `/api` on one port. `npm run
  preview` is not the production server.

## Local Distribution Strategy

- The repository plus `start_on_windows.ps1` is the supported Windows
  operational path. It remains a source-only folder distribution.
- The launcher records backend dependency state from `pyproject.toml` and
  `uv.lock`, uses `uv sync --locked` only when that state is missing or stale,
  and repairs frontend dependencies with `npm ci` independently. Application
  launch rebuilds Angular only when the production entry or deterministic
  build stamp is missing, malformed, or stale. A backend repair alone does
  not force a frontend build. Explicit dependency installation and **Rebuild
  frontend** still build intentionally.
- Before setup and immediately before process start, the launcher checks only
  the configured backend port. An occupied listener is reported by PID and
  port; interactive launch asks before terminating the approved listener PID.
  Redirected launch, declined approval, or failed release aborts. `UI_PORT` is
  reserved for manual development and does not block production launch.
- The launcher supervises the FastAPI process it starts, opens the same-origin
  application after `/api/health` and `/` respond, and handles Ctrl+C with a
  graceful shutdown followed by bounded cleanup of that process tree.
- The default launcher binds locally with `FASTAPI_HOST=127.0.0.1`.
- Network-hosted deployments require an external authentication boundary
  before exposing key management or destructive API routes.

## Packaging and Constraints

- The repository has no Tauri or other desktop packaging configuration. No
  desktop package is introduced; the Angular build is served by FastAPI.
- The repository does not currently include an active Docker runtime
  configuration in the root.
- Automatic Python and Node.js downloads target Windows x64.
- Manual Linux startup, browser routes, and backend restart reconciliation
  were exercised on Ubuntu 26.04 on 2026-09-22 as diagnostic evidence only.
  Linux and macOS are not supported release targets; no additional platform
  validation is planned or required for the Windows release.
- Releases contain source files only, with no installer, executable, or other
  binary artifact. Extract or clone the application folder and run
  `start_on_windows.ps1` from its root.
