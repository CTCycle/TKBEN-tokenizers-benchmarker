# Deployment
Last updated: 2026-09-24

## Dependency Prerequisites
From project and runtime scripts:
- Windows launcher mode bootstraps pinned Python 3.14.7, Node.js 22.23.1, and uv 0.12.17 locally; it replaces managed runtimes when their versions differ from these pins.
- The launcher downloads portable runtimes into the ignored `runtimes/` directory when they are missing.
- Windows x64 is the supported release target. Manual execution on other operating systems is unsupported and outside release validation.
- On a fresh manual checkout, run `npm run build` after `npm ci` before starting `npm run preview`; the preview serves the production output under `dist/tkben-angular/browser`.

## Local Distribution Strategy
- The repository plus `start_on_windows.ps1` is the supported Windows operational path.
- CI runner operating systems and diagnostic manual runs on other platforms do not extend the Windows release support contract.
- The launcher records backend dependency state from `pyproject.toml` and
  `uv.lock`, uses `uv sync --locked` only when that state is missing or stale,
  and repairs frontend dependencies with `npm ci` independently. Application
  launch rebuilds Angular only when the production entry or deterministic build
  stamp is missing, malformed, or stale; a backend repair alone does not force a
  frontend build. Explicit dependency installation and **Rebuild frontend**
  still build intentionally.
- Before setup and immediately before process start, the launcher validates the
  configured ports. If a listener exists, interactive launch displays the unique
  listener PIDs and occupied ports and asks once before terminating approved
  PIDs; redirected launch and declined or failed release leave processes intact
  and abort. `-KillAll` is the explicit TKBEN process-tree cleanup action.
- The default launcher binds locally with `FASTAPI_HOST=127.0.0.1`.
- Network-hosted deployments require an external authentication boundary before exposing key management or destructive API routes.

## Constraints
- The repository does not currently include an active Docker runtime configuration in the root.
- Automatic Python and Node.js downloads target Windows x64.
- Manual Linux startup, browser routes, and backend restart reconciliation were
  exercised on Ubuntu 26.04 on 2026-09-22 as diagnostic evidence only. Linux and
  macOS are not supported release targets; no additional platform validation is
  planned or required for the Windows release.
- The latest public release is source-only `v4.4.0`; no installer, executable,
  package, or other binary artifact is part of that release workflow. Extract or
  clone the application folder and run `start_on_windows.ps1` from its root.
