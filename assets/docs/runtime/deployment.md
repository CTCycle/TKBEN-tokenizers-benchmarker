# Deployment
Last updated: 2026-09-21

## Dependency Prerequisites
From project and runtime scripts:
- Windows launcher mode bootstraps pinned Python 3.14.7 and Node.js 22.23.1 locally; it replaces an older managed Python runtime when required and downloads uv from the current uv release when uv is missing.
- The launcher downloads portable runtimes into the ignored `runtimes/` directory when they are missing.
- Manual macOS/Linux use requires system Python 3.14+, Node.js 22.22.3+ on a supported Angular engine line, and uv.

## Local Distribution Strategy
- The repository plus `start_on_windows.ps1` is the supported Windows operational path.
- The launcher synchronizes Python dependencies, reuses unchanged frontend dependencies on application launch, builds the frontend when dependencies or the production entry are missing, verifies the configured ports, then starts FastAPI and Angular preview locally.
- The default launcher binds locally with `FASTAPI_HOST=127.0.0.1`.
- Network-hosted deployments require an external authentication boundary before exposing key management or destructive API routes.

## Constraints
- The repository does not currently include an active Docker runtime configuration in the root.
- Automatic Python and Node.js downloads target Windows x64.
- The latest public release is source-only `v4.4.0`; no installer, executable,
  package, or other binary artifact is part of that release workflow. Extract or
  clone the application folder and run `start_on_windows.ps1` from its root.
