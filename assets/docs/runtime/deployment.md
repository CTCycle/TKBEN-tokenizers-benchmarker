# Deployment
Last updated: 2026-08-02

## Dependency Prerequisites
From project and runtime scripts:
- Windows launcher mode bootstraps pinned Python 3.14.2, Node.js 22.13.0, and uv locally.
- The launcher downloads portable runtimes into the ignored `runtimes/` directory when they are missing.
- Manual macOS/Linux use requires system Python 3.14+, Node.js 22+, and uv.

## Local Distribution Strategy
- The repository plus `start_on_windows.ps1` is the supported Windows operational path.
- The launcher synchronizes Python dependencies, reuses unchanged frontend dependencies on application launch, builds the frontend when configured, then starts FastAPI and Vite preview locally.
- The default launcher binds locally with `FASTAPI_HOST=127.0.0.1`.
- Network-hosted deployments require an external authentication boundary before exposing key management or destructive API routes.

## Constraints
- The repository does not currently include an active Docker runtime configuration in the root.
- Automatic Python and Node.js downloads target Windows x64.
- The current public release is source-only `v3.7.1`; no installer, executable,
  package, or other binary artifact is part of that release workflow.
