# Deployment
Last updated: 2026-07-11

## Dependency Prerequisites
From project and runtime scripts:
- Windows launcher mode bootstraps pinned Python 3.14.2, Node.js 22.12.0, and uv locally.
- The launcher downloads portable runtimes into the ignored `runtimes/` directory when they are missing.
- Manual macOS/Linux use requires system Python 3.14+, Node.js 22+, and uv.

## Local Distribution Strategy
- The repository plus `start_on_windows.ps1` is the supported Windows operational path.
- The launcher synchronizes Python and frontend dependencies, builds the frontend, then starts FastAPI and Vite preview locally.
- The supported built-in security mode is local-only: keep `FASTAPI_HOST=127.0.0.1` or `localhost`.
- Network-hosted deployments require an external authentication boundary before exposing key management or destructive API routes.
- The backend refuses non-loopback binds by default. `TKBEN_ALLOW_UNAUTHENTICATED_NETWORK_BIND=true` is an explicit override for environments that provide their own access control.

## Constraints
- The repository does not currently include an active Docker runtime configuration in the root.
- Automatic Python and Node.js downloads target Windows x64.
