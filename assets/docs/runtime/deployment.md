# Deployment
Last updated: 2026-06-21

## Dependency Prerequisites
From project and runtime scripts:
- Python `>=3.14`
- Node.js `>=22`
- `uv`
- Rust and Cargo are required only for Tauri packaging

## Packaging Notes
### Desktop Build Outputs
- `release/windows/installers`
- `release/windows/portable`

### Packaging Flow Summary
- Build frontend with `npm run build`
- Build Tauri app with `npm run tauri:build:release`
- Export artifacts with `release/tauri/scripts/export-windows-artifacts.ps1`
- Tauri packaging is rooted at `app/src-tauri`, with frontend output read from `app/client/dist`.

## Local Distribution Strategy
- For non-packaged use, the repository plus `start_on_windows.bat` is the operational deployment path.
- The supported built-in security mode is local-only: keep `FASTAPI_HOST=127.0.0.1` or `localhost`.
- Network-hosted deployments require an external authentication boundary before exposing key management or destructive API routes.
- The backend refuses non-loopback binds by default. `TKBEN_ALLOW_UNAUTHENTICATED_NETWORK_BIND=true` is an explicit override for environments that provide their own access control.

## Constraints
- The repository does not currently include an active Docker runtime configuration in the root.
