# Deployment
Last updated: 2026-06-06

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

## Constraints
- The repository does not currently include an active Docker runtime configuration in the root.
