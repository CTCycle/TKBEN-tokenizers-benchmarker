# Deployment
Last updated: 2026-07-11

## Dependency Prerequisites
From project and runtime scripts:
- Developer mode bootstraps pinned Python 3.14.2, Node.js 22.14.0, and uv locally.
- Desktop builds require stable Rust/Cargo with `x86_64-pc-windows-msvc`, Windows SDK/WiX, and network access to assemble the locked runtime.

## Packaging Notes
### Desktop Build Outputs
- `release/windows/TKBEN-Desktop-<version>-windows-x64.msi`
- `release/windows/TKBEN-Desktop-<version>-windows-x64-portable.zip`
- `release/windows/TKBEN-Desktop-<version>-windows-x64-SHA256SUMS.txt`

### Packaging Flow Summary
- Restore frontend dependencies with `npm ci` and build the SPA.
- Export locked backend requirements and install them into an embedded Python payload at build time.
- Build Tauri 2 for Windows x64/MSI and export a portable resource layout.
- End-user startup does not invoke uv or require developer tools.
- Tauri packaging is rooted at `app/src-tauri`, with frontend output read from `app/client/dist`.

## Local Distribution Strategy
- For non-packaged use, the repository plus `start_on_windows.ps1` is the operational deployment path.
- The supported built-in security mode is local-only: keep `FASTAPI_HOST=127.0.0.1` or `localhost`.
- Network-hosted deployments require an external authentication boundary before exposing key management or destructive API routes.
- The backend refuses non-loopback binds by default. `TKBEN_ALLOW_UNAUTHENTICATED_NETWORK_BIND=true` is an explicit override for environments that provide their own access control.

## Constraints
- The repository does not currently include an active Docker runtime configuration in the root.
- Windows ARM64 and x86 are not supported release targets.
- Unsigned local builds are allowed; CI signs when certificate secrets are configured.
