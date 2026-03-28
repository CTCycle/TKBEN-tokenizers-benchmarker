# Windows Desktop Release v2.1.0 (2026-03-25)

## Scope
Release record for the Windows desktop package built through the Tauri workflow.

- Product: `TKBEN Desktop`
- App version: `2.1.0`
- Build date: `2026-03-25`
- Build entrypoint: `release/tauri/build_with_tauri.bat`

Repository verification date:
- `2026-03-28` (artifact names and paths below confirmed in `release/windows`)

## Build Prerequisites
- Rust toolchain with `cargo` available and configured
- Portable runtimes provisioned under `runtimes/`
- Runtime lockfile present at `runtimes/uv.lock`

## Build Command
Run from repository root:

```bat
release\tauri\build_with_tauri.bat
```

## Generated Artifacts
User-facing outputs are exported under `release/windows`.

Installers (`release/windows/installers`):
- `TKBEN Desktop_2.1.0_x64-setup.exe`
- `TKBEN Desktop_2.1.0_x64_en-US.msi`

Portable (`release/windows/portable`):
- `tkben-desktop.exe`
- packaged runtime payload (`TKBEN`, `resources`, `runtimes`, lockfiles)

## Observed Build Notes
- Frontend dependency install and production build completed in pipeline.
- Tauri bundling/export completed for NSIS and MSI installers.
- Frontend bundle size warning may appear for chunks larger than 500 kB.

## GitHub Release Publishing
Suggested metadata:
- Tag: `v2.1.0`
- Title: `TKBEN Desktop v2.1.0`

Attach these artifacts:
- `release/windows/installers/TKBEN Desktop_2.1.0_x64-setup.exe`
- `release/windows/installers/TKBEN Desktop_2.1.0_x64_en-US.msi`
- `release/windows/portable/tkben-desktop.exe`
