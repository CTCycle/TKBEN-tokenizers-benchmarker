# Windows Desktop Release v2.1.0 (2026-03-25)

## Scope
This release publishes the Windows desktop package built from the Tauri workflow.

- Product: `TKBEN Desktop`
- Tauri app version: `2.1.0`
- Build date: `2026-03-25`
- Build entrypoint: `release/tauri/build_with_tauri.bat`

## Build Prerequisites
- Rust toolchain with `cargo` available
- Portable runtimes provisioned under `runtimes/`
- Runtime lockfile available at `runtimes/uv.lock`

## Build Command
Run from repository root:

```bat
cmd /c release\tauri\build_with_tauri.bat
```

## Generated Artifacts
The build/export pipeline writes user-facing artifacts under `release/windows`.

- Installer EXE: `release/windows/installers/TKBEN Desktop_2.1.0_x64-setup.exe`
- Installer MSI: `release/windows/installers/TKBEN Desktop_2.1.0_x64_en-US.msi`
- Portable app: `release/windows/portable/tkben-desktop.exe`

## Observed Build Notes
- Frontend dependency install and production build completed successfully.
- Tauri bundling completed successfully for NSIS and MSI targets.
- A frontend bundle size warning was emitted for a chunk >500 kB.

## GitHub Release Publishing
Suggested tag and title for this build:

- Tag: `v2.1.0`
- Release title: `TKBEN Desktop v2.1.0`

Attach the three artifacts listed above to the GitHub release.
