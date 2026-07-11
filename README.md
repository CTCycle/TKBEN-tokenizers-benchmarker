# TKBEN Tokenizer Benchmarker
[![Release](https://img.shields.io/github/v/release/CTCycle/TKBEN-tokenizers-benchmarker?display_name=tag)](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/releases)
![Python](https://img.shields.io/badge/python-%3E%3D3.14-3776AB?logo=python&logoColor=white)
![Node.js](https://img.shields.io/badge/node.js-%3E%3D22-339933?logo=node.js&logoColor=white)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CTCycle Portfolio](https://img.shields.io/badge/CTCycle-Portfolio-58a6ff?style=flat-square)](https://ctcycle.github.io/CTCycle/)
[![CI](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/workflows/ci.yml?query=branch%3Adevelop)

## 1. Project Overview

TKBEN is a tokenizer benchmarking platform for text datasets and tokenizer assets.

Main workflow routes:
- `/dataset`
- `/tokenizers`
- `/cross-benchmark`

Runtime model:
- **Local webapp mode (default)**: run directly on host via `start_on_windows.ps1`.
- **Packaged desktop mode**: build and run the local Tauri package.
- **Single runtime env file**: launcher-managed runs edit `%LOCALAPPDATA%\TKBEN\config\.env`, seeded from `settings/.env.example`.

Packaged and launcher-managed runs store mutable data under `%LOCALAPPDATA%\TKBEN`; repository paths are used only by direct/manual development and tests.

## 2. Installation

### 2.1 Windows (One-Click Local Setup)

Run the single developer and maintenance entry point from any directory:

```powershell
.\start_on_windows.ps1
```

`start_on_windows.ps1` is the sole root launcher and opens the full combined menu. Choose **Launch application** to download pinned portable Python, uv, and Node.js runtimes on first use, restore both committed lockfiles, initialize user configuration/data, then start FastAPI and Vite. Later launches reuse the synchronized environment and do not rebuild Tauri.

### 2.2 Windows Packaged Desktop (Tauri)

Supported release target: Windows 10/11 x64. End users need no Python, Node.js, uv, Rust, or source checkout. Microsoft WebView2 is installed through Tauri's supported bootstrap flow when it is absent.

Build prerequisites: Rust stable with the x64 MSVC target, Windows SDK/WiX support, and the development runtimes installed by `start_on_windows.ps1`.

```bat
.\release\tauri\build_with_tauri.bat
```

Outputs:
- `release/windows/TKBEN-Desktop-<version>-windows-x64-portable.zip`
- `release/windows/TKBEN-Desktop-<version>-windows-x64.msi`
- `release/windows/TKBEN-Desktop-<version>-windows-x64-SHA256SUMS.txt`

### 2.3 macOS / Linux (Manual Local Setup)

**Prerequisites**:
- Python 3.14+
- Node.js 22+
- `uv`

**Setup Steps**:
1. Install backend dependencies
   ```bash
   uv sync
   ```
2. Start backend
   ```bash
   uv run python -m uvicorn server.app:app --host 127.0.0.1 --port 5000
   ```
3. Start frontend (new terminal)
   ```bash
   cd app/client
   npm ci
   npm run preview -- --host 127.0.0.1 --port 8000 --strictPort
   ```

## 3. How to Use

### 3.1 Runtime Configuration (`.env`)

Developer launcher and packaged desktop runs seed user configuration at `%LOCALAPPDATA%\TKBEN\config`. `settings/.env.example` and `settings/configurations.json` are immutable templates.

Initialize from the single template:
```bat
Copy-Item settings\.env.example "$env:LOCALAPPDATA\TKBEN\config\.env"
```

### 3.2 Local Webapp Mode (Default)

```powershell
.\start_on_windows.ps1
```

Runtime addresses are taken from the user configuration:
- **Web UI**: `http://<UI_HOST>:<UI_PORT>`
- **Backend API**: `http://<FASTAPI_HOST>:<FASTAPI_PORT>`
- **API Docs**: `http://<FASTAPI_HOST>:<FASTAPI_PORT>/docs`

### 3.3 Packaged Desktop Mode (Tauri)

The desktop app starts its bundled backend automatically and performs no dependency installation. Application data, downloaded tokenizer/dataset content, configuration, caches, and logs are stored below `%LOCALAPPDATA%\TKBEN`. Closing the desktop window terminates the managed backend process tree.

### 3.4 Application Flow

**Dataset (`/dataset`)**

Load data from Hugging Face presets or manual IDs, or upload local CSV/XLS/XLSX files. Then run dataset analysis and reopen saved reports for statistics and charts.

**Tokenizers (`/tokenizers`)**

Scan available tokenizer IDs, download selected tokenizers, optionally upload a custom `tokenizer.json`, and inspect tokenizer reports.

**Cross Benchmark (`/cross-benchmark`)**

Create benchmark runs by selecting dataset, tokenizers, and metric categories, then compare persisted results across tokenizer candidates.
### 3.5 Product Snapshots

The following snapshots were captured in local webapp mode with backend and frontend running:

- Dataset dashboard with a loaded persisted validation session, aggregate statistics, histograms, and word-cloud analytics.
![Dataset workspace](assets/figures/release-v3.4.0-dataset.png)

- Tokenizers dashboard with an opened tokenizer report, vocabulary statistics, and populated token preview table.
![Tokenizers workspace](assets/figures/release-v3.4.0-tokenizers.png)

- Cross-benchmark dashboard with a loaded run summary and comparative metric panels.
![Cross-benchmark dashboard](assets/figures/release-v3.4.0-cross-benchmark.png)

## 4. Setup and Maintenance

Run the unified menu:

```powershell
.\start_on_windows.ps1
```

The same menu contains launch and all maintenance actions: first-time install, locked dependency synchronization, deliberate lockfile upgrades, database initialization, repair, generated-cache cleanup, diagnostics, log viewing, service shutdown, tests, desktop build/output cleanup, and developer-runtime uninstall. For automation, use the same entry point, for example `.\start_on_windows.ps1 -Action Diagnostics`.

## 5. Resources

Key paths:
- `%LOCALAPPDATA%\TKBEN\data`: SQLite database and mutable application data.
- `%LOCALAPPDATA%\TKBEN\config`: User `.env` and structured configuration.
- `%LOCALAPPDATA%\TKBEN\logs`: Developer and desktop logs.
- `%LOCALAPPDATA%\TKBEN\cache`: Runtime/model/rendering caches.
- `runtimes`: Portable Windows runtimes.
- `assets/docs/project_index.md`: Documentation root index and topic map.
- `assets/docs/runtime/modes.md`: Runtime packaging and mode details.
- `assets/docs/runtime/startup.md`: Startup procedures and launcher commands.
- `app/src-tauri`: Desktop shell and packaging configuration.

## 6. Configuration

Configuration is split between:
- `settings/.env.example`: Versioned configuration template; `settings/.env` must never be committed.
- `settings/configurations.json`: Backend structured settings for datasets, tokenizers, benchmarks, jobs, and optional database overrides.

Core runtime keys you will commonly edit:
- `FASTAPI_HOST`, `FASTAPI_PORT`
- `UI_HOST`, `UI_PORT`
- `VITE_API_BASE_URL` (normally `/api`)
- `RELOAD`
- `DATABASE_EMBEDDED`
- `DATABASE_URL`
- `DATABASE_HOST`, `DATABASE_PORT`
- `DATABASE_NAME`
- `DATABASE_USERNAME`, `DATABASE_PASSWORD`
- `HF_KEYS_ENCRYPTION_KEY`

Determinism:
- Backend lockfile: `app/server/uv.lock` (generated/updated directly by running `uv sync` from `app/server`).
- Frontend lockfile: committed `app/client/package-lock.json` + `npm ci`.

## 7. Releases and Repository Hygiene

Tags matching `v*` and manual workflow dispatches run `.github/workflows/desktop-release.yml`. The workflow validates the locked backend/frontend/Rust sources, builds the x64 MSI and portable ZIP, verifies required runtime content, generates SHA-256 checksums, and publishes only final distributables. Optional signing uses the `WINDOWS_SIGNING_CERTIFICATE_BASE64` and `WINDOWS_SIGNING_CERTIFICATE_PASSWORD` repository secrets.

Never commit `.env` files, credentials, signing certificates, databases, downloaded model/data caches, logs, virtual environments, Node dependencies, Tauri targets, prepared runtime payloads, MSI/EXE/ZIP outputs, or `release/windows` contents. Keep `package-lock.json`, `uv.lock`, `Cargo.lock`, Tauri configuration/source, icons, templates, scripts, workflows, and migrations tracked.

## 8. License

This project is licensed under the MIT License. See `LICENSE` for details.


