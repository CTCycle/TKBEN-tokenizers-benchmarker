# TKBEN Tokenizer Benchmarker

## 1. Project Overview

TKBEN is a tokenizer benchmarking platform for text datasets and tokenizer assets.

Main workflow routes:
- `/dataset`
- `/tokenizers`
- `/cross-benchmark`

Runtime model:
- **Local webapp mode (default)**: run directly on host via `TKBEN/start_on_windows.bat`.
- **Packaged desktop mode**: build and run the local Tauri package.
- **Single runtime env file**: edit `TKBEN/settings/.env` (seed from `TKBEN/settings/.env.example`).

Default embedded storage is the SQLite file at `TKBEN/resources/database.db`.

## 2. Installation

### 2.1 Windows (One-Click Local Setup)

Run the launcher from repository root:

```bat
.\TKBEN\start_on_windows.bat
```

The launcher will:
1. Install portable Python, `uv`, and Node.js under `runtimes`.
2. Install backend dependencies using `uv sync`.
3. Install frontend dependencies (`npm ci` when lockfile is present, fallback to `npm install`).
4. Build and start backend + frontend.

### 2.2 Windows Packaged Desktop (Tauri)

Prerequisites:
- Rust toolchain (`cargo`) with a default toolchain configured.

```bat
copy /Y TKBEN\settings\.env.example TKBEN\settings\.env
.\release\tauri\build_with_tauri.bat
```

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
   uv run python -m uvicorn TKBEN.server.app:app --host 127.0.0.1 --port 5000
   ```
3. Start frontend (new terminal)
   ```bash
   cd TKBEN/client
   npm ci
   npm run preview -- --host 127.0.0.1 --port 8000 --strictPort
   ```

## 3. How to Use

### 3.1 Runtime Configuration (`.env`)

Use `TKBEN/settings/.env` as the active runtime file for both local webapp mode and packaged desktop mode.

Initialize from the single template:
```bat
copy /Y TKBEN\settings\.env.example TKBEN\settings\.env
```

### 3.2 Local Webapp Mode (Default)

```bat
.\TKBEN\start_on_windows.bat
```

Runtime addresses are taken from `TKBEN/settings/.env`:
- **Web UI**: `http://<UI_HOST>:<UI_PORT>`
- **Backend API**: `http://<FASTAPI_HOST>:<FASTAPI_PORT>`
- **API Docs**: `http://<FASTAPI_HOST>:<FASTAPI_PORT>/docs`

### 3.3 Packaged Desktop Mode (Tauri)

```bat
copy /Y TKBEN\settings\.env.example TKBEN\settings\.env
.\release\tauri\build_with_tauri.bat
```

Build artifacts are produced under:
- `release/windows/installers`
- `release/windows/portable`

### 3.4 Application Flow

**Dataset (`/dataset`)**

Load data from Hugging Face presets or manual IDs, or upload local CSV/XLS/XLSX files. Then run dataset analysis and reopen saved reports for statistics and charts.

**Tokenizers (`/tokenizers`)**

Scan available tokenizer IDs, download selected tokenizers, optionally upload a custom `tokenizer.json`, and inspect tokenizer reports.

**Cross Benchmark (`/cross-benchmark`)**

Create benchmark runs by selecting dataset, tokenizers, and metric categories, then compare persisted results across tokenizer candidates.
### 3.5 End-to-End Activity Snapshots

The following snapshots were captured from a full interactive E2E run in local webapp mode:

- Dataset dashboard after completed validation (5,000 sampled documents)
![Dataset dashboard complete](assets/figures/e2e-dataset-dashboard-complete.png)

- Tokenizer dashboard with loaded report (`bert-base-uncased`)
![Tokenizer dashboard](assets/figures/e2e-tokenizer-dashboard.png)

- Cross-benchmark run #1 report (`bert-base-uncased`, `gpt2`, `roberta-base`, full metric set)
![Benchmark report config 1](assets/figures/e2e-benchmark-report-config1.png)

- Cross-benchmark run #2 (`bert-base-uncased`, `roberta-base`, reduced metric set without per-document stats)
![Benchmark report config 2](assets/figures/e2e-benchmark-report-config2.png)

## 4. Setup and Maintenance

Run:

```bat
.\TKBEN\setup_and_maintenance.bat
```

Available actions:
- **Remove logs**: Deletes `*.log` files under `TKBEN/resources/logs`.
- **Uninstall app**: Removes local runtime artifacts under `runtimes\` (including `runtimes\.venv` and `runtimes\uv.lock`), frontend `node_modules`, frontend `dist`, and related local artifacts.
- **Initialize database**: Executes `TKBEN/scripts/initialize_database.py` using local `uv` + portable Python runtime.

## 5. Resources

Key paths:
- `TKBEN/resources/database.db`: Embedded SQLite database path.
- `TKBEN/resources/sources/datasets`: Dataset source/download artifacts.
- `TKBEN/resources/sources/tokenizers`: Tokenizer source/download artifacts.
- `TKBEN/resources/logs`: Launcher and backend logs.
- `runtimes`: Portable Windows runtimes.
- `assets/docs/PACKAGING_AND_RUNTIME_MODES.md`: Runtime packaging and mode details.
- `assets/docs/USER_MANUAL.md`: End-user journeys, commands, and feature usage.

## 6. Configuration

Configuration is split between:
- `TKBEN/settings/.env`: Active runtime/process configuration used by launcher, tests, frontend dev/preview, and desktop startup.
- `TKBEN/settings/.env.example`: Single template for `.env`.
- `TKBEN/settings/configurations.json`: Backend structured settings, including all database mode/connection/tuning values.

Core runtime keys you will commonly edit:
- `FASTAPI_HOST`, `FASTAPI_PORT`
- `UI_HOST`, `UI_PORT`
- `VITE_API_BASE_URL` (normally `/api`)
- `RELOAD`
- `HF_KEYS_ENCRYPTION_KEY`

Determinism:
- Backend lockfile: `runtimes/uv.lock` (copied to root `uv.lock` before `uv sync --frozen`, then copied back).
- Frontend lockfile: committed `TKBEN/client/package-lock.json` + `npm ci` in local runtime/bootstrap flow.

## 7. License

This project is licensed under the MIT License. See `LICENSE` for details.

