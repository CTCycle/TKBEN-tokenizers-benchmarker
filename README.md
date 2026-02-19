# TKBEN Tokenizer Benchmarker

## 1. Project Overview

TKBEN is a local benchmarking platform for text datasets and tokenizers. It helps teams ingest datasets, inspect tokenizer behavior, and compare tokenizer performance through repeatable benchmark runs.

The application is organized as:
- **Backend**: FastAPI services for data ingestion, tokenizer operations, benchmark jobs, and persisted reports.
- **Frontend**: React-based UI for dataset management, tokenizer analysis, and cross-benchmark reporting.

Main workflow routes:
- `/dataset`
- `/tokenizers`
- `/cross-benchmark`

Default embedded storage is the SQLite file at `TKBEN/resources/database.db`.

> **Work in Progress**: The project is actively evolving, so some features and workflows may still change.

---

## 2. Installation

### 2.1 Windows (One-Click Setup)

Run the launcher from the repository root:

```bat
.\TKBEN\start_on_windows.bat
```

The launcher will:
1. Install portable Python, `uv`, and Node.js under `TKBEN/resources/runtimes`.
2. Install backend dependencies from `pyproject.toml` using `uv sync`.
3. Install/build frontend dependencies when required.
4. Start backend + frontend services and open the web interface.

The first execution can take longer because runtimes and dependencies are downloaded locally.

### 2.2 macOS / Linux (Manual Setup)

**Prerequisites**:
- Python 3.14+
- Node.js 22+
- `uv`

**Setup Steps**:
1. **Install backend dependencies**
   ```bash
   uv sync
   ```
2. **Start backend**
   ```bash
   uv run python -m uvicorn TKBEN.server.app:app --host 127.0.0.1 --port 8000
   ```
3. **Start frontend** (new terminal)
   ```bash
   cd TKBEN/client
   npm install
   npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
   ```

---

## 3. How to Use

### 3.1 Windows

Launch with:

```bat
.\TKBEN\start_on_windows.bat
```

Runtime addresses are taken from `TKBEN/settings/.env` (or launcher defaults if that file is missing):
- **Web UI**: `http://<UI_HOST>:<UI_PORT>`
- **Backend API**: `http://<FASTAPI_HOST>:<FASTAPI_PORT>`
- **API Docs**: `http://<FASTAPI_HOST>:<FASTAPI_PORT>/docs`

### 3.2 macOS / Linux

Run backend/frontend manually with the commands from section `2.2`.

### 3.3 Application Flow

**Dataset (`/dataset`)**

Load data from Hugging Face presets or manual IDs, or upload local CSV/XLS/XLSX files. Then run dataset analysis and reopen saved reports for statistics and charts.

![Dataset workspace](./assets/figures/dataset_page.png)

**Tokenizers (`/tokenizers`)**

Scan available tokenizer IDs, download selected tokenizers, optionally upload a custom `tokenizer.json`, and inspect tokenizer reports.

![Tokenizer workspace](./assets/figures/tokenizer_page.png)

**Cross Benchmark (`/cross-benchmark`)**

Create benchmark runs by selecting dataset, tokenizers, and metric categories, then compare persisted results across tokenizer candidates.

![Cross benchmark workspace](./assets/figures/benchmarks_page.png)

---

## 4. Setup and Maintenance

Run:

```bat
.\TKBEN\setup_and_maintenance.bat
```

Available actions:
- **Remove logs**: Deletes `*.log` files under `TKBEN/resources/logs`.
- **Uninstall app**: Removes local runtimes, caches, `.venv`, frontend `node_modules`, frontend `dist`, and related local artifacts.
- **Initialize database**: Executes `TKBEN/scripts/initialize_database.py` using the local `uv` + portable Python runtime.

---

## 5. Resources

Key runtime paths:
- `TKBEN/resources/database.db`: Default embedded database.
- `TKBEN/resources/sources/datasets`: Dataset sources and download artifacts.
- `TKBEN/resources/sources/tokenizers`: Tokenizer sources and download artifacts.
- `TKBEN/resources/logs`: Launcher and backend log output.
- `TKBEN/resources/runtimes`: Portable Windows runtimes (Python, `uv`, Node.js).
- `assets/figures`: README images (`dataset_page.png`, `tokenizer_page.png`, `benchmarks_page.png`).

---

## 6. Configuration

Configuration is split between:
- `TKBEN/settings/configurations.json`: Backend operational settings (database mode, limits, batch sizes, polling intervals).
- `TKBEN/settings/.env`: Runtime variables used by launcher, backend, and frontend proxy behavior.

---

## 7. License

This project is licensed under the MIT License. See `LICENSE` for details.
