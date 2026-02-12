# TKBEN Tokenizer Benchmarker

## 1. Project Overview
TKBEN is a web application for exploring and benchmarking open-source tokenizers against text datasets. It helps compare tokenizer speed, token counts, and coverage so you can choose a tokenizer for your NLP workload.

The system is split into a Python/FastAPI backend and a React frontend. The backend downloads tokenizers and datasets, runs benchmarks, and stores results in a local database. The frontend provides the UI for managing datasets, running benchmarks, and visualizing results.

> **Work in Progress**: This project is still under active development. It will be updated regularly, but you may encounter bugs, issues, or incomplete features.

## 2. Model and Dataset (Optional)
This project does not train or fine-tune machine learning models. It benchmarks existing tokenizers (including Hugging Face tokenizers) using text datasets.

Datasets can be downloaded from Hugging Face or uploaded by the user (CSV/XLS/XLSX) and are stored locally for analysis. Results depend on the chosen datasets and tokenizers.

## 3. Installation

### 3.1 Windows (One Click Setup)
Run `TKBEN/start_on_windows.bat`.

The launcher performs the following steps in order:
- Downloads portable Python, uv, and Node.js into `TKBEN/resources/runtimes`.
- Installs backend dependencies from `pyproject.toml`.
- Installs frontend dependencies and builds the UI if needed.
- Starts the backend and frontend servers and opens the UI in your browser.

First run downloads and builds the required runtimes and frontend assets. Subsequent runs reuse the local runtimes and only rebuild when missing.
The setup is portable and only writes to the project directory.

### 3.2 macOS / Linux (Manual Setup)
Manual setup is required on macOS and Linux.

Prerequisites:
- Python 3.14+ (to match the Windows launcher runtime)
- Node.js 22+
- `uv`

Steps:
1. From the repository root, install backend dependencies:
   ```bash
   uv sync --all-extras
   ```
2. Start the backend:
   ```bash
   uv run python -m uvicorn TKBEN.server.app:app --host 127.0.0.1 --port 8000
   ```
3. In a separate terminal, install and build the frontend:
   ```bash
   cd TKBEN/client
   npm install
   npm run build
   npm run preview -- --host 127.0.0.1 --port 5173 --strictPort
   ```

## 4. How to Use

### 4.1 Windows
Launch `TKBEN/start_on_windows.bat`. The UI opens at `http://127.0.0.1:5173`.
The backend API is available at `http://127.0.0.1:8000`, with documentation at `http://127.0.0.1:8000/docs`.

### 4.2 macOS / Linux
Start the backend and frontend with the commands from the manual setup section:
- Backend: `uv run python -m uvicorn TKBEN.server.app:app --host 127.0.0.1 --port 8000`
- Frontend: `npm run preview -- --host 127.0.0.1 --port 5173 --strictPort`

The UI is available at `http://127.0.0.1:5173`, and the backend API at `http://127.0.0.1:8000` (docs at `http://127.0.0.1:8000/docs`).

### 4.3 Using the Application
Typical workflow:
- Load a dataset by downloading from Hugging Face or uploading a CSV/XLS/XLSX file.
- Scan and select tokenizers to benchmark.
- Run benchmarks and review token counts, throughput, and other metrics.
- Inspect charts and tables, and keep results for later comparisons.

**Dataset management screen**:

![Dataset management and download/upload](assets/figures/dataset_page.png)

**Benchmark results screen**:

![Benchmark charts and metrics](assets/figures/benchmarks_page.png)

**Database browser**:

![Database browser](assets/figures/database_browser.png)


## 5. Setup and Maintenance
Run `TKBEN/setup_and_maintenance.bat` and choose an action:
- Remove logs: deletes log files from `TKBEN/resources/logs`.
- Uninstall app: removes local runtimes, caches, virtual environment, frontend dependencies, and built assets.
- Initialize database: recreates or prepares the local database used by the backend.

## 6. Resources
- database: local SQLite database file `TKBEN/resources/database/sqlite.db` storing datasets and benchmark results.
- datasets: downloaded or uploaded datasets used for benchmarking, stored under `TKBEN/resources/datasets`.
- logs: backend and launcher logs stored in `TKBEN/resources/logs`.
- runtimes: portable Python/uv/Node.js used by the Windows launcher, stored in `TKBEN/resources/runtimes`.
- templates: sample files such as a starter `.env` in `TKBEN/resources/templates`.

## 7. Configuration
Backend configuration is defined in `TKBEN/settings/configurations.json` and can be overridden via environment variables in `TKBEN/settings/.env` (loaded on startup).
Frontend hosting (host/port) is controlled by the Windows launcher and Vite using the same `TKBEN/settings/.env` file; there is no separate frontend configuration file.

| Variable | Description |
|----------|-------------|
| FASTAPI_HOST | Backend bind host; set in `TKBEN/settings/.env` (read by `TKBEN/start_on_windows.bat`), default `127.0.0.1`. |
| FASTAPI_PORT | Backend bind port; set in `TKBEN/settings/.env`, default `8000`. |
| UI_HOST | Frontend bind host for preview; set in `TKBEN/settings/.env` if overridden, default `127.0.0.1` in `TKBEN/start_on_windows.bat`. |
| UI_PORT | Frontend bind port for preview; set in `TKBEN/settings/.env` if overridden, default `5173` in `TKBEN/start_on_windows.bat`. |
| RELOAD | Enables FastAPI reload; set in `TKBEN/settings/.env`, default `true`. |
| VITE_API_BASE_URL | Frontend API base path; set in `TKBEN/settings/.env`, default `/api`. |
| HF_KEYS_ENCRYPTION_KEY | Required Fernet key used to encrypt/decrypt stored Hugging Face keys; must be provided per environment and not committed as a shared secret. |
| MPLBACKEND | Matplotlib backend; set in `TKBEN/settings/.env`, default `Agg`. |


## 8. License
This project is licensed under the MIT license. See `LICENSE` for details.

