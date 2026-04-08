# TKBEN User Manual
Last updated: 2026-04-08

## 1. Purpose
This manual explains how to use TKBEN for dataset validation, tokenizer analysis, and cross-tokenizer benchmarking.

## 2. Access the Application

### 2.1 Windows local webapp mode
Run from repository root:

```bat
.\TKBEN\start_on_windows.bat
```

Open the UI at:
- `http://<UI_HOST>:<UI_PORT>`

Default values from `.env.example` are:
- `UI_HOST=127.0.0.1`
- `UI_PORT=8000`

### 2.2 Packaged desktop mode
Prepare environment and build:

```bat
copy /Y TKBEN\settings\.env.example TKBEN\settings\.env
.\release\tauri\build_with_tauri.bat
```

Start the packaged executable from `release/windows/portable` or install from `release/windows/installers`.

## 3. Primary Navigation
Main pages:
- `/dataset`
- `/tokenizers`
- `/cross-benchmark`

Global operations available from the shell/header include Hugging Face key management and route switching.

## 4. User Journeys

### 4.1 Journey A: Validate a dataset
1. Open `/dataset`.
2. Choose one ingestion path:
- Download dataset from Hugging Face.
- Upload a local `.csv`, `.xls`, or `.xlsx`.
3. Start analysis.
4. Polling is automatic through the jobs API.
5. Review report outputs:
- document and word length histograms
- aggregate statistics
- most/least common words
- word cloud terms
- per-document statistics

### 4.2 Journey B: Analyze tokenizers
1. Open `/tokenizers`.
2. Configure or confirm an active Hugging Face key.
3. Scan tokenizer IDs from Hugging Face.
4. Download selected tokenizers.
5. Generate tokenizer report for one tokenizer.
6. Review outputs:
- vocabulary statistics
- token length histogram
- tokenizer metadata and derived metrics
7. Optionally upload custom `tokenizer.json` and use it for benchmark runs.

### 4.3 Journey C: Run cross-benchmark
1. Open `/cross-benchmark`.
2. Select an already loaded dataset.
3. Select tokenizers (downloaded and optionally one uploaded custom tokenizer).
4. Choose metric categories and run configuration.
5. Start benchmark run.
6. Review benchmark dashboard outputs:
- efficiency/throughput
- latency
- fidelity
- fragmentation
- resource metrics
7. Reopen historical reports for comparison.

## 5. Primary Backend Commands
Common local operations:

Start local app:
```bat
.\TKBEN\start_on_windows.bat
```

Maintenance utility:
```bat
.\TKBEN\setup_and_maintenance.bat
```

Run backend manually:
```bat
uv run python -m uvicorn TKBEN.server.app:app --host 127.0.0.1 --port 5000
```

Run frontend manually:
```bat
cd TKBEN/client
npm ci
npm run preview -- --host 127.0.0.1 --port 8000 --strictPort
```

## 6. Usage Patterns
- Ingest once, analyze many times: load a dataset, then run validation with different metric selections.
- Cache tokenizers before reporting: scan/download in batches, then generate per-tokenizer reports on demand.
- Compare benchmark runs by report ID over time to validate tokenizer choices under different metric sets.
- Use async job status as source of truth for long-running operations (`/jobs`, `/jobs/{job_id}`).

## 7. Key Features Summary
- Dataset ingestion from Hugging Face and local file upload.
- Dataset validation and statistics reports with persisted history.
- Tokenizer scan/download/report workflows.
- Support for uploaded custom tokenizer JSON files (session-scoped).
- Cross-tokenizer benchmark execution with persisted reports.
- PDF export endpoint for dashboard reports.
- Dual API routing (`/` and `/api` aliases) for compatibility across runtime modes.

