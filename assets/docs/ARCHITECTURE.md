# Architecture

## 1. High-Level Architecture Overview

### 1.1 Purpose and Scope
TKBEN is a local web application for downloading text datasets and tokenizers, running tokenizer benchmarks, and exploring results through charts and a database browser. It focuses on comparing tokenizer behavior and performance (speed, vocabulary traits, and tokenization metrics) using datasets stored locally in a database.

### 1.2 System Overview
- Frontend components: React + TypeScript SPA with pages for dataset management, tokenizer selection/benchmarking, and a database browser. State is managed via React contexts and API helpers. See `TKBEN/client/src/App.tsx`, `TKBEN/client/src/pages`, `TKBEN/client/src/contexts`, `TKBEN/client/src/services`.
- Backend services: FastAPI app exposing dataset, tokenizer, benchmark, and browser endpoints. Heavy work is delegated to service classes (dataset processing, benchmarking, and tokenizer scanning) with persistence via SQLAlchemy repositories. See `TKBEN/server/app.py`, `TKBEN/server/routes`, `TKBEN/server/utils/services`, `TKBEN/server/database`.
- External dependencies and integrations: HuggingFace `datasets` for dataset downloads, `transformers` and `tokenizers` for tokenizer loading/processing, `huggingface_hub` for tokenizer discovery, plus pandas/numpy/matplotlib for metrics and chart data. See `pyproject.toml`.

### 1.3 Deployment and Runtime Assumptions
- Local-first runtime: Windows batch scripts install portable Python (embeddable) and Node.js, then run FastAPI (uvicorn) and Vite preview. See `TKBEN/start_on_windows.bat`.
- Default ports: backend `FASTAPI_PORT=8000`, UI `UI_PORT=7861` with `.env` overrides in `TKBEN/settings/.env`. See `TKBEN/start_on_windows.bat`, `TKBEN/server/utils/variables.py`, `TKBEN/resources/templates/.env`.
- Configuration: JSON config in `TKBEN/settings/configurations.json` with environment variable overrides via `.env` (loaded at runtime). See `TKBEN/server/utils/configurations/server.py`.
- Database: embedded SQLite file at `TKBEN/resources/database/sqlite.db` by default, optional PostgreSQL when `embedded_database=false`. See `TKBEN/server/utils/constants.py`, `TKBEN/server/database/sqlite.py`, `TKBEN/server/database/postgres.py`.
- Frontend dev proxy: Vite proxies `/api` to `http://localhost:8000` in dev. See `TKBEN/client/vite.config.ts`.

---

## 2. Codebase Structure

### 2.1 Directory Layout
| Path | Description |
|-----|-------------|
| `/TKBEN` | Main application package and assets |
| `/TKBEN/server` | FastAPI backend, routes, services, database layer |
| `/TKBEN/client` | React + Vite frontend |
| `/TKBEN/settings` | Runtime configuration JSON and `.env` overrides |
| `/TKBEN/resources` | Runtime data (datasets cache, database, logs, templates) |
| `/docs` | Project documentation |

### 2.2 Key Modules
- Backend entrypoint: `TKBEN/server/app.py` creates the FastAPI app and mounts routers.
- API routes: `TKBEN/server/routes/*.py` (datasets, tokenizers, benchmarks, browser).
- Services: `TKBEN/server/utils/services/*.py` for dataset ingestion, tokenizer scanning, and benchmarking logic.
- Persistence: `TKBEN/server/database/*.py` for repository implementations and schema definitions.
- Configuration: `TKBEN/server/utils/configurations/*.py` and `TKBEN/settings/configurations.json`.
- Frontend pages: `TKBEN/client/src/pages/*.tsx` (DatasetPage, TokenizersPage, DatabaseBrowserPage).
- Frontend contexts/services: `TKBEN/client/src/contexts/*.tsx`, `TKBEN/client/src/services/*.ts`.

### 2.3 Core Classes and Functions
- `DatasetService` handles dataset download/upload, persistence, and analysis: `TKBEN/server/utils/services/datasets.py`.
- `BenchmarkService.run_benchmarks` loads tokenizers, computes metrics, persists results, and builds chart data: `TKBEN/server/utils/services/benchmarks.py`.
- `TokenizersService.get_tokenizer_identifiers` scans HuggingFace model hub: `TKBEN/server/utils/services/tokenizers.py`.
- `TKBENWebappDatabase` and repository backends for SQLite/Postgres: `TKBEN/server/database/database.py`, `TKBEN/server/database/sqlite.py`, `TKBEN/server/database/postgres.py`.
- SQLAlchemy schema models (tables): `TKBEN/server/database/schema.py`.
- FastAPI route handlers: `TKBEN/server/routes/datasets.py`, `TKBEN/server/routes/tokenizers.py`, `TKBEN/server/routes/benchmarks.py`, `TKBEN/server/routes/browser.py`.
- Frontend state orchestration: `DatasetProvider` and `TokenizersProvider`: `TKBEN/client/src/contexts/DatasetContext.tsx`, `TKBEN/client/src/contexts/TokenizersContext.tsx`.

---

## 3. Backend API

### 3.1 API Overview
The backend is a REST-style FastAPI service exposing JSON endpoints. Pydantic models define request/response schemas, and FastAPI serves OpenAPI docs under `/docs`. See `TKBEN/server/app.py`, `TKBEN/server/schemas/*.py`.

### 3.2 Endpoints
| Method | Route | Description |
|-------|-------|-------------|
| GET | `/` | Redirects to `/docs` |
| GET | `/datasets/list` | List datasets in the database |
| POST | `/datasets/download` | Download a dataset from HuggingFace and store it |
| POST | `/datasets/upload` | Upload a CSV/Excel dataset and store it |
| POST | `/datasets/analyze` | Analyze word-level statistics for a dataset |
| GET | `/tokenizers/settings` | Get scan limits and tokenizer settings |
| GET | `/tokenizers/scan` | Fetch popular tokenizer identifiers from HuggingFace |
| POST | `/tokenizers/upload` | Upload a custom `tokenizer.json` |
| DELETE | `/tokenizers/custom` | Clear uploaded custom tokenizers |
| POST | `/benchmarks/run` | Run benchmarks on tokenizers for a dataset |
| GET | `/browser/tables` | List database tables for browsing |
| GET | `/browser/data` | Paginate through table data |

### 3.3 Request and Response Models
- Dataset models: `DatasetDownloadRequest`, `DatasetDownloadResponse`, `DatasetAnalysisRequest`, `DatasetAnalysisResponse`, `CustomDatasetUploadResponse`, `HistogramData` in `TKBEN/server/schemas/dataset.py`.
- Tokenizer models: `TokenizerScanResponse`, `TokenizerSettingsResponse`, `TokenizerUploadResponse` in `TKBEN/server/schemas/tokenizers.py`.
- Benchmark models: `BenchmarkRunRequest`, `BenchmarkRunResponse`, `GlobalMetrics`, `ChartData` and chart sub-models in `TKBEN/server/schemas/benchmarks.py`.
- Validation is mainly handled by Pydantic plus route-level checks (e.g., required dataset/tokenizers, file extensions). See `TKBEN/server/routes/*.py`.

### 3.4 Authentication and Authorization
No authentication or authorization is implemented. Some endpoints accept a HuggingFace access token as a request parameter (`hf_access_token`) but it is not used for access control.

### 3.5 Error Handling
Routes raise `HTTPException` with 400/404/500 status codes and a `detail` message. Errors are logged via the backend logger. See `TKBEN/server/routes/*.py`, `TKBEN/server/utils/logger.py`.

---

## 4. Main Components

### 4.1 Component List
- Dataset ingestion and analysis (HuggingFace download, CSV/Excel upload, histogram + word stats).
- Tokenizer discovery and custom tokenizer upload.
- Benchmark engine computing per-tokenizer metrics and chart summaries.
- Persistence layer for datasets, metrics, vocabularies, and browser queries.
- Frontend dashboard and database browser UI.

### 4.2 Responsibilities and Boundaries
- Routes orchestrate requests, validate inputs, and format responses (`TKBEN/server/routes`).
- Services perform heavy IO/CPU work and database persistence (`TKBEN/server/utils/services`).
- Database repositories encapsulate SQLAlchemy engines and CRUD operations (`TKBEN/server/database`).
- Frontend pages render UI and call API helpers; contexts manage application state (`TKBEN/client/src/pages`, `TKBEN/client/src/contexts`).

---

## 5. Main Application Flows

### 5.1 Typical Request Flow
1. User triggers an action in the React UI (e.g., dataset download or benchmark run).
2. Frontend service calls a backend endpoint via `/api/*` (`TKBEN/client/src/services`).
3. FastAPI route validates input and calls a service method in a thread (`asyncio.to_thread`).
4. Service performs work (download/process/benchmark) and persists results to the database.
5. Route returns a Pydantic response which the UI renders into charts and stats.

### 5.2 Critical Workflows
- Dataset ingestion flow:
  1) `/datasets/download` or `/datasets/upload` invoked from DatasetPage.  
  2) `DatasetService` loads or parses the data, filters invalid text, computes histograms, and persists to `TEXT_DATASET`.  
  3) Response includes histogram summary for UI display.
- Dataset analysis flow:
  1) `/datasets/analyze` invoked.  
  2) `DatasetService.analyze_dataset` streams texts, computes per-document word stats, and stores results in `TEXT_DATASET_STATISTICS`.  
  3) Aggregate statistics are returned.
- Benchmark flow:
  1) `/benchmarks/run` invoked with dataset and tokenizer list.  
  2) `BenchmarkService` loads tokenizers, streams texts, computes metrics, and stores results in `TOKENIZATION_GLOBAL_METRICS` and vocabulary tables.  
  3) Chart data and global metrics are returned for the dashboard.
- Database browser flow:
  1) `/browser/tables` and `/browser/data` allow paginated browsing of table contents.  
  2) UI performs incremental fetch as the user scrolls (`DatabaseBrowserPage`).

---

## 6. Data Model and Data Structures

### 6.1 Core Domain Entities
- Dataset text records: dataset name + raw text lines.
- Dataset analysis stats: per-document word counts and average/std word lengths.
- Tokenizer benchmarks: per-tokenizer global metrics (speed, fertility, coverage).
- Vocabulary data: per-tokenizer vocabulary tokens and decoded tokens.

### 6.2 Database Schema
Defined in `TKBEN/server/database/schema.py` with SQLAlchemy:
- `TEXT_DATASET`: dataset_name, text (unique per dataset).
- `TEXT_DATASET_STATISTICS`: dataset_name, text, words_count, AVG_words_length, STD_words_length.
- `TOKENIZATION_GLOBAL_METRICS`: tokenizer, dataset_name, speed/coverage metrics.
- `VOCABULARY_STATISTICS`: tokenizer-level vocabulary composition stats.
- `VOCABULARY`: tokenizer + token_id + vocabulary/decoded token strings.
- `TOKENIZATION_LOCAL_STATS` exists in schema but is not currently persisted by `BenchmarkService`.

### 6.3 In-Memory Data Structures
- In-memory custom tokenizer registry: `custom_tokenizers` dict in `TKBEN/server/routes/tokenizers.py`.
- Pandas DataFrames for bulk inserts and metric aggregation in services.
- Streaming generators for dataset length/analysis to avoid full in-memory loads (`DatasetService`).
- Token frequency maps and intermediate arrays in `BenchmarkService` for metric computation.

---

## 7. Component Relationships

### 7.1 Dependency Graph
- Frontend SPA -> Backend REST API (`/api/*`).
- Backend routes -> Service layer -> Database repositories.
- Service layer -> HuggingFace APIs (`datasets`, `transformers`, `huggingface_hub`) for downloads.
- Database -> SQLite by default, optional PostgreSQL with config toggles.

### 7.2 Communication Patterns
- Synchronous HTTP/JSON between frontend and backend.
- Backend uses async endpoints but offloads heavy work to threads via `asyncio.to_thread`.
- No message queues or background job systems.

---

## 8. WebSocket Implementation

### 8.1 Presence and Purpose
WebSockets are not used.

### 8.2 Protocol and Message Format
Not applicable.

---

## 9. Database Browsing and Inspection

### 9.1 Admin or Browser Interfaces
The app includes a database browser UI (`TKBEN/client/src/pages/DatabaseBrowserPage.tsx`) backed by `/browser/tables` and `/browser/data` to list tables and stream rows.

### 9.2 Access Control and Security
No access control is implemented; any user with access to the UI can view database tables via the browser endpoints.

---

## 10. Training Pipeline and Dashboard

### 10.1 Training Workflow
No ML training pipeline is implemented. The application focuses on tokenizer benchmarking, not model training.

### 10.2 Model Artifacts
No model artifacts are stored or versioned in this codebase.

### 10.3 Dashboard and Monitoring
The frontend dashboard visualizes benchmark metrics and distributions from `/benchmarks/run`, but it is not a training/monitoring dashboard.

---

## 11. Known Limitations and Open Questions
- Some modules appear unrelated to the tokenizer benchmarker (e.g., adsorption/AEGIS references in `TKBEN/server/utils/repository/serializer.py`, `TKBEN/server/utils/services/downloads.py`, and `TKBEN/setup_and_maintenance.bat`). Clarify whether these are legacy files or still required.
- `TOKENIZATION_LOCAL_STATS` is defined in `TKBEN/server/database/schema.py` but not written by `BenchmarkService`; determine if local stats are planned or should be removed.
- README mentions `ACCESS_TOKEN`, but runtime code accepts `hf_access_token` per request and does not read `ACCESS_TOKEN`. Align documentation or implementation.
- Database filename mismatch: README references `tkben_database.db` while code uses `sqlite.db` (`TKBEN/server/utils/constants.py`).
- `TKBEN/resources/templates/.env` contains `ADSORFIT_API_URL` and appears unused by the server; confirm intended location and keys for env overrides.
- No authentication or multi-user separation is present; database browser exposes full tables to any UI user.
