# System Overview
Last updated: 2026-10-25

## System Summary
TKBEN is a tokenizer benchmarking platform with:
- FastAPI backend (`app/server`)
- Angular 22 frontend (`app/client`)
- Shared local data and settings (`data`, `settings`)
- Alembic-owned persistence with direct metric keys, persisted tokenizer
  sources, and relational benchmark-report summaries and tags

The current public release is `v4.5.1` with backend `3.5.1` and frontend
`2.5.1`. It remains a source-only folder distribution launched with
`start_on_windows.ps1` on Windows.

Backend APIs are mounted under `/api/*`. Frontend code uses relative `/api`
requests. Angular CLI proxies those requests during development; FastAPI serves
the built SPA and API from the same origin in production.

## Repository Structure
Source-level structure, with generated and environment-specific folders omitted:

```text
.
├─ assets/
│  ├─ docs/
│  └─ figures/
├─ start_on_windows.ps1
├─ settings/
│  ├─ .env
│  └─ .env.example
├─ app/
│  ├─ client/
│  │  ├─ package.json
│  │  ├─ angular.json
│  │  ├─ public/
│  │  │  └─ tkben-logo.png
│  │  ├─ angular/
│  │  │  ├─ app/
│  │  │  └─ styles.css
│  ├─ server/
│  │  ├─ pyproject.toml
│  │  ├─ app.py
│  │  ├─ api/
│  │  ├─ contracts/
│  │  ├─ configurations/
│  │  ├─ common/
│  │  ├─ services/
│  │  │  └─ benchmark_reports.py
│  │  ├─ repositories/
│  │  │  ├─ datasets.py
│  │  │  ├─ tokenizer_reports.py
│  │  │  ├─ database/
│  │  │  ├─ queries/
│  │  │  └─ schemas/
│  │  └─ migrations/
│  │     └─ versions/0001_current_schema.py
│  ├─ scripts/
│  └─ tests/
├─ data/
└─ LICENSE
```

## Application Entry Points
- Backend app factory/module:
  - `server.app:create_app` constructs the FastAPI app and registers API and frontend routes.
  - `server.app:app` is the canonical ASGI entry point.
- Frontend entry:
  - `app/client/angular/main.ts`
- Frontend routing root:
  - `app/client/angular/app/app.routes.ts`
- Frontend shell:
  - `app/client/angular/app/components/app-shell.component.ts` provides the branded header, primary route tabs, and Settings route action. Hugging Face key management is owned by the canonical Settings → Keys section and reuses the existing `/api/keys` service.
- Frontend data and interaction helpers:
  - Signal stores under `app/client/angular/app/core/state/` own catalog loading,
    report state, polling, and in-memory UI state; only `BenchmarkStore` owns
    persisted dashboard preferences and report-scoped baseline preferences.
  - Pure normalization helpers under `app/client/angular/app/core/utils/` own dataset and chart payload shaping.
- Windows launcher:
  - `start_on_windows.ps1` is the single user-facing root entry point for the combined launch and maintenance menu.

Startup loads `settings/.env` bootstrap values and the database-only
`<TKBEN_DATA_DIR>/database-config.json` before opening the selected database.
Typed Pydantic models own application defaults; the active database stores
validated sparse overrides and dashboard UI preferences in
`application_runtime_settings`. The Settings page exposes application settings
and a Database section. A database that is
unavailable at startup leaves the app in a limited repair mode so the Database
section remains reachable.

## Reporting Service Boundaries
- `server.services.TokenizersService` owns Hugging Face discovery, catalog,
  download, and custom-tokenizer workflows. Custom tokenizer identity is
  stored in the database and its canonical artifact is stored as persistent
  application data under `<TKBEN_DATA_DIR>/sources/tokenizers`; downloaded
  datasets use the corresponding persistent `sources/datasets` location.
- Disposable tooling and runtime caches are owned by the repository-wide
  `runtimes/cache` root and are not the storage location for datasets or
  tokenizer artifacts.
- `server.services.TokenizerReportingService` owns tokenizer metadata, vocabulary analysis, report generation, and report retrieval.
- `server.services.BenchmarkService` owns benchmark admission, execution, and runtime result construction.
- `server.services.BenchmarkReportService` owns benchmark report contract validation, persistence orchestration, and response normalization.
- `server.repositories.BenchmarkRepository` owns projected report tags and the
  dedicated tag update transaction; tags are relational metadata outside the
  immutable benchmark detail payload.
- `server.repositories.DatasetRepository` owns dataset, analysis-session, metric, and histogram persistence.
- `server.repositories.TokenizerReportRepository` owns tokenizer report and vocabulary persistence; `TokenizerRepository` owns tokenizer identity/catalog storage.
- `server.services.dataset_statistics` owns the focused `LengthStatistics` and `HistogramBuilder` components used by dataset analysis.
- `server.services.ManagedJobService`, exposed to API handlers through `ManagedJobHttpAdapter`, centralizes optional job conflict checks, start-up, and initial status validation. Dataset validation, benchmark runs, and tokenizer reports set no `conflict_detail`, so any number run concurrently with independent progress and cancellation; the frontend result panels show the report of the job that completed most recently. Downloads and uploads still reject a second concurrent job with HTTP 409.
- There are no legacy service aliases, registry-only tokenizer rows, or
  compatibility forwarding methods between these boundaries.

## High-Level Architecture

The application separates external provider and filesystem I/O from relational
persistence. Hugging Face and PDF export are service-side integrations; SQLite
or PostgreSQL is reached only through repositories and SQLAlchemy.

```mermaid
flowchart LR
    User[User] --> SPA[Angular SPA]
    SPA --> Stores[Angular signal stores]
    Stores --> Clients[Angular API clients]
    Clients --> FastAPI[FastAPI]
    FastAPI --> Routers[API routers]
    Routers --> Contracts[Request and response contracts]
    Routers --> Services[Application services]
    Services --> Jobs[Managed jobs]
    Services --> Repositories[Repositories]
    Repositories --> ORM[SQLAlchemy ORM]
    ORM --> Relational[(SQLite/PostgreSQL)]
    Services --> Cache[(Disposable tooling cache\nruntimes/cache)]
    Services --> Sources[(Persistent datasets and tokenizer artifacts\n<TKBEN_DATA_DIR>/sources)]
    Services --> HF[Hugging Face provider I/O]
    Services --> PDF[PDF export]
```

## Runtime Interaction Topology
- Local webapp mode:
  - Browser -> FastAPI (`FASTAPI_HOST:FASTAPI_PORT`) for the built SPA and
    `/api` requests on one origin.
- Manual development mode:
  - Browser -> Angular CLI (`UI_HOST:UI_PORT`) -> proxied `/api` -> FastAPI
    (`FASTAPI_HOST:FASTAPI_PORT`).
- Startup readiness:
  - The launcher starts one FastAPI process, waits for `/api/health` and the
    SPA root to respond, opens the same-origin URL, and supervises the process
    until stop or exit.
  - Angular renders a startup gate while polling /api/health. AppShell,
    routing content, and settings loading begin only after the backend reports
    status ok.
  - The launcher uses the canonical backend environment at `app/server/.venv`,
    records backend lockfile readiness, installs the frontend tree with
    `npm ci` only when its dependency stamp is stale, and builds the frontend
    only when its deterministic production-input stamp is missing or stale. It
    checks the backend port before setup and immediately before startup;
    `UI_PORT` is used by manual development only and does not block production
    launch. Ctrl+C requests graceful FastAPI shutdown, with bounded cleanup
    limited to the process tree the launcher started.
