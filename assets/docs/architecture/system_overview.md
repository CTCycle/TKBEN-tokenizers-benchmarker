# System Overview
Last updated: 2026-08-01

## System Summary
TKBEN is a tokenizer benchmarking platform with:
- FastAPI backend (`app/server`)
- React + Vite frontend (`app/client`)
- Shared local resources and settings (`app/resources`, `settings`)

Backend APIs are mounted under `/api/*`. Frontend calls `/api` and relies on the Vite proxy in dev and preview modes.

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
│  ├─ .env.example
│  └─ configurations.json
├─ app/
│  ├─ client/
│  │  ├─ package.json
│  │  ├─ vite.config.ts
│  │  ├─ public/
│  │  │  └─ tkben-logo.png
│  │  ├─ src/
│  │  └─ package-lock.json
│  ├─ server/
│  │  ├─ pyproject.toml
│  │  ├─ uv.lock
│  │  ├─ app.py
│  │  ├─ api/
│  │  ├─ configurations/
│  │  ├─ domain/
│  │  ├─ services/
│  │  ├─ repositories/
│  │  └─ common/
│  ├─ scripts/
│  ├─ tests/
│  └─ resources/
└─ LICENSE
```

## Application Entry Points
- Backend app factory/module:
  - `server.app:create_app` constructs the FastAPI app and registers API and frontend routes.
  - `server.app:app` is the canonical ASGI entry point.
- Frontend entry:
  - `app/client/src/main.tsx`
- Frontend routing root:
  - `app/client/src/App.tsx`
- Frontend shell:
  - `app/client/src/components/AppShell.tsx` provides the branded header, primary route tabs, and Hugging Face key manager control.
- Windows launcher:
  - `start_on_windows.ps1` is the single user-facing root entry point for the combined launch and maintenance menu.

## Reporting Service Boundaries
- `server.services.TokenizersService` owns Hugging Face discovery, catalog, download, cache, persistence, and custom-tokenizer workflows.
- `server.services.TokenizerReportingService` owns tokenizer metadata, vocabulary analysis, report generation, and report retrieval.
- `server.services.dataset_statistics` owns the focused `LengthStatistics` and `HistogramBuilder` components used by dataset analysis.
- `server.services.ManagedJobService`, exposed to API handlers through `ManagedJobHttpAdapter`, centralizes job conflict checks, start-up, and initial status validation.
- There are no legacy service aliases or compatibility forwarding methods between these boundaries.

## Runtime Interaction Topology
- Local webapp mode:
  - Browser -> Vite preview (`UI_HOST:UI_PORT`) -> proxied `/api` -> FastAPI (`FASTAPI_HOST:FASTAPI_PORT`)
- The launcher uses the canonical backend environment at `app/server/.venv` and lockfile at `app/server/uv.lock`, builds the frontend, starts both services, and opens the configured UI URL.
