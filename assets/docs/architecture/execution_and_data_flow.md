# Execution and Data Flow
Last updated: 2026-08-29

## Layered Architecture

The backend uses a one-way ownership flow:

`API router -> contract model -> operational service -> repository -> database/filesystem/provider`

API routers translate HTTP concerns and delegate work. Contract models define
the request and response boundary. Services own business rules and
orchestration. Repositories own persistence and query details. Configuration,
common catalogs, and managed-job infrastructure are shared inward-facing
dependencies rather than a second domain layer.

## Key Module Responsibilities

- `server/api/*`
  - HTTP routing, dependency construction, status codes, multipart handling,
    and HTTP error mapping. Routers do not contain persistence logic.
- `server/contracts/*`
  - Pydantic request, response, settings, and report contracts shared at the
    API and service boundary. These modules do not import API, service, or
    repository implementations.
- `server/configurations/*`
  - Environment loading, structured settings, startup configuration, and
    database configuration. One process-level `ServerSettings` snapshot is
    resolved for application startup; structured JSON contains content
    catalogs/jobs only and cannot override database settings.
- `server/services/*`
  - Operational business logic, validation, orchestration, and report
    workflows. `services/benchmark_reports.py` owns benchmark report
    persistence orchestration.
- `server/repositories/*`
  - Dataset, tokenizer-report, benchmark-report, database, query, and schema
    persistence boundaries. Repositories do not call API or service modules.
- `server/common/*`
  - Shared metric catalogs, benchmark metric definitions, logging, and small
    cross-cutting helpers with no feature-service ownership.
- `app/client/angular/app/core/api/*`
  - Typed HTTP clients and the canonical API models used by signal stores.
- `app/client/angular/app/core/state/*`
  - Signal-based catalog, report, persistence, and modal state. Page components
    consume state and rendering helpers rather than reconstructing API rules.

The current import boundary is checked by
`tests/unit/server/test_architecture_boundaries.py`.

## Configuration and schema flow

`create_app()` resolves the process-level settings snapshot once and places it
on application state. Lifespan startup, database initialization, repositories,
and services consume that snapshot rather than independently reloading
configuration. `.env` owns operational values; `configurations.json` owns
datasets, tokenizers, benchmarks, and jobs. PostgreSQL uses the fixed
`postgresql+psycopg` driver when external mode is selected.

Alembic is the only schema authority. Empty databases upgrade through the
tracked graph, while non-empty databases without an Alembic version row,
unknown revisions, multiple heads, or ahead-of-application revisions fail
explicitly. Revision `0003_canonical_state_cleanup` removes incompatible
report rows and normalizes direct metric keys and persisted tokenizer sources.

## Dependency Maps

The implemented backend dependency direction is:

```mermaid
flowchart LR
    API[api routers] --> Contracts[contracts]
    API --> Services[services]
    API --> Config[configurations]
    Services --> Contracts
    Services --> Repositories[repositories]
    Services --> Config
    Services --> Common[common]
    Repositories --> Database[repositories/database]
    Repositories --> Config
    Repositories --> Common
    Database --> Schemas[repositories/schemas]
    Contracts --> Common
```

The target architecture for feature work is the same direction, with shared
contracts and common catalogs kept below the feature services:

```mermaid
flowchart TD
    API[API routers] --> Contracts[Contract models]
    API --> Services[Operational services]
    API --> Config[Configuration]
    Services --> Contracts
    Services --> Repositories[Repositories]
    Services --> Config
    Services --> Common[Common catalogs and helpers]
    Repositories --> Database[Database adapters and schemas]
    Repositories --> Config
    Repositories --> Common
    Database --> ORM[SQLAlchemy entities]
```

Forbidden directions include API modules importing repositories directly,
contracts importing services or persistence, repositories importing services,
and services importing API modules. The removed `server.domain` and
`server.repositories.serialization` paths are not compatibility namespaces;
new code must use the ownership paths above.

## Benchmark Admission and Execution

`POST /api/benchmarks/run` is a thin orchestration endpoint. It validates the
HTTP contract, calls `BenchmarkService.prepare_run()` in a worker thread, maps
known admission failures to HTTP 400, and starts the managed benchmark job.
Preparation performs the rules that need the configured repositories: it
rejects an empty tokenizer or dataset selection, checks persisted tokenizer
identity, source, and canonical artifact availability, checks that the selected
dataset has ready documents, and returns a normalized payload. The
execution mixin retains a defensive ready-document count check at the actual
run boundary.

This keeps admission failures synchronous and deterministic while ensuring
benchmark execution remains off the async request handler.

```mermaid
sequenceDiagram
    participant UI as Angular store
    participant API as POST /benchmarks/run
    participant BS as BenchmarkService
    participant Repo as Dataset/Tokenizer repositories
    participant Jobs as JobManager
    participant Runner as BenchmarkJobService

    UI->>API: normalized BenchmarkRunRequest
    API->>BS: prepare_run(payload) in worker thread
    BS->>Repo: resolve and validate selected assets
    Repo-->>BS: ready dataset and tokenizer state
    BS-->>API: normalized payload
    API->>Jobs: start managed benchmark job
    Jobs->>Runner: execute normalized payload
    API-->>UI: job_id and initial status
```

## Catalog and Report Flows

Catalog reads use synchronous repository calls inside worker-thread service
operations when invoked from async endpoints. The service applies business
filters and returns contract-shaped data; the router owns only HTTP response
construction.

```mermaid
sequenceDiagram
    participant UI as Angular signal store
    participant API as Catalog/report router
    participant Service as Operational service
    participant Repo as Repository
    participant DB as SQLAlchemy session

    UI->>API: list or report request
    API->>Service: validated contract and filters
    Service->>Repo: synchronous query in worker thread
    Repo->>DB: select summary or report data
    DB-->>Repo: persistence rows
    Repo-->>Service: repository records
    Service-->>API: response contract
    API-->>UI: typed response
```

Benchmark report list, load, save, and delete operations belong to
`BenchmarkReportService`, using `BenchmarkReportRepository` for persistence.
`BenchmarkService` prepares and executes benchmarks but does not own report
storage. Tokenizer report storage belongs to
`TokenizerReportRepository`; dataset materialization belongs to
`DatasetRepository`.

```mermaid
sequenceDiagram
    participant API as Benchmark report router
    participant ReportService as BenchmarkReportService
    participant ReportRepo as BenchmarkReportRepository
    participant DB as benchmark_report table

    API->>ReportService: list/load/save/delete request
    ReportService->>ReportRepo: repository operation
    ReportRepo->>DB: summary query or transaction over owned columns/details
    DB-->>ReportRepo: rows or affected count
    ReportRepo-->>ReportService: persistence result
    ReportService-->>API: reconstructed response or not-found result
```

## Managed Jobs

Long-running download, analysis, benchmark, and export operations run through
`JobManager`. Endpoints return after job creation; polling reads in-memory
status, and cancellation sets a cooperative cancellation event consumed by the
runner. Initialization failures and conflicts are translated by the managed
job HTTP adapter rather than leaking worker exceptions.

```mermaid
sequenceDiagram
    participant UI as Angular store
    participant API as Job endpoint
    participant JM as JobManager
    participant Worker as Background worker

    UI->>API: start operation
    API->>JM: start(kind, callable)
    JM->>Worker: run callable
    API-->>UI: job_id and initial status
    loop until terminal state
        UI->>API: GET /jobs/{job_id}
        API->>JM: read status
        JM-->>API: progress/status/error
        API-->>UI: typed job status
    end
    UI->>API: POST /jobs/{job_id}/cancel
    API->>JM: request cooperative cancellation
    Worker-->>JM: cancelled or completed
```

## Benchmark Persistence Flow

The benchmark engine produces observations; result construction and report
materialization are separate from report persistence. The job service owns the
managed execution lifecycle, `BenchmarkService` owns benchmark orchestration,
and `BenchmarkReportService` owns the final report transaction.

```mermaid
sequenceDiagram
    participant Job as BenchmarkJobService
    participant Bench as BenchmarkService
    participant Adapter as UniversalTokenizerAdapter
    participant Spool as BenchmarkTextSpool
    participant Engine as benchmark_engine
    participant Builder as BenchmarkResultBuilder
    participant Repo as BenchmarkRepository
    participant Report as BenchmarkReportService
    participant DB as benchmark_report table

    Job->>Bench: run_benchmarks(normalized inputs)
    Bench->>Adapter: load selected tokenizer
    Bench->>Spool: prepare benchmark text
    Bench->>Engine: warmup and timed trials
    Engine-->>Bench: per-batch observations
    Bench->>Builder: build schema-3/report-5 result
    Builder-->>Bench: immutable report payload
    Bench-->>Job: benchmark result
    Job->>Report: save report
    Report->>Repo: persist relational summary and JSON detail snapshot
    Repo->>DB: transaction
    DB-->>Repo: committed report id
    Repo-->>Report: persisted report
    Report-->>Job: report id
```

## Async and Sync Constraints

- FastAPI endpoints are mostly `async def` orchestrators.
- Blocking repository calls, provider access, and CPU-heavy preparation are
  offloaded with `asyncio.to_thread(...)` or run inside `JobManager` workers.
- Job polling and cancellation are short synchronous reads/writes over
  in-memory managed-job state.
- Repository and database operations use synchronous SQLAlchemy sessions.
- Hugging Face discovery failures propagate through the service and become a
  sanitized HTTP 500; an outage is not represented as a successful empty
  catalog.
- The Angular stores debounce filters and use request sequencing or
  `switchMap` so stale catalog/report responses cannot replace newer state.

Async handlers must not execute CPU-heavy work or blocking I/O inline.
