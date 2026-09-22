# Project Overview
Last updated: 2026-09-22

## Purpose
This file is the root index for `assets/docs`. Read it first to find the narrowest topic file for the task at hand.

The current codebase uses schema-3/report-5 benchmark reports, Alembic revision
`0005_managed_job_lifecycle`, typed frontend catalog controls, persisted
tokenizer source/artifact state, and a `BenchmarkStore` that owns dashboard
preferences and report-scoped baseline selections. Reports own relational
summary fields and tags while JSON stores immutable detail fields; incompatible
historical rows are rejected or purged rather than silently adapted. Managed
job lifecycle metadata is durable, with interrupted work reconciled as failed
after restart rather than automatically resumed.

The latest published release is `v4.4.0` with backend package `3.4.0` and
frontend package `2.4.0`. Configuration ownership is canonical: `settings/.env`
and the process environment own startup/infrastructure values, typed backend
models own application defaults, and sparse user overrides live under
`<TKBEN_DATA_DIR>/runtime-settings.json`. The environment bootstrap runs before
configuration or database imports; runtime overrides are optional and are not
part of startup configuration.

## How To Navigate
1. Start with this file only.
2. Read project_status_ledger.md when the task depends on current
   implementation, validation, blockers, open issues, or revalidation debt.
3. Identify the topic area that matches the task.
4. Open the smallest leaf file that answers the question.
5. Open sibling files only when the task clearly crosses topic boundaries.
6. Do not read the entire tree unless the task explicitly requires broad context.

## Naming Rules
- All documentation files and folders under `assets/docs` use lower-case names.
- Topic folders group related leaf files by subject.
- Root-level files are reserved for entry points and top-level guidance.

## Documentation Ontology
### Root
- `project_index.md`
  - Entry point and index for the full documentation tree.

- [project_status_ledger.md](project_status_ledger.md)
  - Canonical current operational status catalog: component statuses,
    validation evidence, blockers, open issues, validation debt, and resolved
    findings. Update it when implementation, validation evidence, or current
    project risks change; detailed reports remain in their topic documents.

### Architecture
- `architecture/architecture_review.md`
  - Implementation-derived architecture findings, target state, remediation status, and risks.
- `architecture/system_overview.md`
  - Repository layout, entry points, and high-level runtime interaction topology.
- `architecture/backend_api.md`
  - API surface, catalog filter parameters, and response contracts.
- `architecture/execution_and_data_flow.md`
  - Layered backend flow, catalog filtering, module responsibilities, and async/sync behavior.
- `architecture/persistence.md`
  - Data storage model and persisted artifact locations.
- `architecture/benchmark_contract.md`
  - Benchmark payload, runtime metadata, and report contract notes.

### Coding
- `coding/python.md`
  - Python runtime, typing, validation, async, and structure guidance.
- `coding/typescript.md`
  - Frontend TypeScript, typed catalog state, UI, and styling guidance.
- `coding/testing_and_quality.md`
  - Testing, linting, and cross-language quality gates.

### Runtime
- `runtime/modes.md`
  - Supported runtime modes and operational differences.
- `runtime/startup.md`
  - Launcher commands, startup procedures, readiness checks, and first-launch
    behavior.
- `runtime/configuration.md`
  - Environment variables, typed runtime defaults, and persisted user overrides.
- `runtime/deployment.md`
  - Dependencies and local distribution notes.
- `runtime/release.md`
  - Source-release preparation, validation, branch synchronization, and publication.

### UI
- `ui/design_tokens.md`
  - Typography, layout, spacing, and color system.
- `ui/components_and_patterns.md`
  - Branded navigation, catalog controls, forms, overlays, and feedback states.
- `ui/experience.md`
  - Page structure, UX rules, responsiveness, accessibility, and design principles.
- `ui/ui_standards.md`
  - Practical UI implementation standards for spacing, typography, colors, components, and responsive polish.
- `ui/benchmark_dashboard.md`
  - Normalized cross-benchmark dashboard contract, customization, persistence, and version policy.

## Reading Order
1. Read this root index.
2. Open the smallest leaf file that covers the current question.
3. Expand to adjacent files only when the task crosses topic boundaries.
4. Return here when you need to jump to a different topic branch.

## Context Rules
- Treat project_status_ledger.md as the canonical source for current
  operational project status. Update the affected component, issue, evidence,
  and validation-debt entries after meaningful implementation or validation
  work; do not use historical reports as proof of the current state without
  rechecking the checkout.
- Read documentation files only when required by the current task.
- Defer reading until the task proves the file is needed.
- Keep all affected documents updated whenever implementation changes alter behavior.
- Always include a `Last updated: YYYY-MM-DD` line when modifying a document.
- Do not read all `SKILL.md` files by default.
- Pre-select files to read based on folder structure and user intent before opening documentation.

## Environment Rules
- Windows is the default operating environment for this project.
- Document and support both PowerShell and CMD usage patterns where commands differ.
- Keep environment guidance aligned with `start_on_windows.ps1`, its root-level helper scripts, and `app/tests/run_tests.bat`.
- Update this section when new environment-specific constraints or solutions are introduced.
