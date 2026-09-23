# Project Status Ledger
Last updated: 2026-09-23

This document is the canonical catalog of the current operational state of
TKBEN. It summarizes what is implemented, what has been observed working,
what has been formally validated, what remains partial or unvalidated, and
what is blocked by an external or environment-dependent gate. Detailed
architecture decisions, implementation plans, test code, and validation
records remain in their dedicated documents; this ledger links to them rather
than copying their narratives.

The bootstrap baseline is checkout e318852 (tag v4.4.0), with local develop
and main aligned. The evidence snapshot below was collected on 2026-09-20
after restoring the repository-managed Python and frontend dependencies.
V-20260921 adds launcher/configuration evidence on local develop revision
4a0e77f with the scoped working-tree changes described below. Statuses describe
this checkout and must be refreshed after behavioral changes.

V-20260922 adds gate-closure evidence from the develop checkout at
`e864197c33b1e6a157fa5db33f6781df848c2d20`. The launcher contract coverage,
privilege-isolated harness, and permission-denied evidence are committed; the
record links to their detailed disposable-environment evidence.

V-20260922 also adds T1-05 catalogue-filtering and race evidence at validated
implementation revision `993e52e8f0617d4bf98a60d62b4c5d5588541218`. The
populated Dataset and Tokenizer matrices, live catalogue races, deterministic
discovery race, regression gates, and cleanup are recorded in the [T1-05 QA
record](../../assets/QA/tkben-t1-05-catalog-filtering-races-20260922.md).

V-20260923 adds the controlled T2-03 dataset quality, structure, and compression
dashboard evidence at implementation revision `de1aaee6a229bfb7ff64b163761fc78c144efaf8`.
The focused Chrome E2E, metric contract checks, rendered reload, and cleanup are
recorded in the [T2-03 QA record](../../assets/QA/tkben-t2-03-dataset-quality-structure-compression-20260923.md).

V-20260923 also closes T2-05 against application revision
`5cf39a0a49e7f51e57438520901bda3551e96600`. A local 1,207-entry custom
tokenizer report persisted and rendered across reload, with complete API and UI
vocabulary paging evidence in the [T2-05 QA
record](../../assets/QA/tkben-t2-05-tokenizer-report-vocabulary-20260923.md).

V-20260923 closes T2-06 at validation implementation revision
`f736f9386a119df0bec330cd7a6e7212fd6a61e7`. The opt-in local browser E2E
created a two-document benchmark through the populated wizard, rendered the
persisted report, and reloaded the same report ID. The focused Cross Benchmark
dashboard and benchmark API E2E set passed 9/9; details and screenshot are in
the [T2-06 QA record](../../assets/QA/tkben-t2-06-cross-benchmark-wizard-20260923.md).

V-20260923 closes T3-01 and T5-04, and validates the supported configuration
controls in T3-02 at local `develop` based on `d412b61`. The controlled Windows
campaign records three comparable 1,000-document reports, non-default option
effects and saved configuration, plus 10,000-document streaming progress,
cancellation, resource samples, and a rendered rerun. T3-02 remains PARTIAL
because parallelism is accepted and saved but is not applied to tokenizer
execution. See the [benchmark validation campaign QA
record](../../assets/QA/tkben-benchmark-validation-campaign-20260923/README.md).

V-20260923 closes T2-07 at implementation revision `cc91fec1b7307ae59091dc9e522627d8b69c6924`. The isolated Windows Chrome E2E cancelled a 10,000-document run after API progress reached 20%, observed terminal `cancelled` with no report, then immediately completed and rendered a two-document rerun. The focused frontend and backend gates, lint, and production build passed; see the [T2-07 QA record](../../assets/QA/tkben-t2-07-benchmark-cancellation-20260923.md).

V-20260923 closes T3-03 and T3-04 and validates the populated Cross Benchmark
portion of T5-03. The live Windows Chrome scenario created one- and two-tokenizer
reports through the official launcher, exercised persisted report management
and dashboard preferences, and reviewed the populated view and dialogs at four
viewport sizes. It also found and fixed a single-tokenizer chart label collision
and a baseline selector hydration defect. See the [T3-03/04 Cross Benchmark QA
record](../../assets/QA/tkben-t3-03-04-cross-benchmark-workflows-20260923.md).

The Cross Benchmark workflow component remains PARTIAL because its
parallelism setting is persisted without affecting execution. T5-03 remains
PARTIAL because its Tokenizers responsive and keyboard coverage has not been completed.
Hosted CI run 35875638854 passed for implementation commit
`dc8cae26ea81b1c621c800168d5b3581c9fab04f`; T5-06 remains PARTIAL because
release publication evidence is separate. Hugging Face and PostgreSQL gates
remain BLOCKED; T3-02/05, T4-02, T5-02/03/05, and the remaining
responsive matrix are still open as listed below.

## Maintenance Rules

Future coding and validation agents must:

1. inspect this ledger before substantial implementation or validation work;
2. use it to identify known defects and previously validated behavior;
3. update affected entries after implementation;
4. update validation evidence after meaningful tests;
5. never mark a component VALIDATED without supporting evidence;
6. downgrade a status when a regression is discovered;
7. close or archive an issue only after successful remediation and
   revalidation;
8. avoid creating duplicate issue entries for the same underlying defect;
9. link detailed reports instead of copying large reports into this ledger;
10. keep the ledger synchronized with the actual repository state.

VALIDATED is an evidence claim, not a synonym for "the code exists" or "a
unit test exists." When the evidence is narrower than the component scope,
use WORKING, PARTIAL, or UNVALIDATED as appropriate.

## Status Taxonomy

| Status | Meaning |
| --- | --- |
| VALIDATED | Implemented and confirmed through meaningful testing at the stated validation level. |
| WORKING | Believed to work from implementation and limited or indirect evidence, but not fully validated for the stated scope. |
| PARTIAL | Implemented, but incomplete, degraded, or operationally confirmed for only part of the expected behavior. |
| BROKEN | Known not to work correctly. |
| BLOCKED | Cannot currently be validated or completed because of an external dependency, missing credential, unavailable service, hardware constraint, or similar blocker. |
| UNVALIDATED | Implementation exists, but available evidence is insufficient to claim that it works. |
| NOT_IMPLEMENTED | The expected capability is absent from the current repository. |
| DEPRECATED | Intentionally retained for compatibility or scheduled for removal. |

Validation levels used below are None, unit, integration, E2E, and manual.
Combined levels mean that all named layers contributed evidence. An em dash
means that a field does not apply.

## Current Truth at a Glance

- Local webapp startup, the primary API contract, SQLite/Alembic persistence,
  dataset analysis, custom-tokenizer upload/deletion, benchmark report
  round-trip, frontend quality gates, and the exercised UI routes are
  VALIDATED in the current bootstrap evidence.
- Durable job visibility and restart reconciliation are VALIDATED at unit and
  Linux E2E levels. The Windows portable bootstrap and launcher are VALIDATED
  on the exercised Windows host, including the privilege-isolated denied
  process-termination branch.
- Two Windows launcher defects were reproduced and corrected during validation:
  script-block invocation mishandled ordinary stderr, and Kill All missed a
  quoted npm preview process while also matching nested process roots.
- Hugging Face live discovery/report flows and PostgreSQL runtime equivalence
  are BLOCKED by conditional external-provider/database gates.
- Broader Cross Benchmark report management and dashboard customization, PDF
  export, hosted CI, and the documented responsive visual matrix remain
  validation debt.
- Containerized deployment and binary packaging are explicitly
  NOT_IMPLEMENTED; source-only local distribution is the supported release
  model.

## Current Component Ledger

### Architecture, runtime, and configuration

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture.canonical-ownership | VALIDATED | API, contracts, configuration, services, repositories, and frontend state ownership boundaries. | V-20260920: 323 backend unit tests passed, including the architecture-boundary contract; current architecture docs describe the same ownership graph. | Remaining canonicalization follow-ups are tracked as ISSUE-003; no current regression observed. | — | 2026-09-20 | unit | [architecture review](architecture/architecture_review.md), [canonical-source remediation](architecture/canonical_source_remediation.md), [boundary test](../../app/tests/unit/server/test_architecture_boundaries.py) | Revalidate the boundary test after ownership or schema changes. |
| runtime.startup.local-webapp | VALIDATED | FastAPI readiness, Angular production preview, local API proxy, restart, and major-route loading. | V-20260922: Windows clean bootstrap returned backend health 200 and the browser rendered Dataset; Linux manual startup returned health 200, proxied `/api/datasets/list`, rendered Dataset/Tokenizers/Cross Benchmark/Settings, and recovered after backend restart. | This validates the exercised startup routes; populated report dashboards and responsive layouts remain separate. | — | 2026-09-22 | E2E + manual | [startup](runtime/startup.md), [runtime modes](runtime/modes.md), [system overview](architecture/system_overview.md), [gate-closure QA record](../../assets/QA/tkben-partial-gates-20260922.md) | Revalidate after launcher or readiness changes. |
| runtime.windows-launcher | VALIDATED | `start_on_windows.ps1` dependency bootstrap, pinned runtimes, port-conflict handling, stamped dependency/build reuse, readiness, maintenance menu, and process cleanup. | V-20260922: clean bootstrap and warm reuse passed; all 13 menu routes were exercised, including both install profiles, expected `Update` refusal on `develop`, destructive decline/approval, and Kill All. Malformed/missing stamps, stale frontend build, backend-only stale state, invalid port, redirected conflict, quoted preview cleanup, and a port reacquisition between checks were exercised. The launcher contract suite passes 19/19. A safety-gated disposable Windows run used a real elevated synthetic listener; approved termination produced localized `Accesso negato`, the listener remained on the configured port, and no service started. | The repository-standard runner has separate host ACL failures in its Ruff, BasedPyright, and Python phases; these do not affect the launcher evidence and are recorded in the QA record. | — | 2026-09-22 | unit + integration + manual | [startup](runtime/startup.md), [deployment](runtime/deployment.md), [launcher QA record](../../assets/QA/tkben-partial-gates-20260922.md), [contract tests](../../app/tests/unit/server/test_windows_launcher_contract.py), [permission harness](../../app/tests/integration/windows/test_launcher_port_permissions.ps1) | Revalidate after launcher or readiness changes. |
| configuration.runtime-settings | WORKING | Typed runtime defaults, sparse persisted overrides, revision checks, reset behavior, and Settings API/page. | V-20260922 at `41f265a`: all 504 backend unit tests passed with an isolated cache under the canonical cache root; installed-Chrome Settings E2E passed 2/2; in-app browser save/reload and two backend restarts preserved settings and then reset to defaults; downstream new-work effects and revision conflict passed. The repository runner's Python collection remains blocked by access denied on a generated cache subtree. | The T1-02 slice is closed; the broader component remains WORKING because this gate does not promote component status beyond its separately tracked lifecycle scope. | — | 2026-09-22 | unit + E2E + manual | [configuration](runtime/configuration.md), [backend API](architecture/backend_api.md), [Settings E2E](../../app/tests/e2e/test_settings_ui.py), [T1-02 lifecycle QA record](../../assets/QA/tkben-t1-02-runtime-settings-20260922.md), [earlier T1-02 boundary QA](../../assets/QA/tkben-t1-02-settings-boundary-20260922.md), [Tier 1 QA record](../../assets/QA/tkben-tier1-20260921.md) | Keep the broader component at WORKING until its separate lifecycle scope is promoted. |
| runtime.managed-job-lifecycle | VALIDATED | Start, poll, complete, fail, cancel, persist, and reconcile managed jobs across application restart. | V-20260922: 212 backend unit tests passed. Linux restart E2E kept completed upload job `c079b263` addressable, retained its 25,000-document dataset, and reconciled active analysis job `3fa194eb` to `failed` with an explicit restart-interruption error. | Job runners are not checkpointed or resumed; pending/running jobs fail deterministically on restart by design. | — | 2026-09-22 | unit + E2E | [execution and data flow](architecture/execution_and_data_flow.md), [persistence](architecture/persistence.md), [job tests](../../app/tests/unit/server/services/test_jobs_manager.py), [gate-closure QA record](../../assets/QA/tkben-partial-gates-20260922.md) | Keep non-resumption explicit unless runner checkpointing and idempotency are designed. |

### Backend, persistence, and data workflows

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| backend.api-contracts | VALIDATED | /api/* routing, request/response contracts, health, settings, datasets, tokenizers, benchmarks, jobs, keys, and exports. | V-20260922: 212 backend unit tests passed; after a real restart, `GET /api/jobs/3fa194eb` returned the same job as failed instead of 404, and terminal cancellation conflict behavior is covered by route tests. | Provider-dependent paths are separately bounded below. | — | 2026-09-22 | unit + integration + E2E | [backend API](architecture/backend_api.md), [OpenAPI test](../../app/tests/unit/server/test_openapi_schema.py), [gate-closure QA record](../../assets/QA/tkben-partial-gates-20260922.md) | Re-run focused API checks when a contract or route changes. |
| persistence.sqlite-alembic | VALIDATED | Embedded SQLite initialization, migration locking, current schema, report/tag persistence, rollback, and managed-job lifecycle rows. | V-20260922: migration tests cover SQLite upgrade from 0004 to 0005; Windows clean bootstrap and Linux startup reached Alembic head `0005_managed_job_lifecycle`; the restart E2E preserved the active/completed job rows and test dataset. | The application intentionally rejects incompatible or unversioned non-empty databases rather than adapting them silently. | — | 2026-09-22 | integration + unit + E2E | [persistence](architecture/persistence.md), [database initialization](../../app/tests/unit/server/repositories/test_database_initialization.py), [migration tests](../../app/tests/unit/server/repositories/test_database_migrations.py), [persistence tests](../../app/tests/unit/server/repositories/test_persistence_contract.py), [gate-closure QA record](../../assets/QA/tkben-partial-gates-20260922.md) | Re-run migration and persistence contracts for schema changes. |
| persistence.postgresql-runtime | BLOCKED | External PostgreSQL initialization, migration locking, concurrency, and runtime equivalence with SQLite. | Unit coverage exercises initializer branches, but no disposable PostgreSQL target was available in V-20260920. | SQLite success must not be promoted to PostgreSQL equivalence. | Requires a disposable PostgreSQL target and configured connection/credentials. | — | None | [persistence](architecture/persistence.md), [configuration](runtime/configuration.md) | Provision a disposable target and run the documented PostgreSQL integration validation. |
| data.dataset-import-and-analysis | VALIDATED | CSV upload, ready-state persistence, list visibility, missing-dataset handling, asynchronous analysis, histograms, and statistics. | V-20260920: all dataset API E2E cases passed, including upload and analysis of a small CSV; generated records were removed and list state returned to empty. | Large files and remote dataset downloads remain dependent on local disk/network conditions. | — | 2026-09-20 | E2E + unit | [backend API](architecture/backend_api.md), [persistence](architecture/persistence.md), [dataset E2E](../../app/tests/e2e/test_datasets_api.py) | Revalidate download and large-streaming paths when those surfaces change. |
| data.custom-tokenizer-storage | VALIDATED | Custom tokenizer JSON compatibility, canonical artifact storage, catalog visibility, and deletion. | V-20260920: valid upload/deletion and invalid-input API E2E cases passed; generated tokenizer artifacts were removed afterward. | Report generation for a persisted tokenizer was not included in the current run. | — | 2026-09-20 | E2E + unit | [system overview](architecture/system_overview.md), [benchmark contract](architecture/benchmark_contract.md), [tokenizer E2E](../../app/tests/e2e/test_tokenizers_api.py) | Add the custom-tokenizer report/vocabulary path to the next live validation. |
| integration.huggingface-discovery-and-download | BLOCKED | Live Hugging Face discovery, gated/public access, metadata filtering, download, and provider-backed tokenizer report flow. | The current tokenizer E2E run passed local cases but skipped the two provider-discovery cases and the report-flow case because their opt-in gates were not enabled. | No current live claim is made for provider availability, credential handling, or remote repository compatibility. | Requires a usable provider key/network and explicit opt-in live tests. | — | None | [backend API](architecture/backend_api.md), [testing and quality](coding/testing_and_quality.md), [tokenizer E2E](../../app/tests/e2e/test_tokenizers_api.py) | Run with an approved provider credential and record pass/skip/failure separately. |
| benchmark.execution-and-reporting | PARTIAL | Benchmark admission, custom-tokenizer execution, schema-3/report-5 report persistence, list/load, metadata, and physical deletion. | V-20260923: T2-07 cancellation and rerun passed; the controlled T3/T5 campaign collected three 1,000-document reports with raw observations and resource samples, verified saved option configuration, and cancelled a 10,000-document run without saving a report before rendering a successful rerun. See the [T2-07 QA record](../../assets/QA/tkben-t2-07-benchmark-cancellation-20260923.md), [T2-06 QA record](../../assets/QA/tkben-t2-06-cross-benchmark-wizard-20260923.md), and [benchmark validation campaign](../../assets/QA/tkben-benchmark-validation-campaign-20260923/README.md). | The parallelism setting is accepted and persisted but does not change the currently serial tokenizer execution. Multi-tokenizer/provider, PDF, and broader performance coverage remain separate. | — | 2026-09-23 | E2E + unit + manual | [benchmark contract](architecture/benchmark_contract.md), [execution and data flow](architecture/execution_and_data_flow.md), [benchmark E2E](../../app/tests/e2e/test_benchmarks_api.py) | Define and implement parallelism semantics before promoting the component; continue separately scoped provider and export validation. |
| benchmark.dashboard-and-pdf-export | WORKING | Normalized report-v5 widgets, visualizations, data tables, dashboard layout, and PDF export contracts. | V-20260923: populated live report rendered at four viewport sizes; baseline deltas, data tables, visualization selection, widget ordering, customization, and reload persistence passed. | Live PDF download/render parity remains unvalidated. | — | 2026-09-23 | unit + E2E + manual | [benchmark dashboard](ui/benchmark_dashboard.md), [T3-03/04 QA record](../../assets/QA/tkben-t3-03-04-cross-benchmark-workflows-20260923.md), [export tests](../../app/tests/unit/test_dashboard_export_service.py), [export route tests](../../app/tests/unit/server/api/test_exports_routes.py) | Download and inspect a populated-report PDF, including visualization overrides and chart/data-table parity. |

### Frontend and user workflows

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ui.route-shell-and-empty-states | VALIDATED | Primary navigation, route loading, dataset/tokenizer empty states, benchmark empty state, and Settings shell. | V-20260920: four routes rendered in the in-app browser; browser console error log was empty. | Current visual check used the normal browser viewport, not the full documented responsive matrix. | — | 2026-09-20 | manual | [experience](ui/experience.md), [components and patterns](ui/components_and_patterns.md), [app-flow E2E](../../app/tests/e2e/test_app_flow.py) | Repeat at the required viewport sizes after layout changes. |
| ui.startup-readiness | VALIDATED | Frontend-first startup screen, asynchronous backend readiness polling, slow-start notice, retryable failure state, and transition into the existing shell. | V-20260921: the canonical launcher opened the frontend before backend health; the in-app browser showed the quiet token stream/benchmark graph while offline, the slow state, the 60-second failure state, and a retry that transitioned into Dataset after backend recovery. Follow-up validation confirmed the rebuilt production preview keeps the artwork moving while connecting and pauses it in the terminal failure state. Focused browser E2E passed 2/2; frontend unit tests 60/60, lint, and production build passed. | The current application has a dark theme only; the graph is intentionally illustrative and not a benchmark measurement. | — | 2026-09-21 | unit + E2E + manual | [startup](runtime/startup.md), [experience](ui/experience.md), [startup E2E](../../app/tests/e2e/test_startup_loading.py) | Revalidate after launcher or readiness changes. |
| ui.dataset-dashboard | WORKING | Dataset selection, validation controls, persisted analysis dashboard, charts, and export action. | V-20260923: T2-03 controlled quality/structure/compression dashboard flow populated and restored after reload; metric unit tests passed 78/78. See the [T2-03 QA record](../../assets/QA/tkben-t2-03-dataset-quality-structure-compression-20260923.md). Earlier T2-02 six-family coverage is recorded in the [T2-02 QA record](../../assets/QA/tkben-t2-02-dataset-metric-families-20260922.md). | Export and malformed optional-payload behavior remain unvalidated. | — | 2026-09-23 | unit + E2E + manual | [experience](ui/experience.md), [dataset E2E](../../app/tests/e2e/test_datasets_api.py) | Validate export and malformed optional-payload handling before promoting the full component scope. |
| ui.settings-page | WORKING | Settings tabs, typed controls, inline validation, persistence, conflict handling, reset, and Keys section navigation. | V-20260922: the rendered Settings route had no overlay or console errors; Chrome browser E2E passed 2/2 with all 16 controls, boundary errors, cross-field recovery, persistence, reload hydration, conflict, reset, and new-operation effects. | Rendered key-management lifecycle remains outside the current slice. | — | 2026-09-22 | unit + E2E + manual | [configuration](runtime/configuration.md), [experience](ui/experience.md), [Settings E2E](../../app/tests/e2e/test_settings_ui.py), [T1-02 closure QA record](../../assets/QA/tkben-t1-02-settings-boundary-20260922.md), [Tier 1 QA record](../../assets/QA/tkben-tier1-20260921.md) | Complete the separate rendered key-management lifecycle before promoting the full page scope. |
| ui.cross-benchmark-workflow | PARTIAL | Benchmark wizard, report manager, baseline selection, clone eligibility/configuration, tags, dashboard customization, data tables, and populated report rendering. | V-20260923: report-manager, dashboard, and four-viewport dialog flows passed; the Chrome run-options flow exercised non-default values and saved configuration, and its options/progress/report screenshots are in the [benchmark validation campaign QA record](../../assets/QA/tkben-benchmark-validation-campaign-20260923/README.md). | Parallelism remains a saved-only setting; Tokenizers responsive and keyboard coverage remains in T5-03; populated-report PDF rendering remains separately tracked. | — | 2026-09-23 | unit + E2E + manual | [benchmark dashboard](ui/benchmark_dashboard.md), [experience](ui/experience.md), [Cross Benchmark E2E](../../app/tests/e2e/test_cross_benchmark_report_workflows.py), [T3-03/04 QA record](../../assets/QA/tkben-t3-03-04-cross-benchmark-workflows-20260923.md), [benchmark validation campaign](../../assets/QA/tkben-benchmark-validation-campaign-20260923/README.md) | Define and implement parallelism semantics; complete separately scoped responsive and PDF validation. |
| ui.tokenizer-report-and-vocabulary | VALIDATED | Tokenizer report generation, vocabulary paging, report dashboard, and vocabulary preview. | V-20260923: a local 1,207-entry custom tokenizer generated and persisted a report; reload restored the same report ID and the populated UI navigated three vocabulary pages. API item IDs were contiguous across offsets 0, 500, and 1,000. Focused Chrome E2E passed 1/1 and 34 report, vocabulary, route, and service unit tests passed. See the [T2-05 QA record](../../assets/QA/tkben-t2-05-tokenizer-report-vocabulary-20260923.md). | — | — | 2026-09-23 | unit + E2E | [experience](ui/experience.md), [backend API](architecture/backend_api.md), [tokenizer E2E](../../app/tests/e2e/test_tokenizers_api.py), [T2-05 QA record](../../assets/QA/tkben-t2-05-tokenizer-report-vocabulary-20260923.md) | Revalidate after tokenizer report or vocabulary paging changes. |

### Distribution and test infrastructure

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deployment.source-only-local | VALIDATED | Supported source-folder distribution, current version alignment, and local production frontend artifact. | Current checkout is tag v4.4.0; backend package is 3.4.0, frontend package is 2.4.0, and production build passed. | GitHub-hosted release publication was not rechecked in this bootstrap. | — | 2026-09-20 | manual + integration | [release procedure](runtime/release.md), [deployment](runtime/deployment.md), [system overview](architecture/system_overview.md) | Recheck remote tag/release and hosted CI before the next publication. |
| deployment.windows-portable-bootstrap | VALIDATED | Automatic Windows Python/Node/uv bootstrap, stamped dependency repair, deterministic build reuse, and local launch path. | V-20260922: clean Windows 11 bootstrap from absent managed runtimes, `.venv`, `node_modules`, build output, stamps, database, and `.env` downloaded Python 3.14.7, Node 22.23.1, and uv 0.12.17; created the environment file, synced locked dependencies, migrated to 0005, built Angular, wrote stamps, and started both services. Warm launch reused all three stamps; stale uv 0.12.16 was replaced by the pin. | Evidence is host-specific; it does not establish every Windows edition or hardware architecture. | — | 2026-09-22 | integration + manual | [runtime modes](runtime/modes.md), [startup](runtime/startup.md), [deployment](runtime/deployment.md), [gate-closure QA record](../../assets/QA/tkben-partial-gates-20260922.md) | Repeat on supported Windows architectures when the portable runtime pins change. |
| deployment.cross-platform-manual | PARTIAL | Manual Linux/macOS startup versus Windows-only automatic bootstrap. | V-20260922: clean manual startup, API proxy, four browser routes, service restart, persistent job reconciliation, and clean shutdown passed in a disposable Ubuntu 26.04 container. | macOS and hosted `ubuntu-latest` were not exercised. The container used Python 3.14.4 while CI pins 3.14.7; Linux success does not claim macOS or hosted-CI validation. | — | 2026-09-22 | E2E + manual | [runtime modes](runtime/modes.md), [deployment](runtime/deployment.md), [gate-closure QA record](../../assets/QA/tkben-partial-gates-20260922.md) | Validate the exact hosted runner and macOS separately before expanding the scope. |
| deployment.containerized | NOT_IMPLEMENTED | Docker or other active container runtime configuration. | [Runtime modes](runtime/modes.md) explicitly records containerized mode as not implemented; no active root container configuration exists. | This is an absent capability, not a current local-app failure. | — | — | None | [runtime modes](runtime/modes.md), [deployment](runtime/deployment.md) | Add a separately scoped deployment design before implementation. |
| deployment.binary-packaging | NOT_IMPLEMENTED | Installer, executable, Tauri, portable binary, or package artifact. | [Release procedure](runtime/release.md) states that releases are source-only and contain no binary packaging workflow. | Source-only distribution is the intended current release model. | — | — | None | [release procedure](runtime/release.md) | Do not add packaging work to a source-only release. |
| test-infrastructure.local-quality-gates | VALIDATED | Backend compile, Ruff, BasedPyright, SQLite initialization, unit tests, OpenAPI smoke, frontend lint, unit tests, and production build. | V-20260923: focused backend services passed 24/24, frontend unit suite passed 62/62 across 14 files, Ruff and lint passed, BasedPyright reported 0 errors and 2,004 warnings, the production build passed, and the opt-in Chrome campaign passed 2/2. Backend pytest emitted one unknown `cache_dir` warning. | Hosted CI, full browser E2E outside this campaign, and provider/database gates remain separate. | — | 2026-09-23 | unit + E2E + integration | [testing and quality](coding/testing_and_quality.md), [CI workflow](../../.github/workflows/ci.yml), [test runner](../../app/tests/run_tests.bat), [benchmark validation campaign](../../assets/QA/tkben-benchmark-validation-campaign-20260923/README.md) | Keep hosted and live-provider gates explicit in future reports. |
| test-infrastructure.hosted-ci-and-release-evidence | PARTIAL | Current hosted CI result and committed detailed QA/release evidence. | Hosted CI run 35875638854 passed for pushed implementation commit `dc8cae26ea81b1c621c800168d5b3581c9fab04f`; both frontend and backend validation jobs succeeded. See the [T3-03/04 QA record](../../assets/QA/tkben-t3-03-04-cross-benchmark-workflows-20260923.md). | Release publication evidence was not checked and cannot be inferred from CI. | Release audit is outside this validation scope. | 2026-09-23 | hosted CI | [CI workflow](../../.github/workflows/ci.yml), [run 35875638854](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/runs/35875638854), [release procedure](runtime/release.md), [T3-03/04 QA record](../../assets/QA/tkben-t3-03-04-cross-benchmark-workflows-20260923.md) | Validate release publication and correlate it with a separately selected release SHA. |
| api.tokenizers.settings-compatibility | DEPRECATED | Legacy GET /api/tokenizers/settings compatibility response. | The API contract and OpenAPI test retain the endpoint as deprecated; new clients use /api/settings. | Compatibility surface should not become a second settings source. | — | 2026-09-20 | unit | [backend API](architecture/backend_api.md), [configuration](runtime/configuration.md) | Remove only after supported clients no longer depend on it and the removal is validated. |

## Open Issues

Severity and component status are separate. The following are actionable
current limitations or architectural follow-ups supported by repository
evidence; none is being mislabeled as a BROKEN component.

| ID | Affected component | Severity | Concise description | Current impact | Reproduction or evidence | Suspected cause | Blocker | Remediation status | Required revalidation | Related documentation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISSUE-002 | deployment.network-auth-boundary | HIGH | Network-hosted deployments require an external authentication boundary before exposing key-management or destructive routes. | A network deployment without that boundary would expose sensitive operational actions beyond the local-app threat model. | The supported deployment documentation states the external-auth requirement; the repository does not provide that boundary. | Deployment scope is intentionally local/source-only. | An external authentication layer is not part of this repository. | Open pre-deployment requirement; not a local-webapp defect. | Before any network-hosted deployment, document and validate authentication, authorization, key protection, and destructive-route controls. | [deployment](runtime/deployment.md#constraints), [configuration](runtime/configuration.md#security-controls) |
| ISSUE-003 | architecture.canonical-ownership | MEDIUM | Several canonicalization follow-ups remain: environment-derived paths, duplicate metric representations, export dictionaries, handwritten frontend contracts, duplicated catalog metadata, and toolchain version declarations. | Future changes can drift across duplicated sources even though current ownership boundaries are explicit. | The remaining-work list is recorded in the canonical-source remediation document. | Historical duplication outside the first cleanup scope. | Broader contract/migration work is required; no current defect was reproduced. | Open planned architecture work. | Revalidate affected contracts, migration behavior, and UI/API parity after each follow-up. | [canonical-source remediation](architecture/canonical_source_remediation.md#remaining-canonicalization-work) |

No other reproducible functional defect was found in the current bootstrap.
Validation gaps below remain validation debt, not hidden issue records.

## Validation Debt

Validation debt identifies important areas with insufficient recent evidence.
It does not assert that the component is broken.

| Component | Current confidence | Missing validation | Priority |
| --- | --- | --- | --- |
| benchmark.dashboard-and-pdf-export | WORKING | Download and inspect a PDF from a populated report, including visualization overrides and chart/data-table parity. | MEDIUM |
| benchmark.execution-and-reporting | PARTIAL | The parallelism field is accepted and persisted but currently does not change serial tokenizer execution; define the intended semantics and implement or remove the setting before promoting the full component. | MEDIUM |
| data.dataset-import-and-analysis | UNTESTED | Validate public dataset download, disk handling, and cleanup against expected external sources. | MEDIUM |
| integration.huggingface-discovery-and-download | BLOCKED | Run the opt-in provider discovery, gated/public access, download, and report-flow tests with approved credentials and network. | HIGH |
| persistence.postgresql-runtime | BLOCKED | Run disposable PostgreSQL migration, concurrency, rollback, and runtime-equivalence checks. | MEDIUM |
| ui.responsive-visual-matrix | PARTIAL | Populated Cross Benchmark dashboard and report/clone/customize dialogs passed viewport bounds and keyboard-dismiss checks at all four sizes. Tokenizers, Dataset, and Settings routes plus empty, loading, error, and long-identifier states remain. | MEDIUM |
| test-infrastructure.hosted-ci-and-release-evidence | PARTIAL | Hosted CI and committed QA evidence passed for `dc8cae26ea81b1c621c800168d5b3581c9fab04f`; validate release publication separately. | MEDIUM |

## Validation Campaign Roadmap

Last updated: 2026-09-23

The comprehensive validation roadmap supplied for this repository is normalized
here into stable slice IDs. This campaign layer does not replace the component
ledger above: it records what was exercised at a particular revision and keeps
feature existence, execution, partial behavior, and pass status distinct.

Campaign statuses map to the component taxonomy as follows: PASS is meaningful
evidence at the stated slice scope; PARTIAL is incomplete or bounded evidence;
FAIL is a reproduced application defect; BLOCKED is an external or environment
gate; UNTESTED means the implementation exists but the slice has not been
exercised; UNKNOWN means the contract itself needs clarification. The execution
order is dependency-first: Tier 0 environment/startup, Tier 1 foundations,
Tier 2 local workflows, Tier 3 populated reports/export, Tier 4 external
providers/PostgreSQL, then Tier 5 resilience/responsive/performance/hosted
evidence.

| Slice | Feature exists | Exercised | Slice status | Current evidence or next action |
| --- | --- | --- | --- | --- |
| T0-01 | yes | yes | PASS | Revision and local quality baseline accepted conditionally at `c48eefa`; refresh only when dependencies or gates change. |
| T0-02 | yes | yes | PASS | V-20260922 at `e864197c`: clean bootstrap, warm reuse, missing/malformed stamps, stale build, backend-only repair, invalid ports, redirected conflict, and a real port reacquisition race passed; the launcher contract suite passed 19/19; and the safety-gated harness reproduced a real localized `Stop-Process` access denial against synthetic PID 39516 on port 63335. The listener remained, the blocked PID/port and termination error were reported, and no backend/frontend service started. |
| T0-03 | yes | yes | PASS | All 13 maintenance-menu routes were exercised in a disposable Windows checkout, including Standard/Development installs, expected update refusal on `develop`, destructive declines/approvals, cleanup, uninstall, and Kill All. |
| T1-01 | yes | yes | PASS | Startup E2E passed 2/2 and the rendered shell reloaded after the persistence restart. |
| T1-02 | yes | yes | PASS | V-20260922 at `41f265a`: all 16 fields, boundaries, sparse persistence, restart, 409 conflict, individual/reset-all semantics, and new-work effects have backend, Chrome E2E, and in-app browser evidence; see the [current T1-02 QA record](../../assets/QA/tkben-t1-02-runtime-settings-20260922.md). |
| T1-03 | yes | yes | PASS | V-20260922 at `09805b7`: rendered Settings > Keys lifecycle and direct SQLite ciphertext inspection passed with masked API responses, reveal-policy `403`, single-active switching, active-delete protection, and cleanup; supplied-credential coverage was skipped because `TKBEN_TEST_HF_KEY` was unavailable. See the [T1-03 QA record](../../assets/QA/tkben-t1-03-key-management-20260922.md). |
| T1-04 | yes | yes | PASS | Focused migration/persistence/settings contracts passed and a runtime override survived an official launcher restart; PostgreSQL remains separate. |
| T1-05 | yes | yes | PASS | V-20260922 at `993e52e8f0617d4bf98a60d62b4c5d5588541218`: populated Dataset and Tokenizer filter matrices passed for search, source, exact numeric boundaries, combined/no-match/reset states; the Dataset stale-request case, both Tokenizer catalogue completion orders, and deterministic Tokenizer discovery stale-result/stale-error cases passed. Frontend 60/60, lint, production build, and relevant backend route/filter tests passed. See the [T1-05 QA record](../../assets/QA/tkben-t1-05-catalog-filtering-races-20260922.md). |
| T2-01 | yes | yes | PASS | Local dataset API/UI lifecycle evidence exists; CSV/XLSX and complete UI consolidation remain. |
| T2-02 | yes | yes | PASS | V-20260922 at base revision `0993e900d44897975f495dc6a7f7acd520d6a70a`: all six metric families were selected, analyzed, persisted, rendered in the populated dashboard, and restored after reload; metric unit tests passed 78/78 and dataset-analysis API E2E passed 1/1. See the [T2-02 QA record](../../assets/QA/tkben-t2-02-dataset-metric-families-20260922.md). |
| T2-03 | yes | yes | PASS | V-20260923 at implementation revision `de1aaee6`: the controlled four-document dataset included an exact duplicate pair, URL/email/HTML structure, and an empty document; selected keys and aggregates matched the metric contract, and the populated report restored after reload. Focused Chrome E2E passed 1/1, metric unit tests passed 78/78, and the dataset/report were removed. See the [T2-03 QA record](../../assets/QA/tkben-t2-03-dataset-quality-structure-compression-20260923.md). |
| T2-04 | yes | yes | PASS | Custom tokenizer upload/delete API lifecycle passed; restart, re-upload collision, and complete UI coverage remain. |
| T2-05 | yes | yes | PASS | V-20260923: local 1,207-entry custom tokenizer report persisted and reopened with the same report ID; API pages returned 500/500/207 contiguous entries and the populated UI exercised first/middle/final navigation. Focused Chrome E2E passed 1/1; see the [T2-05 QA record](../../assets/QA/tkben-t2-05-tokenizer-report-vocabulary-20260923.md). |
| T2-06 | yes | yes | PASS | V-20260923 at `f736f938`: opt-in Chrome E2E created a local report through the populated wizard, verified the persisted report via API, and reloaded the same report ID with populated charts. Cleanup verified report, tokenizer, and dataset removal; the combined Cross Benchmark and benchmark API E2E set passed 9/9. See the [T2-06 QA record](../../assets/QA/tkben-t2-06-cross-benchmark-wizard-20260923.md). |
| T2-07 | yes | yes | PASS | V-20260923 at `cc91fec1`: opt-in Chrome E2E reached API progress >=20%, cancelled to terminal `cancelled` with no report, restored Run, then completed and rendered an immediate two-document rerun. See the [T2-07 QA record](../../assets/QA/tkben-t2-07-benchmark-cancellation-20260923.md). |
| T3-01 | yes | yes | PASS | V-20260923: three comparable 1,000-document Windows runs recorded throughput with 95% intervals, p50/p95/p99 latency (80 observations/run), phase timings, peak RSS, and memory delta. Raw observations and runtime/hardware profile are in the [benchmark validation campaign QA record](../../assets/QA/tkben-benchmark-validation-campaign-20260923/README.md); samples describe this host and do not establish general performance. |
| T3-02 | yes | yes | PARTIAL | V-20260923: special-token, padding, truncation, non-default run configuration, and per-document sample behavior were exercised and persisted report responses omit undeclared config fields. Parallelism value 2 persists but has no execution effect while the tokenizer loop remains serial; see the [benchmark validation campaign QA record](../../assets/QA/tkben-benchmark-validation-campaign-20260923/README.md). |
| T3-03 | yes | yes | PASS | V-20260923: live persisted report search, 25/1 pagination, inline tag save/reload, confirmation, and physical deletion passed against 26 current-schema reports. See the [T3-03/04 QA record](../../assets/QA/tkben-t3-03-04-cross-benchmark-workflows-20260923.md). |
| T3-04 | yes | yes | PASS | V-20260923: live baseline/delta and data-table rendering, baseline reload, clone configuration, and visualization/order/customization persistence passed. See the [T3-03/04 QA record](../../assets/QA/tkben-t3-03-04-cross-benchmark-workflows-20260923.md). |
| T3-05 | yes | yes | PARTIAL | Export routes/services are covered by tests; live dataset, tokenizer, and benchmark PDFs remain uninspected. |
| T4-01 | yes | no | BLOCKED | Requires approved public Hugging Face network/provider validation. |
| T4-02 | yes | no | UNTESTED | Requires public dataset network and disk validation. |
| T4-03 | yes | no | BLOCKED | Requires an explicitly approved gated/private Hugging Face credential. |
| T4-04 | yes | no | BLOCKED | Requires a disposable PostgreSQL target and credentials. |
| T5-01 | yes | yes | PASS | Real Linux restart E2E kept the same active-job ID addressable as terminal `failed` with an interruption reason; a completed upload job and its 25,000-document dataset remained persisted. |
| T5-02 | yes | no | UNTESTED | Run populated and empty Dataset/Settings responsive and keyboard matrix at documented viewports. |
| T5-03 | yes | yes | PARTIAL | V-20260923: populated Cross Benchmark view plus report-manager, clone, and customize dialogs passed visual bounds checks at 1920x1080, 1440x900, 1024x768, and 390x844; Escape and focus-return behavior passed. Tokenizers route responsive/keyboard coverage remains. See the [T3-03/04 QA record](../../assets/QA/tkben-t3-03-04-cross-benchmark-workflows-20260923.md). |
| T5-04 | yes | yes | PASS | V-20260923: official Windows launcher streamed a 10,000-document run to visible 20% progress; cancellation reached terminal `cancelled` with no report saved, eight backend RSS samples were collected, and a two-document rerun completed and rendered without browser or HTTP errors. See the [benchmark validation campaign QA record](../../assets/QA/tkben-benchmark-validation-campaign-20260923/README.md). Browser heap was not instrumented; the gate's live progress, cancellation, available resource metrics, and rerun checks passed. |
| T5-05 | yes | yes | PARTIAL | Linux manual startup, proxy, major routes, restart, reconciliation, and shutdown passed in Ubuntu 26.04; macOS and hosted `ubuntu-latest` remain untested. |
| T5-06 | yes | yes | PARTIAL | Hosted CI run 35875638854 passed for exact pushed implementation commit `dc8cae26ea81b1c621c800168d5b3581c9fab04f` (frontend and backend jobs); release publication evidence remains separate. See the [T3-03/04 QA record](../../assets/QA/tkben-t3-03-04-cross-benchmark-workflows-20260923.md). |

### Tier 1 execution record

The first foundation slice was executed on 2026-09-21 at revision
`c48eefa8bad8e85d1bc18bcb8cf22042e1f72b38` on Windows using the supported
launcher. The compact, non-sensitive command/result record is
[assets/QA/tkben-tier1-20260921.md](../../assets/QA/tkben-tier1-20260921.md).

- `T1-01`: the startup browser suite passed 2/2; the in-app browser rendered
  Dataset and Settings after the backend/frontend restart.
- `T1-02`: the Settings browser round-trip passed 1/1 with the installed Chrome
  channel; the in-app browser also showed the tokenizer cross-field error and
  recovery. A runtime override survived launcher restart, then reset-all
  returned the default and removed `runtime-settings.json`.
- `T1-03`: synthetic key CRUD passed duplicate rejection, single-active
  switching, active-delete protection, reveal policy, masked responses, and
  plaintext-absence checks. Synthetic rows were removed and the key list ended
  empty.
- `T1-04`: 55 focused backend tests passed, including settings, migration,
  persistence, and key-route contracts; the runtime persistence restart check
  passed live.
- `T1-05`: the dataset stale-response browser scenario passed 1/1; the
  frontend unit suite passed 60/60 across 14 files, including dataset and
  tokenizer store behavior.
- `T1-05` closure at implementation revision `993e52e8f0617d4bf98a60d62b4c5d5588541218`:
  the populated Dataset and Tokenizer filter matrices passed end to end with
  request-query and rendered-row assertions; the Tokenizer catalogue race
  passed in both completion orders; deterministic discovery stale-result and
  stale-error cases passed; local fixture state was removed afterward. The
  detailed command, environment, and cleanup record is the [T1-05 QA
  record](../../assets/QA/tkben-t1-05-catalog-filtering-races-20260922.md).

### T1-02 closure follow-up

The T1-02 boundary closure was executed on 2026-09-22 at revision
`2966daf1732e6604570bab43b0728d8015c947ad` on Windows using the supported
launcher. The detailed non-sensitive record is
[assets/QA/tkben-t1-02-settings-boundary-20260922.md](../../assets/QA/tkben-t1-02-settings-boundary-20260922.md).

- The rendered Settings matrix covered all 16 runtime fields, backend-derived
  bounds, 38 invalid cases, valid boundary values, decimal semantics, MiB
  conversion, and all tokenizer relationship states.
- The combined save/reload assertion verified the exact effective values,
  revision increment, complete `overridden_keys` set, and all 16 controls after
  reload. The original snapshot was restored in `finally`; the final API state
  had defaults active and no runtime-settings file.
- The sweep exposed and corrected two frontend issues: the dynamic default
  discovery maximum could mask its cross-field error, and the combined
  tokenizer invalid state could hide the candidate-cap field error. Focused
  frontend regression coverage now preserves both relationship errors.

The repository-local Playwright Chromium executable was unavailable, so the
browser E2E used the installed Chrome channel. The official `-KillAll` command
also hit Windows `Access denied` while inspecting process command lines; only
the known launcher-owned process trees were then stopped explicitly. This is an
environment/permission gap, not a reproduced application defect.

### T1-02 runtime settings evidence refresh

The validation additions were committed on `develop` at revision
`41f265af004fd9dd99eaa0445a3c04000d6b106c`. The current acceptance evidence and
field-by-field schema matrix are recorded in
[the T1-02 runtime settings QA record](../../assets/QA/tkben-t1-02-runtime-settings-20260922.md).

- **Schema matrix:** All 16 public settings were derived from the current
  request schema and cross-checked against typed defaults, editable keys, UI
  controls, sparse persistence representation, and runtime consumers. The
  field-by-field matrix is in the linked QA record.
- **Validation and atomicity:** Backend coverage passed numeric boundaries,
  malformed and empty/null input, tokenizer cross-field constraints, and
  rejected mixed updates without in-memory or persisted mutation.
- **Persistence lifecycle:** Sparse-store recreation resolved omitted fields
  to defaults. In the in-app browser, five saved overrides survived Settings
  reload and backend restart; the API returned the exact overrides and default
  snapshot.
- **Conflicts and resets:** Stale revision updates returned HTTP 409 without
  replacing newer values. Individual reset retained the unrelated streaming
  override; reset-all followed by another backend restart restored defaults
  and left no `runtime-settings.json`.
- **Downstream effects:** New dataset, tokenizer-discovery, benchmark-prefill,
  streaming, and job workflows used changed settings. E2E-created dataset and
  tokenizer inputs were removed; completed upload-job history remains under
  normal terminal-job retention, and existing artifacts were not rewritten.
- **Regression and browser gates:** The isolated backend unit suite passed
  504/504; Settings E2E passed 2/2; Ruff, BasedPyright, 60 Angular unit tests,
  frontend lint, and production build passed. The repository runner's Python
  collection remains blocked by access denied on a generated
  `app/tests/cache/pytest` directory. Commands and browser steps are recorded
  in the QA artifact.

### T1-03 closure follow-up

The T1-03 closure was executed on 2026-09-22 at revision
`09805b7062325b6180ef956419c1d5db1b4d2e89` on Windows using the supported
launcher with disposable validation ports and data. The detailed non-sensitive
record is the [T1-03 QA record](../../assets/QA/tkben-t1-03-key-management-20260922.md).

- The rendered Settings > Keys flow passed add, masking, duplicate rejection,
  activation/deactivation/reactivation, single-active switching,
  active-delete protection, inactive deletion, reveal-policy denial, and final
  empty-state rendering.
- Direct SQLite inspection passed for validation-created rows: ciphertext was
  present and did not equal or contain the generated plaintext. The encryption
  material file was confirmed separate from the database. No secret or
  ciphertext value was recorded.
- Validation-created keys and the isolated data directory were removed. The
  isolated API and SQLite key counts returned to zero, the pre-existing port
  5000 backend remained at zero keys, and ports 5001/8001 had no listeners.
- `TKBEN_TEST_HF_KEY` was unavailable, so the supplied-credential-specific
  test was skipped without substituting or logging a live secret.

## Resolved and Historical Findings

Resolved findings are retained here only as provenance. They are not active
issues and must not be copied back into the current component rows unless a
regression is reproduced.

| Finding | Current state | Provenance |
| --- | --- | --- |
| ISSUE-001: active jobs disappeared on restart | Resolved for job visibility and lifecycle state; migration 0005 persists job metadata, and startup reconciles prior `pending`/`running` jobs to `failed` with a restart-interruption reason. Linux restart E2E proved the original job ID remains addressable and the completed upload/data remain persisted. Job resumption remains intentionally unsupported. | [gate-closure QA record](../../assets/QA/tkben-partial-gates-20260922.md), [architecture review](architecture/architecture_review.md#architecture-risks), [persistence](architecture/persistence.md) |
| Architecture P1: ambiguous backend ownership | Resolved; contracts, configuration, observations, repositories, and report orchestration now have explicit homes. | [architecture review findings](architecture/architecture_review.md#findings) |
| Architecture P2: benchmark admission mixed route and execution concerns | Resolved; BenchmarkService.prepare_run() is the admission boundary and execution keeps a defensive dataset check. | [architecture review findings](architecture/architecture_review.md#findings) |
| Architecture P3: redundant frontend API type alias | Resolved; consumers use the canonical API model module. | [architecture review findings](architecture/architecture_review.md#findings) |
| Legacy configuration, cache, tokenizer, report, and dashboard ownership paths | Removed or canonicalized; incompatible persisted rows fail explicitly instead of being silently adapted. | [canonical-source remediation](architecture/canonical_source_remediation.md), [architecture remediation](architecture/architecture_review.md#remediation) |
| v4.4.0 report tags, benchmark cloning, dashboard persistence, chart controls, and cache/launcher cleanup | Included in the current tagged source-only release; detailed live evidence still requires the release gates listed above. | [v4.4.0 release notes](runtime/release.md#v4.4.0-release-notes) |

## Validation Evidence: Current Bootstrap

Evidence identifier: V-20260920.

Baseline:

- Checkout: e318852542755fd54731e19a38f7ef231c0e6ee6
- Tag: v4.4.0
- Branches: local develop, origin/develop, local main, and origin/main were
  aligned at the checkout baseline.
- Reinstalled environments: uv sync --locked --project app/server --extra
  test and npm ci --no-audit --no-fund.

Automated evidence:

- Tracked backend source compilation passed, excluding the repository-managed
  .venv directory from traversal.
- Ruff passed. The command emitted a non-failing access-denied warning from an
  existing test-cache path.
- BasedPyright passed with 0 errors and 1,914 warnings.
- Embedded SQLite initialization passed and verified Alembic head
  0004_benchmark_report_tags.
- Backend unit suite passed: 323 passed, 18 warnings.
- OpenAPI smoke import and schema generation passed.
- Frontend lint passed.
- Frontend unit suite passed: 12 test files and 55 tests.
- Frontend production build passed.
- Focused live API checks passed: 9 basic contract/error checks, 10 dataset
  and custom-tokenizer API cases passed with 3 conditional provider/report
  cases skipped, and 5 benchmark API cases passed including report
  persistence and deletion.

Browser/manual evidence:

- The in-app browser rendered /dataset, /tokenizers, /cross-benchmark, and
  /settings.
- The Cross Benchmark wizard reached Inputs and disabled Next when no dataset
  or tokenizer was available.
- The browser console error log was empty for the exercised tab.

Cleanup:

- The generated dataset, custom tokenizers, and benchmark report were deleted
  and their list endpoints returned empty state.
- Temporary backend/frontend processes were stopped; ports 5000 and 8000 had
  no remaining listeners.

This evidence does not establish live Hugging Face, PostgreSQL, hosted CI,
official launcher, populated report-dashboard, PDF, or full responsive visual
coverage. Those boundaries remain explicit in the ledger above.

## Validation Evidence: Gate Closure

Evidence identifier: V-20260922. The validated source changes are an
uncommitted working tree based on develop HEAD
`4b608b7d44a22767e665669cfc420409ffd2f006`; detailed commands, state checks,
versions, and remaining gaps are in the
[gate-closure QA record](../../assets/QA/tkben-partial-gates-20260922.md).

- Windows 11 Pro, Windows PowerShell 5.1: all 13 launcher menu routes passed in
  a disposable checkout. Both install profiles, rebuild, database init, tests,
  update check, the expected `develop` update refusal, and each destructive
  route's decline/approval behavior were exercised. Sentinels confirmed the
  expected data/log/cache/uninstall preservation boundaries.
- A clean Windows bootstrap started without managed Python, Node, uv, backend
  environment, frontend dependencies/build, stamps, database, or generated
  `.env`. It pinned Python 3.14.7, Node 22.23.1, and uv 0.12.17, installed from
  lockfiles, migrated SQLite to 0005, built Angular, wrote all stamps, and
  reached backend/frontend readiness. Warm launch skipped setup with all stamp
  hashes and timestamps unchanged. A stale uv 0.12.16 was replaced by the pin.
- Windows launcher failure checks covered malformed/missing stamps, stale
  frontend build, backend-only repair, invalid port configuration, a redirected
  launch conflict, and a listener that reclaimed port 5000 between the two
  checks. The launcher aborted without terminating the synthetic listener.
  Permission-denied termination remains untested; therefore the wider launcher
  component remains PARTIAL.
- Ubuntu 26.04 manual startup reached health and frontend readiness; the
  browser rendered Dataset, Tokenizers, Cross Benchmark, and Settings, and the
  `/api/*` proxy worked. A real backend restart left active job `3fa194eb`
  addressable as failed with an interruption reason, kept completed upload job
  `c079b263` addressable, and retained the 25,000-document test dataset.
  Existing application feature tables were empty before this scenario, so this
  evidence does not claim preservation of pre-existing populated reports,
  tags, settings, or tokenizer artifacts.
- macOS and hosted `ubuntu-latest` remain unvalidated. The container's system
  Python was 3.14.4, while CI pins 3.14.7; Linux evidence remains PARTIAL at the
  broader macOS/Linux component scope.
- Current local gates: 212 backend unit tests passed; Ruff passed; BasedPyright
  reported 0 errors and 1,953 warnings; frontend lint and all 60 frontend unit
  tests passed; 17 launcher contract tests passed; the manual Windows test-menu
  run passed 378 tests with 4 skips before the final launcher contract-test
  additions. Hosted CI is not inferred from these local results.
