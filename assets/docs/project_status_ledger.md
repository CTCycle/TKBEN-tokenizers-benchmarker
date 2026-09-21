# Project Status Ledger
Last updated: 2026-09-21

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
Statuses describe this checkout and must be refreshed after behavioral
changes.

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
- No reproducible application defect was observed in this bootstrap, so there
  is no active BROKEN component. This does not make unvalidated areas
  implicitly safe.
- Hugging Face live discovery/report flows and PostgreSQL runtime equivalence
  are BLOCKED by conditional external-provider/database gates.
- Windows launcher maintenance branches, runtime-settings persistence,
  populated report dashboards, tokenizer reports, PDF export, hosted CI, and
  the documented responsive visual matrix remain validation debt.
- Containerized deployment and binary packaging are explicitly
  NOT_IMPLEMENTED; source-only local distribution is the supported release
  model.

## Current Component Ledger

### Architecture, runtime, and configuration

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture.canonical-ownership | VALIDATED | API, contracts, configuration, services, repositories, and frontend state ownership boundaries. | V-20260920: 323 backend unit tests passed, including the architecture-boundary contract; current architecture docs describe the same ownership graph. | Remaining canonicalization follow-ups are tracked as ISSUE-003; no current regression observed. | — | 2026-09-20 | unit | [architecture review](architecture/architecture_review.md), [canonical-source remediation](architecture/canonical_source_remediation.md), [boundary test](../../app/tests/unit/server/test_architecture_boundaries.py) | Revalidate the boundary test after ownership or schema changes. |
| runtime.startup.local-webapp | VALIDATED | FastAPI readiness, Angular production preview, local API proxy, and empty-state route loading. | V-20260920: backend health returned 200, frontend returned 200, and the browser rendered Dataset, Tokenizers, Cross Benchmark, and Settings routes with no console errors. V-20260921: the canonical launcher opened the frontend before backend readiness and the normal shell transitioned cleanly after health recovered. | This validates the local startup path; it does not validate every launcher menu branch. | — | 2026-09-21 | E2E + manual | [startup](runtime/startup.md), [runtime modes](runtime/modes.md), [system overview](architecture/system_overview.md), [startup test](../../app/tests/unit/server/test_app_startup.py) | Revalidate after launcher or readiness changes. |
| runtime.windows-launcher | WORKING | start_on_windows.ps1 dependency bootstrap, source-stale build detection, readiness checks, maintenance menu, and process cleanup. | V-20260921: official `-Launch` rebuilt the stale frontend, started backend and preview independently, opened the browser before backend health, then completed backend supervision successfully. | Maintenance and non-launch menu branches remain outside this validation pass. | — | 2026-09-21 | manual | [startup](runtime/startup.md), [deployment](runtime/deployment.md), [release procedure](runtime/release.md) | Validate the maintenance menu and redirected diagnostic branches separately. |
| configuration.runtime-settings | WORKING | Typed runtime defaults, sparse persisted overrides, revision checks, reset behavior, and Settings API/page. | 323 unit tests passed; the live Settings page rendered all Data, Tokenizers, Benchmarks, Runtime, and Keys sections. | Current run did not execute the browser round-trip, conflict, reset, or new-operation-effect flow. | — | 2026-09-20 | unit + manual | [configuration](runtime/configuration.md), [backend API](architecture/backend_api.md), [Settings E2E](../../app/tests/e2e/test_settings_ui.py) | Run the Settings E2E round-trip and restore the original runtime snapshot. |
| runtime.managed-job-lifecycle | VALIDATED | Start, poll, complete, fail, and cooperatively cancel in-process jobs. | 323 unit tests passed; the current benchmark E2E completed and persisted a report through the managed job path. | Active jobs are held in process memory and are lost on process restart; see ISSUE-001. | — | 2026-09-20 | unit + E2E | [execution and data flow](architecture/execution_and_data_flow.md), [benchmark contract](architecture/benchmark_contract.md), [job tests](../../app/tests/unit/server/services/test_jobs_manager.py) | Keep restart-loss behavior explicit if deployment scope expands. |

### Backend, persistence, and data workflows

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| backend.api-contracts | VALIDATED | /api/* routing, request/response contracts, health, settings, datasets, tokenizers, benchmarks, jobs, keys, and exports. | V-20260920: OpenAPI smoke passed; 323 unit tests and 9 focused live API checks passed. | Provider-dependent paths are separately bounded below. | — | 2026-09-20 | unit + integration + E2E | [backend API](architecture/backend_api.md), [OpenAPI test](../../app/tests/unit/server/test_openapi_schema.py) | Re-run focused API checks when a contract or route changes. |
| persistence.sqlite-alembic | VALIDATED | Embedded SQLite initialization, migration locking, current schema, report/tag persistence, rollback, and lifecycle visibility. | V-20260920: database initialization reached Alembic head 0004_benchmark_report_tags; 323 unit tests passed, including migration and persistence contracts. | The application intentionally rejects incompatible or unversioned non-empty databases rather than adapting them silently. | — | 2026-09-20 | integration + unit | [persistence](architecture/persistence.md), [database initialization](../../app/tests/unit/server/repositories/test_database_initialization.py), [migration tests](../../app/tests/unit/server/repositories/test_database_migrations.py), [persistence tests](../../app/tests/unit/server/repositories/test_persistence_contract.py) | Re-run migration and persistence contracts for schema changes. |
| persistence.postgresql-runtime | BLOCKED | External PostgreSQL initialization, migration locking, concurrency, and runtime equivalence with SQLite. | Unit coverage exercises initializer branches, but no disposable PostgreSQL target was available in V-20260920. | SQLite success must not be promoted to PostgreSQL equivalence. | Requires a disposable PostgreSQL target and configured connection/credentials. | — | None | [persistence](architecture/persistence.md), [configuration](runtime/configuration.md) | Provision a disposable target and run the documented PostgreSQL integration validation. |
| data.dataset-import-and-analysis | VALIDATED | CSV upload, ready-state persistence, list visibility, missing-dataset handling, asynchronous analysis, histograms, and statistics. | V-20260920: all dataset API E2E cases passed, including upload and analysis of a small CSV; generated records were removed and list state returned to empty. | Large files and remote dataset downloads remain dependent on local disk/network conditions. | — | 2026-09-20 | E2E + unit | [backend API](architecture/backend_api.md), [persistence](architecture/persistence.md), [dataset E2E](../../app/tests/e2e/test_datasets_api.py) | Revalidate download and large-streaming paths when those surfaces change. |
| data.custom-tokenizer-storage | VALIDATED | Custom tokenizer JSON compatibility, canonical artifact storage, catalog visibility, and deletion. | V-20260920: valid upload/deletion and invalid-input API E2E cases passed; generated tokenizer artifacts were removed afterward. | Report generation for a persisted tokenizer was not included in the current run. | — | 2026-09-20 | E2E + unit | [system overview](architecture/system_overview.md), [benchmark contract](architecture/benchmark_contract.md), [tokenizer E2E](../../app/tests/e2e/test_tokenizers_api.py) | Add the custom-tokenizer report/vocabulary path to the next live validation. |
| integration.huggingface-discovery-and-download | BLOCKED | Live Hugging Face discovery, gated/public access, metadata filtering, download, and provider-backed tokenizer report flow. | The current tokenizer E2E run passed local cases but skipped the two provider-discovery cases and the report-flow case because their opt-in gates were not enabled. | No current live claim is made for provider availability, credential handling, or remote repository compatibility. | Requires a usable provider key/network and explicit opt-in live tests. | — | None | [backend API](architecture/backend_api.md), [testing and quality](coding/testing_and_quality.md), [tokenizer E2E](../../app/tests/e2e/test_tokenizers_api.py) | Run with an approved provider credential and record pass/skip/failure separately. |
| benchmark.execution-and-reporting | VALIDATED | Benchmark admission, custom-tokenizer execution, schema-3/report-5 report persistence, list/load, metadata, and physical deletion. | V-20260920: the five benchmark API E2E cases passed with a small local dataset/tokenizer; the round-trip report was deleted and verified absent. | Full multi-tokenizer/provider and long-running performance coverage is not implied. | — | 2026-09-20 | E2E + unit | [benchmark contract](architecture/benchmark_contract.md), [execution and data flow](architecture/execution_and_data_flow.md), [benchmark E2E](../../app/tests/e2e/test_benchmarks_api.py) | Revalidate changed metric families and long-running cancellation paths. |
| benchmark.dashboard-and-pdf-export | WORKING | Normalized report-v5 widgets, visualizations, data tables, dashboard layout, and PDF export contracts. | 323 unit tests, frontend 55 unit tests, lint, and production build passed; export route/service tests are present. | No current live PDF download/render or populated report dashboard browser pass was performed. | — | 2026-09-20 | unit | [benchmark dashboard](ui/benchmark_dashboard.md), [export tests](../../app/tests/unit/test_dashboard_export_service.py), [export route tests](../../app/tests/unit/server/api/test_exports_routes.py) | Run populated-report browser and PDF parity validation before release claims. |

### Frontend and user workflows

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ui.route-shell-and-empty-states | VALIDATED | Primary navigation, route loading, dataset/tokenizer empty states, benchmark empty state, and Settings shell. | V-20260920: four routes rendered in the in-app browser; browser console error log was empty. | Current visual check used the normal browser viewport, not the full documented responsive matrix. | — | 2026-09-20 | manual | [experience](ui/experience.md), [components and patterns](ui/components_and_patterns.md), [app-flow E2E](../../app/tests/e2e/test_app_flow.py) | Repeat at the required viewport sizes after layout changes. |
| ui.startup-readiness | VALIDATED | Frontend-first startup screen, asynchronous backend readiness polling, slow-start notice, retryable failure state, and transition into the existing shell. | V-20260921: the canonical launcher opened the frontend before backend health; the in-app browser showed the quiet token stream/benchmark graph while offline, the slow state, the 60-second failure state, and a retry that transitioned into Dataset after backend recovery. Follow-up validation confirmed the rebuilt production preview keeps the artwork moving while connecting and pauses it in the terminal failure state. Focused browser E2E passed 2/2; frontend unit tests 60/60, lint, and production build passed. | The current application has a dark theme only; the graph is intentionally illustrative and not a benchmark measurement. | — | 2026-09-21 | unit + E2E + manual | [startup](runtime/startup.md), [experience](ui/experience.md), [startup E2E](../../app/tests/e2e/test_startup_loading.py) | Revalidate after launcher or readiness changes. |
| ui.dataset-dashboard | WORKING | Dataset selection, validation controls, persisted analysis dashboard, charts, and export action. | Dataset upload/analyze API E2E passed; the live browser confirmed the empty dashboard and disabled export state. | A populated dashboard was not inspected in the current browser pass. | — | 2026-09-20 | E2E + manual | [experience](ui/experience.md), [dataset E2E](../../app/tests/e2e/test_datasets_api.py) | Add a populated report browser pass covering charts, malformed optional payloads, and export. |
| ui.settings-page | WORKING | Settings tabs, typed controls, inline validation, persistence, conflict handling, reset, and Keys section navigation. | Unit coverage passed and the rendered page exposed all five sections; the full settings browser lifecycle remains unrun. | Persistence and optimistic-concurrency behavior is not promoted to VALIDATED from this evidence. | — | 2026-09-20 | unit + manual | [configuration](runtime/configuration.md), [experience](ui/experience.md), [Settings E2E](../../app/tests/e2e/test_settings_ui.py) | Run and archive the settings lifecycle E2E. |
| ui.cross-benchmark-workflow | PARTIAL | Benchmark wizard, report manager, baseline selection, clone eligibility, tags, dashboard customization, and populated report rendering. | V-20260920: empty Cross Benchmark state rendered; wizard reached Inputs and correctly disabled Next with no dataset/tokenizer. Backend report round-trip passed separately. | Only empty-state and incomplete-input behavior was live-confirmed; populated report manager, baseline, clone, tags, and chart interactions remain unvalidated in this checkout. | — | 2026-09-20 | E2E + manual | [benchmark dashboard](ui/benchmark_dashboard.md), [experience](ui/experience.md), [cross-benchmark E2E](../../app/tests/e2e/test_cross_benchmark_dashboard.py) | Run the populated report-manager and responsive dashboard E2E before changing this status. |
| ui.tokenizer-report-and-vocabulary | UNVALIDATED | Tokenizer report generation, vocabulary paging, report dashboard, and vocabulary preview. | Code, API contracts, and unit coverage exist; the provider/report-flow E2E was skipped and no report was available for a populated browser check. | No evidence is sufficient for a current working claim over the full report workflow. | Provider-backed report data or an equivalent local report fixture is required. | — | None | [experience](ui/experience.md), [backend API](architecture/backend_api.md), [tokenizer E2E](../../app/tests/e2e/test_tokenizers_api.py) | Generate a report from an approved local/provider tokenizer and validate paging/rendering. |

### Distribution and test infrastructure

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deployment.source-only-local | VALIDATED | Supported source-folder distribution, current version alignment, and local production frontend artifact. | Current checkout is tag v4.4.0; backend package is 3.4.0, frontend package is 2.4.0, and production build passed. | GitHub-hosted release publication was not rechecked in this bootstrap. | — | 2026-09-20 | manual + integration | [release procedure](runtime/release.md), [deployment](runtime/deployment.md), [system overview](architecture/system_overview.md) | Recheck remote tag/release and hosted CI before the next publication. |
| deployment.windows-portable-bootstrap | WORKING | Automatic Windows Python/Node bootstrap and local launch path. | Runtime documentation and launcher implementation are present; system Node 22.23.1 and restored backend environment supported the direct equivalent path. | The portable runtime download/install path was not exercised after reinstall. | — | — | None | [runtime modes](runtime/modes.md), [startup](runtime/startup.md), [deployment](runtime/deployment.md) | Run a clean-machine or removed-runtime launcher validation. |
| deployment.cross-platform-manual | PARTIAL | Manual macOS/Linux startup versus Windows-only automatic bootstrap. | Manual commands are documented and the local app architecture is not inherently Windows-only. | Automatic runtime bootstrap and a full non-Windows validation path are absent. | — | — | None | [runtime modes](runtime/modes.md), [deployment](runtime/deployment.md) | Treat non-Windows support as manual-only until tested and supported explicitly. |
| deployment.containerized | NOT_IMPLEMENTED | Docker or other active container runtime configuration. | [Runtime modes](runtime/modes.md) explicitly records containerized mode as not implemented; no active root container configuration exists. | This is an absent capability, not a current local-app failure. | — | — | None | [runtime modes](runtime/modes.md), [deployment](runtime/deployment.md) | Add a separately scoped deployment design before implementation. |
| deployment.binary-packaging | NOT_IMPLEMENTED | Installer, executable, Tauri, portable binary, or package artifact. | [Release procedure](runtime/release.md) states that releases are source-only and contain no binary packaging workflow. | Source-only distribution is the intended current release model. | — | — | None | [release procedure](runtime/release.md) | Do not add packaging work to a source-only release. |
| test-infrastructure.local-quality-gates | VALIDATED | Backend compile, Ruff, BasedPyright, SQLite initialization, unit tests, OpenAPI smoke, frontend lint, unit tests, and production build. | V-20260920: compile passed for tracked backend source; Ruff passed with a non-failing cache permission warning; BasedPyright reported 0 errors and 1,914 warnings; 323 backend tests, lint, build, and OpenAPI smoke passed. V-20260921: 14 frontend test files/60 tests, lint, production build, 7 launcher architecture tests, and 2 startup browser E2E tests passed. | Hosted CI and the full browser E2E matrix are separate gates. | — | 2026-09-21 | unit + integration + E2E | [testing and quality](coding/testing_and_quality.md), [CI workflow](../../.github/workflows/ci.yml), [test runner](../../app/tests/run_tests.bat), [startup E2E](../../app/tests/e2e/test_startup_loading.py) | Keep hosted and live-provider gates explicit in future reports. |
| test-infrastructure.hosted-ci-and-release-evidence | UNVALIDATED | Current hosted CI result and committed detailed QA/release evidence. | The repository workflow is present, but no current hosted run or tracked assets/QA record was available in this checkout. | Local gates must not be presented as hosted-CI or publication proof. | Requires hosted CI access and a non-sensitive QA record. | — | None | [CI workflow](../../.github/workflows/ci.yml), [release procedure](runtime/release.md) | Record hosted result and link the detailed QA artifact before release publication. |
| api.tokenizers.settings-compatibility | DEPRECATED | Legacy GET /api/tokenizers/settings compatibility response. | The API contract and OpenAPI test retain the endpoint as deprecated; new clients use /api/settings. | Compatibility surface should not become a second settings source. | — | 2026-09-20 | unit | [backend API](architecture/backend_api.md), [configuration](runtime/configuration.md) | Remove only after supported clients no longer depend on it and the removal is validated. |

## Open Issues

Severity and component status are separate. The following are actionable
current limitations or architectural follow-ups supported by repository
evidence; none is being mislabeled as a BROKEN component.

| ID | Affected component | Severity | Concise description | Current impact | Reproduction or evidence | Suspected cause | Blocker | Remediation status | Required revalidation | Related documentation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISSUE-001 | runtime.managed-job-lifecycle | MEDIUM | Active jobs are stored in an in-process registry and are not recoverable after a process restart. | A restart can lose the status and result of an active operation. | Start a long-running job, restart the backend, and observe that the prior in-memory job state is unavailable; this is also recorded as an architecture risk. | In-process job registry by design. | None for the local-app scope. | Accepted local-scope limitation; open for any broader deployment scope. | If deployment scope expands, add durable job state and restart/recovery E2E coverage. | [architecture review](architecture/architecture_review.md#architecture-risks), [execution and data flow](architecture/execution_and_data_flow.md#managed-jobs) |
| ISSUE-002 | deployment.network-auth-boundary | HIGH | Network-hosted deployments require an external authentication boundary before exposing key-management or destructive routes. | A network deployment without that boundary would expose sensitive operational actions beyond the local-app threat model. | The supported deployment documentation states the external-auth requirement; the repository does not provide that boundary. | Deployment scope is intentionally local/source-only. | An external authentication layer is not part of this repository. | Open pre-deployment requirement; not a local-webapp defect. | Before any network-hosted deployment, document and validate authentication, authorization, key protection, and destructive-route controls. | [deployment](runtime/deployment.md#constraints), [configuration](runtime/configuration.md#security-controls) |
| ISSUE-003 | architecture.canonical-ownership | MEDIUM | Several canonicalization follow-ups remain: environment-derived paths, duplicate metric representations, export dictionaries, handwritten frontend contracts, duplicated catalog metadata, and toolchain version declarations. | Future changes can drift across duplicated sources even though current ownership boundaries are explicit. | The remaining-work list is recorded in the canonical-source remediation document. | Historical duplication outside the first cleanup scope. | Broader contract/migration work is required; no current defect was reproduced. | Open planned architecture work. | Revalidate affected contracts, migration behavior, and UI/API parity after each follow-up. | [canonical-source remediation](architecture/canonical_source_remediation.md#remaining-canonicalization-work) |

No other reproducible functional defect was found in the current bootstrap.
Validation gaps below remain validation debt, not hidden issue records.

## Validation Debt

Validation debt identifies important areas with insufficient recent evidence.
It does not assert that the component is broken.

| Component | Current confidence | Missing validation | Priority |
| --- | --- | --- | --- |
| runtime.windows-launcher | WORKING | Execute the official start_on_windows.ps1 -Launch and -KillAll paths after removing/recreating managed runtimes, including stale-build and port-failure behavior. | HIGH |
| configuration.runtime-settings | WORKING | Run the Settings browser round-trip, inline invalid-value checks, optimistic-concurrency conflict, reset, and new-operation effect; restore the original snapshot. | HIGH |
| ui.cross-benchmark-workflow | PARTIAL | Exercise a populated report manager, dashboard customization, baseline persistence, clone eligibility, inline tags, data tables, and responsive layouts. | HIGH |
| ui.tokenizer-report-and-vocabulary | UNVALIDATED | Generate a report from an approved local/provider tokenizer and validate report rendering plus offset/limit paging. | HIGH |
| benchmark.dashboard-and-pdf-export | WORKING | Download and inspect a PDF from a populated report, including visualization overrides and chart/data-table parity. | MEDIUM |
| integration.huggingface-discovery-and-download | BLOCKED | Run the opt-in provider discovery, gated/public access, download, and report-flow tests with approved credentials and network. | HIGH |
| persistence.postgresql-runtime | BLOCKED | Run disposable PostgreSQL migration, concurrency, rollback, and runtime-equivalence checks. | MEDIUM |
| ui.responsive-visual-matrix | UNVALIDATED | Review populated, empty, loading, error, and long-identifier states at 1920x1080, 1440x900, 1024x768, and 390x844. | MEDIUM |
| test-infrastructure.hosted-ci-and-release-evidence | UNVALIDATED | Observe the hosted workflow result and link a detailed non-sensitive QA/release record; do not infer either from local gates. | MEDIUM |

## Resolved and Historical Findings

Resolved findings are retained here only as provenance. They are not active
issues and must not be copied back into the current component rows unless a
regression is reproduced.

| Finding | Current state | Provenance |
| --- | --- | --- |
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
