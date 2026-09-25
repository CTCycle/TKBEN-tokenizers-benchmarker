# Project Status Ledger
Last updated: 2026-09-25

This document is the canonical catalog of the current operational state of
TKBEN. It summarizes what is implemented, what has been observed working,
what has been formally validated, what remains partial or unvalidated, and
what is blocked by an external or environment-dependent gate. Detailed
architecture decisions, implementation plans, test code, and validation
records remain in their dedicated documents; this ledger links to them rather
than copying their narratives.

The historical bootstrap baseline is checkout e318852 (tag v4.4.0), with local
develop and main aligned. The current application implementation under
validation is `develop` at `21ed4ff6d32fceb5b8e6521e96b1ac9dcd0c75d2`; the
worktree was clean before this validation record was added. Historical v4.4.0
release/tag/CI evidence below is retained for provenance and does not validate
this current candidate.
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
cancellation, resource samples, and a rendered rerun. At that validation point,
T3-02 remained PARTIAL because parallelism was accepted and saved but was not
applied to tokenizer execution. See the [benchmark validation campaign QA
record](../../assets/QA/tkben-benchmark-validation-campaign-20260923/README.md).

V-20260924 closes T3-02 on local `develop` with the working-tree implementation
based on `b087db60df98639d85e294a5435453a45071abbe`. The official Windows
launcher and opt-in local campaign ran two uploaded custom tokenizers at
parallelism 1 and 2. Persisted config and runtime metadata matched both
requests; a worker high-water counter observed 1 and 2 active workers,
respectively. Results, observations, per-document samples, ordering, and report
reload were verified, and generated data was cleaned up. No Hugging Face
credential or network provider was used. Focused and full backend unit tests,
Ruff, BasedPyright, frontend unit tests, lint, and production build passed. See
the [T3-02 parallelism closure QA
record](../../assets/QA/tkben-t3-02-parallelism-closure-20260924/README.md).

V-20260923 closes T2-07 at implementation revision `cc91fec1b7307ae59091dc9e522627d8b69c6924`. The isolated Windows Chrome E2E cancelled a 10,000-document run after API progress reached 20%, observed terminal `cancelled` with no report, then immediately completed and rendered a two-document rerun. The focused frontend and backend gates, lint, and production build passed; see the [T2-07 QA record](../../assets/QA/tkben-t2-07-benchmark-cancellation-20260923.md).

V-20260923 closes T3-03 and T3-04 and validates the populated Cross Benchmark
portion of T5-03. The live Windows Chrome scenario created one- and two-tokenizer
reports through the official launcher, exercised persisted report management
and dashboard preferences, and reviewed the populated view and dialogs at four
viewport sizes. It also found and fixed a single-tokenizer chart label collision
and a baseline selector hydration defect. See the [T3-03/04 Cross Benchmark QA
record](../../assets/QA/tkben-t3-03-04-cross-benchmark-workflows-20260923.md).

V-20260923 closes T3-05 for dataset, tokenizer, and benchmark PDF exports and
validates the remaining malformed optional-histogram fallback in the Dataset
dashboard. The official Windows launcher exercised three populated local
reports; the real export route returned valid 2-, 2-, and 4-page PDFs, and all
eight rendered pages were inspected. The benchmark export preserved its
horizontal-bar override and displayed the complete shortened tokenizer name
and axis labels after a renderer fix. See the [T3-05 PDF export QA
record](../../assets/QA/tkben-t3-05-dashboard-pdf-exports-20260923.md).

V-20260924 closes T2-04 and T5-03's Tokenizers slice, and revalidates T2-05,
against application revision `3150c1603fc3a7cc74bfdc91d88068b80dd651c0`. The
official Windows launcher used an isolated data root. Same-filename upload
replacement, catalog uniqueness, restart persistence, persisted report and
vocabulary rendering, and UI deletion all passed. Tokenizers empty/loading/error
and populated report/long-identifier views plus Tokenizer Manager keyboard and
viewport bounds passed at four sizes. The T2-05 1,207-entry vocabulary paging
flow passed again. See the [T2-04/T2-05/T5-03 Tokenizers QA
record](../../assets/QA/tkben-t2-04-t5-03-tokenizer-validation-20260924/README.md).

V-20260924 also closes T2-01, T4-02, and T5-02 against the current `develop`
working tree. CSV and XLSX imports, three-document persistence and analysis,
unsupported `.xls` rejection, public Wikitext download with 29,119 documents
surviving an official-launcher restart, and post-import source-file cleanup were
verified. Populated/empty Dataset and Settings states plus route-tab keyboard
behavior passed at four viewport sizes without document-level horizontal
overflow. The focused suites passed 5/5 unit and 7/7 API E2E; Ruff, lock check,
Angular lint, and production build passed. Large-file/low-disk handling remains
PARTIAL; gated Hugging Face remains BLOCKED, while the PostgreSQL runtime gate is
now VALIDATED. See the [Dataset and Settings QA record](../../assets/QA/tkben-t2-01-t4-02-t5-02-dataset-settings-validation-20260924/README.md).

V-20260924 rechecks dataset-size and storage-failure handling on base checkout
`6889b1bf743fc2f9bdd3f3691c24c368517038ad`. The upload API accepted exactly the
configured 25 MiB limit and rejected one byte over; a 24 MiB CSV persisted as a
ready dataset. Real SQLite `SQLITE_FULL` errors after a committed batch triggered
partial-row cleanup in both custom-upload and dataset-backed persistence paths;
same-name retries succeeded after restoring the page limit. This controlled
page-limit case does not reproduce an exhausted host filesystem, so
`data.large-file-and-disk-exhaustion` remains PARTIAL. See the [dataset storage
QA record](../../assets/QA/tkben-data-upload-storage-20260924/README.md).

The Cross Benchmark workflow component and the aggregate responsive visual
matrix are now VALIDATED. Hosted CI run 36060275436 passed for storage-validation
commit `e870e135b5163044bf11565ff4ead8e3a72a4871`; both frontend and backend
jobs succeeded. V-20260925 also records the current-candidate local release
suite; the candidate-specific hosted-CI result will be appended to the [T5-06 QA
record](../../assets/QA/tkben-t5-06-current-candidate-20260925/README.md).
T5-06 remains PARTIAL because no `main` synchronization, annotated tag, or
public release was performed in this task. The public Hugging Face tokenizer
flow is validated; gated access remains BLOCKED and PostgreSQL runtime
equivalence is now VALIDATED as listed below. T5-05 is OUT_OF_SCOPE for the
Windows x64 release target.

V-20260925 extends dashboard PDF coverage. Two populated benchmark PDFs covered
all eight supported chart forms across their default and alternate forms; both
four-page exports passed `pdfinfo`, Poppler rendered all eight pages, and visual
review found labels, values, and axes legible without clipping. A persisted
1,207-entry custom-tokenizer report exported as a five-page PDF with 180
vocabulary rows; metadata and every rendered page passed visual review. See the
[open validation debt QA record](../../assets/QA/tkben-open-validation-debt-20260925/README.md).

V-20260925 also validates the public Hugging Face tokenizer path. The supplied
credential was entered in the isolated Settings > Keys UI, stored as decryptable
Fernet ciphertext with separate encryption material, and returned only as a
masked preview. A separate Hub identity request returned HTTP 200. The opt-in
`bert-base-uncased` flow passed discovery, download, report persistence, rendered
reload, and tokenizer cleanup (report 1, vocabulary size 30,522). Gated discovery
returned three candidates; each download failed, so no gated report was produced
and T4-03 remains BLOCKED. See the [HF provider evidence](../../assets/QA/tkben-open-validation-debt-20260925/hf-provider-flow-evidence.json),
[authentication evidence](../../assets/QA/tkben-open-validation-debt-20260925/hf-provider-auth-evidence.json),
and [key-storage evidence](../../assets/QA/tkben-open-validation-debt-20260925/hf-key-storage-evidence.json). The key, isolated database, and separate encryption material were removed after evidence capture; app processes and validation ports were cleaned up. PostgreSQL was left untouched.

A fixed-size VHD could not be provisioned because host OS authorization denied
the request, so physical disk-exhaustion recovery remains PARTIAL. Read-only
GitHub API verification identifies v4.4.0 as the latest public release. The
remote and local annotated tag object SHA is `82ad6c69052d00d75d2b01eb16fbbc5268054602`,
which targets commit `e318852542755fd54731e19a38f7ef231c0e6ee6`. Hosted CI run
35501626659 completed successfully for that commit with both frontend and
backend jobs successful; main's matching CI run 35499881089 also passed. No
new release was published during this validation. See the [release audit](../../assets/QA/tkben-open-validation-debt-20260925/release-audit.json)
and [QA record](../../assets/QA/tkben-open-validation-debt-20260925/README.md).
The existing host PostgreSQL listener was not contacted; no disposable target
or credentials were available. T4-02 remains PASS.

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
| OUT_OF_SCOPE | Explicitly excluded from the supported product or release target; no validation follow-up is planned. |
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
- Benchmark execution honors bounded tokenizer concurrency; T3-02 is PASS with
  persisted requested/effective metadata and local overlap evidence in the
  [T3-02 closure QA record](../../assets/QA/tkben-t3-02-parallelism-closure-20260924/README.md).
- Durable job visibility and restart reconciliation are VALIDATED at unit and
  Linux E2E levels. The Windows portable bootstrap and launcher are VALIDATED
  on the exercised Windows host, including the privilege-isolated denied
  process-termination branch.
- Two Windows launcher defects were reproduced and corrected during validation:
  script-block invocation mishandled ordinary stderr, and Kill All missed a
  quoted npm preview process while also matching nested process roots.
- Public Hugging Face discovery and report persistence are validated; gated
  repository access remains BLOCKED by the external-provider gate, while the
  isolated PostgreSQL runtime equivalence check is VALIDATED.
- Recent local evidence covers report management, dashboard customization,
  visualization overrides, populated PDF exports, and the Dataset/Settings
  responsive matrix and all eight supported dashboard PDF chart forms, plus a
  1,207-entry tokenizer PDF. Remaining validation debt includes physical
  low-disk recovery and the conditional gated-provider route.
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
| persistence.sqlite-alembic | VALIDATED | Embedded SQLite initialization, migration locking, current schema, report/tag persistence, rollback, and managed-job lifecycle rows. | V-20260922: migration tests cover SQLite upgrade from 0004 to 0005; Windows clean bootstrap and Linux startup reached Alembic head `0005_managed_job_lifecycle`; the restart E2E preserved the active/completed job rows and test dataset. V-20260924: controlled `SQLITE_FULL` after a committed import batch removed incomplete dataset and document rows in upload and dataset-backed imports; same-name retries passed after capacity was restored. | The application intentionally rejects incompatible or unversioned non-empty databases rather than adapting them silently. Actual host filesystem exhaustion remains separate PARTIAL validation debt. | — | 2026-09-24 | integration + unit + E2E | [persistence](architecture/persistence.md), [database initialization](../../app/tests/unit/server/repositories/test_database_initialization.py), [migration tests](../../app/tests/unit/server/repositories/test_database_migrations.py), [persistence tests](../../app/tests/unit/server/repositories/test_persistence_contract.py), [dataset storage QA record](../../assets/QA/tkben-data-upload-storage-20260924/README.md), [gate-closure QA record](../../assets/QA/tkben-partial-gates-20260922.md) | Re-run migration and persistence contracts for schema changes. |
| persistence.postgresql-runtime | VALIDATED | External PostgreSQL initialization, migration locking, concurrency, rollback, restart recovery, and runtime equivalence with SQLite. | V-20260925: an isolated PostgreSQL 18.6 cluster initialized to Alembic head `0005_managed_job_lifecycle`; concurrent initialization completed without duplicate creation or migration damage, and the focused migration suite passed 10/10. The rendered UI matched SQLite for upload, analysis, persisted report metrics, dashboard recovery, and reload; PostgreSQL restart recovery passed. An injected mid-upload constraint failure left zero dataset/document rows, and removing the trigger allowed a valid retry to succeed. See the [PostgreSQL runtime validation handoff](../../assets/QA/tkben-postgresql-runtime-validation-20260925/HANDOFF.md) and retained QA logs/fixtures in that folder. | The pre-existing host PostgreSQL listener at 127.0.0.1:5433 was not contacted; this evidence is for the isolated target and does not claim general deployment coverage. | — | 2026-09-25 | integration + E2E + manual | [persistence](architecture/persistence.md), [configuration](runtime/configuration.md), [PostgreSQL runtime validation handoff](../../assets/QA/tkben-postgresql-runtime-validation-20260925/HANDOFF.md) | Re-run after schema, migration, or persistence changes. |
| data.dataset-import-and-analysis | VALIDATED | CSV/XLSX upload, ready-state persistence, catalogue visibility, missing-dataset handling, asynchronous analysis, histograms, and statistics. | V-20260924: focused dataset routes passed 5/5 and API E2E passed 7/7. Generated three-row XLSX and CSV uploads persisted; a synthetic CSV was uploaded through the Dataset UI; unsupported `.xls` rejection was verified. Current checks accepted a 25 MiB API upload at the limit, rejected 25 MiB + 1 byte, and persisted a 24 MiB CSV as ready. See the [Dataset and Settings QA record](../../assets/QA/tkben-t2-01-t4-02-t5-02-dataset-settings-validation-20260924/README.md) and [dataset storage QA record](../../assets/QA/tkben-data-upload-storage-20260924/README.md). | Simulated SQLite `SQLITE_FULL` cleanup and retry pass; actual host filesystem exhaustion remains untested and is tracked separately. Public source coverage is limited to the Wikitext case recorded under T4-02. | — | 2026-09-24 | E2E + unit + manual | [backend API](architecture/backend_api.md), [persistence](architecture/persistence.md), [dataset E2E](../../app/tests/e2e/test_datasets_api.py), [Dataset and Settings QA record](../../assets/QA/tkben-t2-01-t4-02-t5-02-dataset-settings-validation-20260924/README.md), [dataset storage QA record](../../assets/QA/tkben-data-upload-storage-20260924/README.md) | Validate physical low-disk failure and recovery when an isolated volume is available. |
| data.custom-tokenizer-storage | VALIDATED | Custom tokenizer JSON compatibility, canonical artifact storage, catalog visibility, replacement, restart persistence, report generation, and deletion. | V-20260924: focused Chrome E2E verified valid/invalid API uploads, same-filename replacement with one catalog row, canonical seven-token artifact persistence through a real launcher restart, report generation/vocabulary rendering, and UI deletion of the catalog entry, report, and artifact. The local tokenizer E2E run passed 7/7 enabled cases; a separate post-restart report/delete E2E passed 1/1. See the [T2-04/T2-05/T5-03 Tokenizers QA record](../../assets/QA/tkben-t2-04-t5-03-tokenizer-validation-20260924/README.md). | No current local lifecycle limitation observed. Provider-backed discovery and download remain separate under `integration.huggingface-discovery-and-download`. | — | 2026-09-24 | E2E + manual + static checks | [system overview](architecture/system_overview.md), [benchmark contract](architecture/benchmark_contract.md), [tokenizer E2E](../../app/tests/e2e/test_tokenizers_api.py), [Tokenizers QA record](../../assets/QA/tkben-t2-04-t5-03-tokenizer-validation-20260924/README.md) | Revalidate after custom-tokenizer upload, persistence, report, or deletion changes. |
| integration.huggingface-discovery-and-download | PARTIAL | Live Hugging Face tokenizer discovery, metadata filtering, gated access, repository download, and provider-backed tokenizer report flow. | V-20260925: the supplied key authenticated through Hub `whoami`; public `bert-base-uncased` discovery, download, persisted report 1 (vocabulary size 30,522), rendered reload, and tokenizer cleanup passed. Gated discovery returned three repositories; all three download jobs failed, so gated access is not validated. See the [HF provider evidence](../../assets/QA/tkben-open-validation-debt-20260925/hf-provider-flow-evidence.json), [authentication evidence](../../assets/QA/tkben-open-validation-debt-20260925/hf-provider-auth-evidence.json), and [key-storage evidence](../../assets/QA/tkben-open-validation-debt-20260925/hf-key-storage-evidence.json). | No successful authorized gated download or gated report is evidenced. | Requires an already-authorized gated repository that completes download and report generation. | 2026-09-25 | Integration + E2E + manual | [backend API](architecture/backend_api.md), [testing and quality](coding/testing_and_quality.md), [tokenizer E2E](../../app/tests/e2e/test_tokenizers_api.py), [open validation debt QA record](../../assets/QA/tkben-open-validation-debt-20260925/README.md) | Rerun the gated flow after access is already granted to a candidate repository. |
| benchmark.execution-and-reporting | VALIDATED | Benchmark admission, custom-tokenizer execution, bounded tokenizer concurrency, schema-3/report-5 report persistence, list/load, metadata, and physical deletion. | V-20260924 closes T3-02: local two-tokenizer runs at parallelism 1 and 2 persisted the requested config, reported effective worker counts 1 and 2, observed two concurrent workers at parallelism 2, preserved result/observation/per-document ordering, and reloaded populated reports. Existing T2-06/T2-07 and T3 campaign evidence covers report workflows, cancellation, and rerun. See the [T3-02 closure QA record](../../assets/QA/tkben-t3-02-parallelism-closure-20260924/README.md), [T2-07 QA record](../../assets/QA/tkben-t2-07-benchmark-cancellation-20260923.md), [T2-06 QA record](../../assets/QA/tkben-t2-06-cross-benchmark-wizard-20260923.md), and [benchmark validation campaign](../../assets/QA/tkben-benchmark-validation-campaign-20260923/README.md). | Remote-provider validation remains separately tracked under `integration.huggingface-discovery-and-download`; this status makes no claim about remote providers or cross-machine performance. | — | 2026-09-24 | E2E + unit + static checks | [benchmark contract](architecture/benchmark_contract.md), [execution and data flow](architecture/execution_and_data_flow.md), [benchmark E2E](../../app/tests/e2e/test_benchmarks_api.py) | Continue separately scoped provider and performance validation. |
| benchmark.dashboard-and-pdf-export | VALIDATED | Normalized report-v5 widgets, all eight supported chart forms, data tables, dashboard layout, and PDF export contracts. | V-20260925: two populated four-page benchmark exports covered bar, horizontal_bar, interval_bar, dot_whisker, box_plot, histogram, grouped_bar, and heatmap forms; `pdfinfo` and all eight Poppler renders passed visual review. A 1,207-entry tokenizer report exported five pages with 180 vocabulary rows; metadata and all renders were reviewed. See the [open validation debt QA record](../../assets/QA/tkben-open-validation-debt-20260925/README.md), [benchmark PDF review](../../assets/QA/tkben-open-validation-debt-20260925/benchmark-pdf-visual-review.md), and [tokenizer PDF review](../../assets/QA/tkben-open-validation-debt-20260925/tokenizer-report-pdf-validation.md). | No current PDF clipping, chart-label, or large-vocabulary layout defect observed in the reviewed exports. | — | 2026-09-25 | API + manual PDF | [benchmark dashboard](ui/benchmark_dashboard.md), [T3-03/04 QA record](../../assets/QA/tkben-t3-03-04-cross-benchmark-workflows-20260923.md), [T3-05 QA record](../../assets/QA/tkben-t3-05-dashboard-pdf-exports-20260923.md), [open validation debt QA record](../../assets/QA/tkben-open-validation-debt-20260925/README.md), [export tests](../../app/tests/unit/test_dashboard_export_service.py), [export route tests](../../app/tests/unit/server/api/test_exports_routes.py) | Revalidate after widget or export-contract changes. |

### Frontend and user workflows

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ui.route-shell-and-empty-states | VALIDATED | Primary navigation, route loading, dataset/tokenizer empty states, benchmark empty state, and Settings shell. | V-20260924: populated/no-match Dataset and Settings Data/empty Keys states rendered without document-level horizontal overflow at four viewport sizes. Tokenizers empty/loading/error/populated states are recorded in the [Tokenizers QA record](../../assets/QA/tkben-t2-04-t5-03-tokenizer-validation-20260924/README.md); Dataset and Settings are in the [Dataset and Settings QA record](../../assets/QA/tkben-t2-01-t4-02-t5-02-dataset-settings-validation-20260924/README.md). | The Dataset/Settings keyboard check covers route tabs, manager dismissal, and focus return; it does not repeat the full Settings key lifecycle. | — | 2026-09-24 | E2E + manual | [experience](ui/experience.md), [components and patterns](ui/components_and_patterns.md), [app-flow E2E](../../app/tests/e2e/test_app_flow.py), [Dataset and Settings QA record](../../assets/QA/tkben-t2-01-t4-02-t5-02-dataset-settings-validation-20260924/README.md) | Revalidate after route-shell or responsive layout changes. |
| ui.startup-readiness | VALIDATED | Frontend-first startup screen, asynchronous backend readiness polling, slow-start notice, retryable failure state, and transition into the existing shell. | V-20260921: the canonical launcher opened the frontend before backend health; the in-app browser showed the quiet token stream/benchmark graph while offline, the slow state, the 60-second failure state, and a retry that transitioned into Dataset after backend recovery. Follow-up validation confirmed the rebuilt production preview keeps the artwork moving while connecting and pauses it in the terminal failure state. Focused browser E2E passed 2/2; frontend unit tests 60/60, lint, and production build passed. | The current application has a dark theme only; the graph is intentionally illustrative and not a benchmark measurement. | — | 2026-09-21 | unit + E2E + manual | [startup](runtime/startup.md), [experience](ui/experience.md), [startup E2E](../../app/tests/e2e/test_startup_loading.py) | Revalidate after launcher or readiness changes. |
| ui.dataset-dashboard | VALIDATED | Dataset selection, validation controls, persisted analysis dashboard, charts, export action, and safe handling of malformed optional histogram data. | V-20260923: the T2-02/T2-03 metric families and persisted dashboard reloads passed. T3-05 loaded a three-document report through the UI and verified the matching 2-page PDF. Histogram normalization now returns an empty series for malformed envelopes and filters invalid counts while preserving valid numeric strings and index labels; focused frontend tests, lint, and production build passed. See the [T2-03 QA record](../../assets/QA/tkben-t2-03-dataset-quality-structure-compression-20260923.md), [T2-02 QA record](../../assets/QA/tkben-t2-02-dataset-metric-families-20260922.md), and [T3-05 QA record](../../assets/QA/tkben-t3-05-dashboard-pdf-exports-20260923.md). | Malformed optional payloads are verified at the dashboard normalization boundary; the live backend emits well-formed report payloads. | — | 2026-09-23 | unit + E2E + manual | [experience](ui/experience.md), [dataset E2E](../../app/tests/e2e/test_datasets_api.py), [dashboard data tests](../../app/client/angular/app/core/utils/dataset-dashboard-data.spec.ts) | Revalidate after dataset metric, dashboard, or export-contract changes. |
| ui.settings-page | VALIDATED | Settings tabs, typed controls, inline validation, persistence, conflict handling, reset, Keys section navigation, and rendered key-management lifecycle. | V-20260922: the rendered Settings route passed all 16 controls, boundary errors, cross-field recovery, persistence, reload hydration, conflict, reset, and new-operation effects. The T1-03 Chrome flow passed add, masking, duplicate rejection, activation/deactivation/reactivation, active-delete protection, inactive deletion, single-active switching, reveal-policy denial, and final cleanup, with direct ciphertext and separate encryption-material checks. V-20260924: Settings Data and empty Keys states rendered without document-level overflow at four viewport sizes; ArrowRight/End route-tab navigation passed. V-20260925: isolated Settings > Keys credential entry stored decryptable ciphertext and returned only a masked preview. See the [T1-02 QA record](../../assets/QA/tkben-t1-02-runtime-settings-20260922.md), [T1-03 QA record](../../assets/QA/tkben-t1-03-key-management-20260922.md), [Dataset and Settings QA record](../../assets/QA/tkben-t2-01-t4-02-t5-02-dataset-settings-validation-20260924/README.md), and [HF key-storage evidence](../../assets/QA/tkben-open-validation-debt-20260925/hf-key-storage-evidence.json). | The supplied-credential gated tokenizer report remains separately blocked under T4-03; no current Settings page defect was observed. | — | 2026-09-25 | unit + E2E + manual | [configuration](runtime/configuration.md), [experience](ui/experience.md), [Settings E2E](../../app/tests/e2e/test_settings_ui.py), [T1-02 QA record](../../assets/QA/tkben-t1-02-runtime-settings-20260922.md), [T1-03 QA record](../../assets/QA/tkben-t1-03-key-management-20260922.md) | Revalidate after Settings, key-storage, or route interaction changes. |
| ui.cross-benchmark-workflow | VALIDATED | Benchmark wizard, report manager, baseline selection, clone eligibility/configuration, tags, dashboard customization, data tables, populated report rendering, and the Tokenizers portion of the responsive/keyboard workflow. | V-20260923: report-manager, dashboard, and four-viewport dialog flows passed; the Chrome run-options flow exercised non-default values and saved configuration. T3-05 exercised the live PDF export with the saved horizontal-bar choice and confirmed its data table parity. V-20260924 completed the Tokenizers route and Tokenizer Manager four-viewport and keyboard checks. See the [benchmark validation campaign QA record](../../assets/QA/tkben-benchmark-validation-campaign-20260923/README.md), [T3-05 QA record](../../assets/QA/tkben-t3-05-dashboard-pdf-exports-20260923.md), and [T2-04/T2-05/T5-03 Tokenizers QA record](../../assets/QA/tkben-t2-04-t5-03-tokenizer-validation-20260924/README.md). | — | — | 2026-09-24 | unit + E2E + manual | [benchmark dashboard](ui/benchmark_dashboard.md), [experience](ui/experience.md), [Cross Benchmark E2E](../../app/tests/e2e/test_cross_benchmark_report_workflows.py), [T3-03/04 QA record](../../assets/QA/tkben-t3-03-04-cross-benchmark-workflows-20260923.md), [T3-05 QA record](../../assets/QA/tkben-t3-05-dashboard-pdf-exports-20260923.md), [benchmark validation campaign](../../assets/QA/tkben-benchmark-validation-campaign-20260923/README.md), [Tokenizers QA record](../../assets/QA/tkben-t2-04-t5-03-tokenizer-validation-20260924/README.md) | Revalidate after Cross Benchmark workflow or responsive interaction changes. |
| ui.tokenizer-report-and-vocabulary | VALIDATED | Tokenizer report generation, vocabulary paging, report dashboard, and vocabulary preview. | V-20260923: a local 1,207-entry custom tokenizer generated and persisted a report; reload restored the same report ID and the populated UI navigated three vocabulary pages. API item IDs were contiguous across offsets 0, 500, and 1,000. Focused Chrome E2E passed 1/1 and 34 report, vocabulary, route, and service unit tests passed. V-20260924 reran the 1,207-entry report flow and inspected the rendered report at four viewport sizes; the post-restart custom-tokenizer report also rendered the seven-entry replacement vocabulary. See the [T2-05 QA record](../../assets/QA/tkben-t2-05-tokenizer-report-vocabulary-20260923.md) and [combined Tokenizers QA record](../../assets/QA/tkben-t2-04-t5-03-tokenizer-validation-20260924/README.md). | — | — | 2026-09-24 | unit + E2E + manual | [experience](ui/experience.md), [backend API](architecture/backend_api.md), [tokenizer E2E](../../app/tests/e2e/test_tokenizers_api.py), [T2-05 QA record](../../assets/QA/tkben-t2-05-tokenizer-report-vocabulary-20260923.md), [combined Tokenizers QA record](../../assets/QA/tkben-t2-04-t5-03-tokenizer-validation-20260924/README.md) | Revalidate after tokenizer report or vocabulary paging changes. |

### Distribution and test infrastructure

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deployment.source-only-local | PARTIAL | Supported Windows x64 source-folder distribution, current version alignment, and local production frontend artifact. | V-20260925: clean application revision `21ed4ff6d32fceb5b8e6521e96b1ac9dcd0c75d2` passed the local CI-equivalent release suite; candidate-specific hosted CI is pending after the evidence commit and will be recorded in the [T5-06 QA record](../../assets/QA/tkben-t5-06-current-candidate-20260925/README.md). No `main` synchronization, annotated tag, or public release was performed. Historical v4.4.0 evidence remains provenance only. | Current-candidate local release evidence is complete; source-release publication is not part of this task. | A coordinated release workflow is still required before publication. | 2026-09-25 | manual + integration + hosted CI | [release procedure](runtime/release.md), [deployment](runtime/deployment.md), [system overview](architecture/system_overview.md), [T5-06 QA record](../../assets/QA/tkben-t5-06-current-candidate-20260925/README.md), [release audit](../../assets/QA/tkben-open-validation-debt-20260925/release-audit.json) | Keep PARTIAL until candidate CI and a release are intentionally completed. |
| deployment.windows-portable-bootstrap | VALIDATED | Automatic Windows Python/Node/uv bootstrap, stamped dependency repair, deterministic build reuse, and local launch path. | V-20260922: clean Windows 11 bootstrap from absent managed runtimes, `.venv`, `node_modules`, build output, stamps, database, and `.env` downloaded Python 3.14.7, Node 22.23.1, and uv 0.12.17; created the environment file, synced locked dependencies, migrated to 0005, built Angular, wrote stamps, and started both services. Warm launch reused all three stamps; stale uv 0.12.16 was replaced by the pin. | Evidence is host-specific; it does not establish every Windows edition or hardware architecture. | — | 2026-09-22 | integration + manual | [runtime modes](runtime/modes.md), [startup](runtime/startup.md), [deployment](runtime/deployment.md), [gate-closure QA record](../../assets/QA/tkben-partial-gates-20260922.md) | Repeat on supported Windows architectures when the portable runtime pins change. |
| deployment.cross-platform-manual | OUT_OF_SCOPE | Manual startup outside the supported Windows x64 release target. | V-20260922: a Linux manual startup, API proxy, four browser routes, service restart, persistent job reconciliation, and clean shutdown were exercised in a disposable Ubuntu 26.04 container as diagnostic evidence. | Linux/macOS manual execution is not a supported release target. The Linux run does not establish platform support; no macOS or hosted Ubuntu runtime validation is planned. | — | 2026-09-24 | E2E + manual | [runtime modes](runtime/modes.md), [deployment](runtime/deployment.md), [gate-closure QA record](../../assets/QA/tkben-partial-gates-20260922.md) | None; non-Windows platform validation is outside release scope. |
| deployment.containerized | NOT_IMPLEMENTED | Docker or other active container runtime configuration. | [Runtime modes](runtime/modes.md) explicitly records containerized mode as not implemented; no active root container configuration exists. | This is an absent capability, not a current local-app failure. | — | — | None | [runtime modes](runtime/modes.md), [deployment](runtime/deployment.md) | Add a separately scoped deployment design before implementation. |
| deployment.binary-packaging | NOT_IMPLEMENTED | Installer, executable, Tauri, portable binary, or package artifact. | [Release procedure](runtime/release.md) states that releases are source-only and contain no binary packaging workflow. | Source-only distribution is the intended current release model. | — | — | None | [release procedure](runtime/release.md) | Do not add packaging work to a source-only release. |
| test-infrastructure.local-quality-gates | VALIDATED | Backend compile, Ruff, BasedPyright, SQLite initialization, unit tests, OpenAPI smoke, frontend lint, unit tests, and production build. | V-20260923: frontend unit suite passed 62/62 across 14 files, frontend lint/build passed, and the opt-in Chrome campaign passed 2/2. V-20260924 current backend suite passed 515/515 unit tests (23 warnings); Ruff passed on changed files; BasedPyright reported 0 errors and 2,018 warnings. No frontend files changed in the current storage slice. | Hosted CI, full browser E2E outside this campaign, and provider/database gates remain separate. | — | 2026-09-24 | unit + E2E + integration | [testing and quality](coding/testing_and_quality.md), [CI workflow](../../.github/workflows/ci.yml), [test runner](../../app/tests/run_tests.bat), [benchmark validation campaign](../../assets/QA/tkben-benchmark-validation-campaign-20260923/README.md), [dataset storage QA record](../../assets/QA/tkben-data-upload-storage-20260924/README.md) | Keep hosted and live-provider gates explicit in future reports. |
| test-infrastructure.hosted-ci-and-release-evidence | PARTIAL | Hosted CI result, public release/tag commit correlation, and detailed QA evidence. | V-20260925: the clean current application candidate passed the local release suite and is being pushed to `develop` for candidate-specific hosted CI; the matching run will be recorded in the [T5-06 QA record](../../assets/QA/tkben-t5-06-current-candidate-20260925/README.md). Historical v4.4.0 tag/release evidence remains separately linked for provenance. | Candidate CI is pending; no `main` synchronization, annotated tag, or public release was performed. | Requires the intentional release workflow before publication. | 2026-09-25 | hosted CI + manual | [CI workflow](../../.github/workflows/ci.yml), [release procedure](runtime/release.md), [T5-06 QA record](../../assets/QA/tkben-t5-06-current-candidate-20260925/README.md), [release audit](../../assets/QA/tkben-open-validation-debt-20260925/release-audit.json) | Record the matching run, then keep PARTIAL until a release is intentionally synchronized and published. |
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
| data.large-file-and-disk-exhaustion | PARTIAL | V-20260924 simulated `SQLITE_FULL` cleanup and same-name retries for upload and dataset-backed import. A small fixed-size VHD was attempted on 2026-09-25, but host OS authorization denied provisioning before a filesystem-full scenario could run; no VHD artifact remains. Physical exhaustion and recovery are still untested. See the [dataset storage QA record](../../assets/QA/tkben-data-upload-storage-20260924/README.md) and [open validation debt QA record](../../assets/QA/tkben-open-validation-debt-20260925/README.md). | MEDIUM |
| integration.huggingface-discovery-and-download | PARTIAL | Public discovery, download, report persistence, rendered reload, and cleanup passed on 2026-09-25. Missing validation is an authorized gated-repository download followed by gated report generation; T4-03 remains BLOCKED. See the [HF provider evidence](../../assets/QA/tkben-open-validation-debt-20260925/hf-provider-flow-evidence.json). | HIGH |

### Current release gate

The current application candidate at `21ed4ff6d32fceb5b8e6521e96b1ac9dcd0c75d2`
was validated from a clean `develop` tip. The local release suite passed and the
candidate-specific hosted-CI result is recorded in the [T5-06 QA
record](../../assets/QA/tkben-t5-06-current-candidate-20260925/README.md) after
the evidence commit is pushed.
Release readiness remains PARTIAL because this task did not synchronize `main`,
create an annotated tag, or publish a release. T4-03 is still BLOCKED pending
an already-authorized gated Hugging Face repository; physical low-disk recovery
remains PARTIAL pending a host-authorized bounded volume.

## Validation Campaign Roadmap

Last updated: 2026-09-25

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
| T2-01 | yes | yes | PASS | V-20260924: CSV and XLSX upload/persistence, UI CSV upload and catalogue visibility, and explicit `.xls` rejection passed. The upload API accepted 25 MiB at the configured boundary, returned 413 at 25 MiB + 1 byte, and a 24 MiB CSV persisted as ready. Current focused dataset route/storage tests passed 9/9; prior API E2E passed 7/7. Physical low-disk recovery remains separate validation debt. See the [Dataset and Settings QA record](../../assets/QA/tkben-t2-01-t4-02-t5-02-dataset-settings-validation-20260924/README.md) and [dataset storage QA record](../../assets/QA/tkben-data-upload-storage-20260924/README.md). |
| T2-02 | yes | yes | PASS | V-20260922 at base revision `0993e900d44897975f495dc6a7f7acd520d6a70a`: all six metric families were selected, analyzed, persisted, rendered in the populated dashboard, and restored after reload; metric unit tests passed 78/78 and dataset-analysis API E2E passed 1/1. See the [T2-02 QA record](../../assets/QA/tkben-t2-02-dataset-metric-families-20260922.md). |
| T2-03 | yes | yes | PASS | V-20260923 at implementation revision `de1aaee6`: the controlled four-document dataset included an exact duplicate pair, URL/email/HTML structure, and an empty document; selected keys and aggregates matched the metric contract, and the populated report restored after reload. Focused Chrome E2E passed 1/1, metric unit tests passed 78/78, and the dataset/report were removed. See the [T2-03 QA record](../../assets/QA/tkben-t2-03-dataset-quality-structure-compression-20260923.md). |
| T2-04 | yes | yes | PASS | V-20260924: UI upload, same-filename replacement, single catalog entry, canonical artifact content, full official-launcher restart persistence, post-restart report/vocabulary rendering, and UI deletion of catalog/report/artifact passed. Focused tokenizer E2E passed 7/7 enabled cases; the separate post-restart E2E passed 1/1. See the [T2-04/T2-05/T5-03 Tokenizers QA record](../../assets/QA/tkben-t2-04-t5-03-tokenizer-validation-20260924/README.md). |
| T2-05 | yes | yes | PASS | V-20260924 reran the local 1,207-entry custom-tokenizer report: persisted report reopened, API pages returned 500/500/207 contiguous entries, and the populated UI exercised first/middle/final navigation and four viewport widths. Focused tokenizer E2E passed; see the [T2-05 QA record](../../assets/QA/tkben-t2-05-tokenizer-report-vocabulary-20260923.md) and [combined Tokenizers QA record](../../assets/QA/tkben-t2-04-t5-03-tokenizer-validation-20260924/README.md). |
| T2-06 | yes | yes | PASS | V-20260923 at `f736f938`: opt-in Chrome E2E created a local report through the populated wizard, verified the persisted report via API, and reloaded the same report ID with populated charts. Cleanup verified report, tokenizer, and dataset removal; the combined Cross Benchmark and benchmark API E2E set passed 9/9. See the [T2-06 QA record](../../assets/QA/tkben-t2-06-cross-benchmark-wizard-20260923.md). |
| T2-07 | yes | yes | PASS | V-20260923 at `cc91fec1`: opt-in Chrome E2E reached API progress >=20%, cancelled to terminal `cancelled` with no report, restored Run, then completed and rendered an immediate two-document rerun. See the [T2-07 QA record](../../assets/QA/tkben-t2-07-benchmark-cancellation-20260923.md). |
| T3-01 | yes | yes | PASS | V-20260923: three comparable 1,000-document Windows runs recorded throughput with 95% intervals, p50/p95/p99 latency (80 observations/run), phase timings, peak RSS, and memory delta. Raw observations and runtime/hardware profile are in the [benchmark validation campaign QA record](../../assets/QA/tkben-benchmark-validation-campaign-20260923/README.md); samples describe this host and do not establish general performance. |
| T3-02 | yes | yes | PASS | V-20260924: the official-launcher local campaign ran two custom tokenizers with `parallelism=1` and `parallelism=2`, verified persisted config and effective worker metadata (1/1 and 2/2), and recorded a two-worker runtime high-water mark. Both reports preserved tokenizer order, 20 raw observations per tokenizer, 23 per-document samples per tokenizer, and loaded successfully. See the [T3-02 parallelism closure QA record](../../assets/QA/tkben-t3-02-parallelism-closure-20260924/README.md). |
| T3-03 | yes | yes | PASS | V-20260923: live persisted report search, 25/1 pagination, inline tag save/reload, confirmation, and physical deletion passed against 26 current-schema reports. See the [T3-03/04 QA record](../../assets/QA/tkben-t3-03-04-cross-benchmark-workflows-20260923.md). |
| T3-04 | yes | yes | PASS | V-20260923: live baseline/delta and data-table rendering, baseline reload, clone configuration, and visualization/order/customization persistence passed. See the [T3-03/04 QA record](../../assets/QA/tkben-t3-03-04-cross-benchmark-workflows-20260923.md). |
| T3-05 | yes | yes | PASS | V-20260923: original populated dataset/tokenizer/benchmark PDF gate passed. V-20260925 expanded benchmark PDF review to all eight chart forms and a 1,207-entry tokenizer report exported to five pages; `pdfinfo`, every Poppler render, and visual label/layout review passed. See the [open validation debt QA record](../../assets/QA/tkben-open-validation-debt-20260925/README.md) and [T3-05 QA record](../../assets/QA/tkben-t3-05-dashboard-pdf-exports-20260923.md). Existing export contract and route checks remain as recorded there. |
| T4-01 | yes | yes | PASS | V-20260925: supplied key authenticated to Hub; public `bert-base-uncased` discovery, download, persisted report 1 (vocabulary size 30,522), rendered reload, and cleanup passed. See the [HF provider evidence](../../assets/QA/tkben-open-validation-debt-20260925/hf-provider-flow-evidence.json), [authentication evidence](../../assets/QA/tkben-open-validation-debt-20260925/hf-provider-auth-evidence.json), and [key-storage evidence](../../assets/QA/tkben-open-validation-debt-20260925/hf-key-storage-evidence.json). |
| T4-02 | yes | yes | PASS | V-20260924: anonymous public `wikitext/wikitext-2-v1` download persisted 29,119 documents across an official-launcher restart; temporary source files were absent after import. One source and ordinary free-space conditions only; see the [Dataset and Settings QA record](../../assets/QA/tkben-t2-01-t4-02-t5-02-dataset-settings-validation-20260924/README.md). |
| T4-03 | yes | no | BLOCKED | The supplied key authenticated to Hub, but no authorized gated download succeeded across three discovered candidates; no gated report or access-term acceptance occurred. See the [HF provider evidence](../../assets/QA/tkben-open-validation-debt-20260925/hf-provider-flow-evidence.json). |
| T4-04 | yes | yes | PASS | V-20260925: isolated PostgreSQL 18.6 initialization reached Alembic head `0005_managed_job_lifecycle`; concurrent initialization, focused migration coverage (10/10), SQLite/PostgreSQL UI equivalence, PostgreSQL restart recovery, and rollback/retry after an injected mid-upload failure passed. The host listener at 127.0.0.1:5433 was not contacted. See the [PostgreSQL runtime validation handoff](../../assets/QA/tkben-postgresql-runtime-validation-20260925/HANDOFF.md). |
| T5-01 | yes | yes | PASS | Real Linux restart E2E kept the same active-job ID addressable as terminal `failed` with an interruption reason; a completed upload job and its 25,000-document dataset remained persisted. |
| T5-02 | yes | yes | PASS | V-20260924: populated/no-match Dataset and Settings Data/empty Keys states had no document-level horizontal overflow at 1920x1080, 1440x900, 1024x768, and 390x844. Dataset manager ArrowRight navigation, Escape/focus return, and Settings route-tab keyboard navigation passed. See the [Dataset and Settings QA record](../../assets/QA/tkben-t2-01-t4-02-t5-02-dataset-settings-validation-20260924/README.md). |
| T5-03 | yes | yes | PASS | V-20260923 covered the populated Cross Benchmark view and report-manager, clone, and customize dialogs at 1920x1080, 1440x900, 1024x768, and 390x844; Escape and focus return passed. V-20260924 completed Tokenizers empty/loading/error/populated/report states and Tokenizer Manager bounds plus ArrowRight/End/Escape/focus return at the same four sizes. V-20260924 also closed the Dataset/Settings responsive slice under T5-02. See the [T3-03/04 QA record](../../assets/QA/tkben-t3-03-04-cross-benchmark-workflows-20260923.md), [Tokenizers QA record](../../assets/QA/tkben-t2-04-t5-03-tokenizer-validation-20260924/README.md), and [Dataset and Settings QA record](../../assets/QA/tkben-t2-01-t4-02-t5-02-dataset-settings-validation-20260924/README.md). |
| T5-04 | yes | yes | PASS | V-20260923: official Windows launcher streamed a 10,000-document run to visible 20% progress; cancellation reached terminal `cancelled` with no report saved, eight backend RSS samples were collected, and a two-document rerun completed and rendered without browser or HTTP errors. See the [benchmark validation campaign QA record](../../assets/QA/tkben-benchmark-validation-campaign-20260923/README.md). Browser heap was not instrumented; the gate's live progress, cancellation, available resource metrics, and rerun checks passed. |
| T5-05 | — | — | OUT_OF_SCOPE | Release support and release validation target Windows x64. The Ubuntu 26.04 manual run is diagnostic evidence only; macOS and hosted Ubuntu runtime validation are not planned or required. Hosted CI remains tracked separately under T5-06. |
| T5-06 | yes | yes | PARTIAL | V-20260925: clean application revision `21ed4ff6d32fceb5b8e6521e96b1ac9dcd0c75d2` passed the local release suite; candidate-specific hosted CI is pending after the evidence commit and will be recorded in the [T5-06 QA record](../../assets/QA/tkben-t5-06-current-candidate-20260925/README.md). No `main` synchronization, annotated tag, or public release was performed, so this slice remains PARTIAL. Historical release/tag evidence remains in the [release audit](../../assets/QA/tkben-open-validation-debt-20260925/release-audit.json). |

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
- The Ubuntu 26.04 manual run is diagnostic evidence only; macOS and Linux are
  outside the supported Windows x64 release target. Their platform validation
  is not an open release gate. Hosted CI runner OS remains separate from product
  platform support.
- Current local gates: 212 backend unit tests passed; Ruff passed; BasedPyright
  reported 0 errors and 1,953 warnings; frontend lint and all 60 frontend unit
  tests passed; 17 launcher contract tests passed; the manual Windows test-menu
  run passed 378 tests with 4 skips before the final launcher contract-test
  additions. Hosted CI is not inferred from these local results.
