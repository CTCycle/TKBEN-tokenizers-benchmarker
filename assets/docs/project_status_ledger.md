# Project Status Ledger
Last updated: 2026-10-01

This is the canonical source of truth for current implementation status,
validation gates, unresolved limitations, and validation history. The
standalone QA archive was reviewed and consolidated into this ledger and the
topic documents on 2026-09-26. Raw logs, screenshots, generated PDFs,
disposable databases, fixtures, and duplicate reports are not part of the
maintained documentation set.

## Maintenance rules

Future coding and validation agents must:

1. inspect this ledger before substantial implementation or validation work;
2. update affected component and gate rows after meaningful evidence;
3. keep current status separate from historical failures and intermediate
   states;
4. never mark a component or gate VALIDATED without evidence at the stated
   scope;
5. retain a limitation when it is the only record of an unresolved defect,
   blocked check, or deferred validation;
6. keep this ledger synchronized with the repository and the topic documents;
7. link to maintained documentation or executable tests instead of adding raw
   logs or duplicate reports.

## Status taxonomy

| Status | Meaning |
| --- | --- |
| VALIDATED | Implemented and confirmed through meaningful testing at the stated scope. |
| WORKING | Believed to work from limited or indirect evidence, but the stated component scope is not fully validated. |
| PARTIAL | Implemented, but incomplete, degraded, or confirmed for only part of the expected behavior. |
| PASS | A validation slice passed at its stated scope. |
| FAILED | An attempted validation run failed; the failure may be historical if a later run fixed and revalidated it. |
| BLOCKED | Validation cannot currently complete because of an external dependency, unavailable credential/service, or host constraint. |
| DEFERRED | An otherwise known check was intentionally postponed; it is not silently treated as passed. |
| OUT_OF_SCOPE | Explicitly excluded from the supported product or release target. |
| NOT_IMPLEMENTED | The expected capability is absent from the repository. |
| DEPRECATED | Retained for compatibility while new clients use another contract. |

## Current truth

- Local startup, the primary API contract, SQLite/Alembic persistence,
  dataset analysis, custom-tokenizer lifecycle, benchmark reports, exercised
  UI routes, and the local/hosted quality gates are VALIDATED at the scopes
  recorded below.
- Windows portable bootstrap and launcher behavior are VALIDATED on the
  exercised Windows host, including the privilege-isolated denied process-
  termination branch. The runner's unrelated protected-cache ACL warnings
  remain environment limitations, not reproduced product defects.
- Public Hugging Face discovery, download, report persistence, rendered
  reload, and cleanup are VALIDATED. Authorized gated-repository access is
  still BLOCKED; no gated report was produced.
- Isolated PostgreSQL 18.6 initialization, migration, concurrency,
  SQLite-equivalent UI behavior, restart recovery, and injected-failure
  rollback/retry are VALIDATED. This does not claim coverage of the
  pre-existing host PostgreSQL listener or every deployment topology.
- Physical filesystem exhaustion remains PARTIAL and DEFERRED as optional
  validation debt. Controlled `SQLITE_FULL` cleanup and retry passed, but the
  host-authorized bounded-volume scenario could not be provisioned.
- Containerized deployment and binary packaging are NOT_IMPLEMENTED. The
  supported release model is source-only Windows x64.

## Consolidated validation history

The following campaign index replaces the former standalone reports. Campaign
names preserve traceability to the reviewed material; the durable outcome is
recorded here rather than in a copied execution log.

| Evidence period | Consolidated campaigns | Gates represented | Durable outcome |
| --- | --- | --- | --- |
| V-20260920 to V-20260921 | Python 3.14.7 upgrade; startup/loading; launcher startup; initial Tier 1 run | T0-01, T1-01 | PASS for the focused Windows runtime upgrade, frontend-first loading states, launcher warm/repair paths, health/readiness, route smoke, and the initial foundation slice. Provider, PostgreSQL, hosted-CI, and full maintenance coverage were explicitly outside these runs. |
| V-20260922 | Partial-gate closure; Tier 1 runtime settings, key management, and catalog filtering/races | T0-02, T0-03, T1-02 through T1-05, T5-01 | PASS for clean and warm bootstrap, all 13 maintenance routes, launcher conflict/race/permission branches, runtime settings boundaries and conflicts, synthetic key lifecycle, and populated Dataset/Tokenizer filter and stale-result races. The missing live HF key and runner ACL restrictions were recorded as skipped/environment boundaries. |
| V-20260923 | Dataset metrics and quality; tokenizer vocabulary; Cross Benchmark wizard; benchmark campaign; report workflows; dashboard/PDF exports; cancellation | T2-02, T2-03, T2-05 through T2-07, T3-01, T3-03 through T3-05, T5-03, T5-04 | PASS for the six metric families, controlled quality/structure/compression data, 1,207-entry vocabulary paging, populated wizard/report workflows, report management and customization, PDF smoke, and 10,000-document progress/cancel/rerun behavior. Performance samples remain host-specific. |
| V-20260924 | Dataset/Settings matrix; public dataset download; tokenizer lifecycle and responsive states; upload storage; parallelism closure | T2-01, T2-04, T2-05, T3-02, T4-02, T5-02, T5-03 | PASS for CSV/XLSX boundaries, public Wikitext import/restart cleanup, custom-tokenizer replacement/restart/report/delete, four-viewport Dataset/Settings/Tokenizers checks, and actual tokenizer worker overlap at parallelism 2. Controlled `SQLITE_FULL` cleanup/retry passed; physical filesystem exhaustion remained untested. |
| V-20260925 | Open validation debt; PostgreSQL runtime handoff; validation-gate recheck; current-candidate/release closure | T3-05, T4-01, T4-03, T4-04, T5-06 | PASS for all eight supported benchmark PDF chart forms, the five-page 1,207-entry tokenizer PDF, public HF flow, isolated PostgreSQL runtime equivalence, the current local suite, and release v4.5.0. T4-03 gated HF access remains BLOCKED and the physical low-disk scenario remains PARTIAL/DEFERRED. |
| V-20260928 to V-20260929 | Post-release data-root refactor; launcher bytecode-cache hardening | T0-02 (delta), T0-01 (recheck baseline) | The post-release `resources`-to-`data` data-root rename, bytecode-cache clearing fix, and launcher contract expansion (19 to 21 checks) are recorded here as code and documentation deltas after the v4.5.0 tag; the launcher contract and data-path assertions reflect the new layout at the current checkout. No fresh validation run beyond the tracked contract suite has been executed for this delta; the recorded release gate remains at `f8dee1d`. |

### Reconciled status transitions

- The initial Tier 1 `PARTIAL` results for runtime settings, key management,
  and catalog filtering were superseded by the 2026-09-22 boundary,
  persistence, and race evidence; the current T1-02, T1-03, and T1-05 rows
  are PASS. The broader `configuration.runtime-settings` component remains
  WORKING because its component scope is broader than that gate.
- T3-02 changed from PARTIAL to PASS after the 2026-09-24 campaign observed
  two active workers for requested parallelism 2 and verified persisted
  requested/effective metadata, ordering, raw observations, and reload.
- T3-05 changed from a limited/working export check to PASS after all eight
  benchmark chart forms and the 1,207-entry tokenizer PDF were rendered and
  reviewed for page count, labels, axes, and clipping.
- Early T4-01/T4-02/T4-04 and T5-02/T5-03 records were untested or bounded;
  later public-provider, public-dataset, responsive, and isolated-PostgreSQL
  evidence supersedes those current statuses. Their earlier limits remain
  only as this history, not as active failures.
- T5-06 was PARTIAL during the pre-release audit because synchronization,
  tagging, and publication were not yet complete. It became PASS after the
  exact release SHA passed local checks and hosted CI, `main` and `develop`
  were synchronized, tag `v4.5.0` was published, and the public source-only
  release was verified.
- PostgreSQL migration attempts initially FAILED on an Alembic check-
  constraint expression and then on a dependent tokenizer primary-key
  alteration. Both defects were fixed; the focused migration suite passed
  10/10 and the isolated runtime campaign passed. These failures are retained
  as remediation history so they are not mistaken for the current state.
- Two duplicate benchmark snapshots reported different cancellation timing
  (about 217 ms and 526 ms). Both reached the required terminal outcome; the
  non-gating timing difference is intentionally not promoted to a performance
  claim.
- In the benchmark cancellation run, the API reached at least 20% while the
  browser view displayed 5% at the captured moment. The gate is based on the
  API/terminal evidence, and the display sampling difference is a known
  observation rather than a failed cancellation gate.

## Current component ledger

| Component | Status | Current evidence and boundary | Last validated |
| --- | --- | --- | --- |
| architecture.canonical-ownership | VALIDATED | Architecture boundary tests and current ownership docs agree; remaining canonicalization work is ISSUE-003. | 2026-09-20 |
| runtime.startup.local-webapp | VALIDATED | Windows bootstrap and Linux diagnostic startup reached health, proxy, route, and restart-recovery checks. Populated dashboards and responsive coverage are separate gates. | 2026-09-22 |
| runtime.windows-launcher | VALIDATED | Clean/warm bootstrap, stamps, build repair, conflicts, maintenance menu, cleanup, quoted-process handling, and localized denied termination were exercised. The 2026-09-28 post-release bytecode-cache clearing fix and the repository-`data` default-root rename changed launcher behavior after the recorded validation; the expanded 21/21 launcher contract suite now asserts the new layout. | 2026-09-22 |
| configuration.runtime-settings | WORKING | All 16 settings, bounds, sparse persistence, restart, conflict, reset, and new-work effects passed; the component deliberately retains broader WORKING scope. | 2026-09-22 |
| runtime.managed-job-lifecycle | VALIDATED | Job persistence, terminal failure on restart, cancellation conflicts, and retained completed data passed. Resumption/checkpointing is intentionally unsupported. | 2026-09-22 |
| backend.api-contracts | VALIDATED | Route/unit/OpenAPI coverage and restart-addressable job behavior passed. Provider-dependent routes remain separately bounded. | 2026-09-25 |
| persistence.sqlite-alembic | VALIDATED | Alembic head `0005_managed_job_lifecycle`, schema/rollback/cascade contracts, restart reconciliation, and controlled `SQLITE_FULL` cleanup/retry passed. | 2026-09-25 |
| persistence.postgresql-runtime | VALIDATED | Isolated PostgreSQL 18.6 concurrency, migration, UI equivalence, restart, rollback, and retry passed. The existing host listener was untouched. | 2026-09-25 |
| data.dataset-import-and-analysis | VALIDATED | CSV/XLSX import, `.xls` rejection, size boundary, analysis metrics, reload, and public Wikitext import passed. Physical host exhaustion is separate debt. | 2026-09-24 |
| data.custom-tokenizer-storage | VALIDATED | Canonical artifact replacement, catalog identity, restart persistence, report/vocabulary rendering, and deletion passed. | 2026-09-24 |
| integration.huggingface-discovery-and-download | PARTIAL | Public `bert-base-uncased` flow passed with vocabulary 30,522; three gated candidates failed download, so no gated report exists. | 2026-09-25 |
| benchmark.execution-and-reporting | VALIDATED | Admission, reports, bounded worker concurrency, progress/cancel, immediate rerun, metadata, reload, and deletion passed. Performance samples are host-specific. | 2026-09-24 |
| benchmark.dashboard-and-pdf-export | VALIDATED | Dashboard data/customization and all eight supported benchmark PDF chart forms plus the five-page tokenizer PDF passed rendered review. | 2026-09-25 |
| ui.route-shell-and-empty-states | VALIDATED | Dataset, Tokenizers, Cross Benchmark, Settings, loading/error/empty, and populated states passed the exercised route and responsive checks. | 2026-09-25 |
| ui.startup-readiness | VALIDATED | Offline, slow, terminal-failure, retry, and recovered frontend-first states passed in the in-app browser. | 2026-09-21 |
| ui.dataset-dashboard | VALIDATED | Metric families, persisted dashboard reload, malformed optional histogram fallback, and export action passed. | 2026-09-23 |
| ui.settings-page | VALIDATED | Typed settings, conflicts, reset, key lifecycle, masked credential storage, keyboard navigation, and responsive states passed. | 2026-09-25 |
| ui.cross-benchmark-workflow | VALIDATED | Wizard, report manager, tags, baseline/delta, clone/customize, tables, ordering, persistence, and responsive dialogs passed. | 2026-09-24 |
| ui.tokenizer-report-and-vocabulary | VALIDATED | 1,207-entry report persistence, contiguous API paging, rendered pagination, restart report, and four-viewport route states passed. | 2026-09-24 |
| deployment.source-only-local | VALIDATED | v4.5.0 source-only release, local suite, exact hosted CI, synchronized branches, tag, and public release passed. | 2026-09-25 |
| deployment.windows-portable-bootstrap | VALIDATED | Pinned Python/Node/uv bootstrap, dependency repair, stamped build reuse, migration, and launch passed on the exercised Windows host. | 2026-09-22 |
| deployment.cross-platform-manual | OUT_OF_SCOPE | Ubuntu manual execution was diagnostic only; macOS and hosted runtime support are not release targets. | 2026-09-22 |
| deployment.containerized | NOT_IMPLEMENTED | No active container deployment configuration exists. | — |
| deployment.binary-packaging | NOT_IMPLEMENTED | Releases are source-only; no installer or binary workflow exists. | — |
| test-infrastructure.local-quality-gates | VALIDATED | Release revision passed compile, Ruff, BasedPyright with 0 errors, Alembic initialization, 517 backend tests, OpenAPI smoke, 63 frontend tests, lint, and build. | 2026-09-25 |
| test-infrastructure.hosted-ci-and-release-evidence | VALIDATED | Hosted CI run 36158666979 passed both jobs for the release SHA; tag and public release point to that SHA. | 2026-09-25 |
| api.tokenizers.settings-compatibility | DEPRECATED | The legacy compatibility endpoint remains covered by contract/OpenAPI checks; new clients use `/api/settings`. | 2026-09-20 |

Related implementation and test documentation is indexed in
[project_index.md](project_index.md). The most relevant topic documents are
[persistence](architecture/persistence.md), [startup](runtime/startup.md),
[release](runtime/release.md), and [testing and quality](coding/testing_and_quality.md).

## Validation gate roadmap

The gate table records the latest known state of every stable validation ID.
`PASS` is evidence at the slice scope; `PARTIAL`, `BLOCKED`, `DEFERRED`, and
`OUT_OF_SCOPE` are intentionally not treated as PASS.

| Gate | Status | Latest durable outcome or remaining action |
| --- | --- | --- |
| T0-01 | PASS | Revision/quality baseline and release-target assumptions were accepted and rechecked in the current local campaign. |
| T0-02 | PASS | At `e864197c`, clean/warm bootstrap, malformed/missing stamps, stale build, repairs, invalid ports, redirected conflicts, reacquisition race, and the real localized `Accesso negato` termination-denial branch passed; the launcher contract suite then stood at 19/19. The post-release data-root refactor expanded the contract suite to 21/21 at the current checkout (`a84b9c1`), adding repository-`data` default-root and log-root assertions; the release-tagged evidence remains the 19/19 record. |
| T0-03 | PASS | All 13 maintenance-menu routes, install profiles, expected update refusal, destructive decline/approval, cleanup, uninstall, and Kill All passed. |
| T1-01 | PASS | Startup E2E 2/2 and shell recovery after persistence restart passed. |
| T1-02 | PASS | At `41f265a`, all 16 settings, 38 invalid cases, 25 valid boundaries, decimal constraints, tokenizer relations, sparse persistence, restart, 409 conflict, reset, and downstream effects passed. |
| T1-03 | PASS | At `09805b7`, synthetic Settings Keys lifecycle, masking, duplicate/active-delete/reveal policies, single-active switching, ciphertext separation, and cleanup passed; live supplied-key coverage was SKIPPED because no test key was available. |
| T1-04 | PASS | Focused migration/persistence/settings contracts and a runtime override surviving official-launcher restart passed; PostgreSQL is tracked separately under T4-04. |
| T1-05 | PASS | At `993e52e8`, populated Dataset/Tokenizer filter matrices, exact numeric boundaries, combined/no-match/reset, completion-order races, and stale discovery result/error handling passed. |
| T2-01 | PASS | CSV/XLSX upload and persistence, UI catalogue visibility, `.xls` rejection, 25 MiB/25 MiB+1 boundary, and ready-state import passed. Physical disk exhaustion remains separate debt. |
| T2-02 | PASS | At `0993e900`, all six dataset metric families were selected, analyzed, persisted, rendered, and restored; metric unit tests 78/78 and API E2E 1/1 passed. |
| T2-03 | PASS | At `de1aaee6`, a controlled duplicate/URL/email/HTML/line-break/empty-document dataset produced the expected quality, structure, and compression metrics and reloaded correctly. |
| T2-04 | PASS | Same-name custom-tokenizer replacement, one catalog row, canonical seven-token artifact, launcher restart, report/vocabulary rendering, and deletion passed. |
| T2-05 | PASS | The 1,207-entry tokenizer report reopened; API pages 500/500/207 were contiguous; first/middle/final UI paging and four viewport rendering passed. |
| T2-06 | PASS | At `f736f938`, the populated Cross Benchmark wizard created a local report, verified persistence/reload, and cleaned its dataset/tokenizer/report; combined E2E 9/9. |
| T2-07 | PASS | At `cc91fec1`, API progress reached at least 20%, cancellation became terminal with no report, and an immediate two-document rerun completed and rendered. Browser sampling showed 5% at one capture; the API and terminal evidence remain the gate basis. |
| T3-01 | PASS | Three comparable 1,000-document runs recorded throughput, latency percentiles from 80 observations per run, phase timing, RSS, and memory delta. Samples describe this host and are not a general performance guarantee. |
| T3-02 | PASS | At `b087db60`, requested/effective parallelism 1/1 and 2/2, two-worker overlap, persisted configuration, ordering, observations, per-document samples, and report reload passed. |
| T3-03 | PASS | Report search, 25/1 pagination, tags, confirmation, and physical deletion passed against populated current-schema reports. |
| T3-04 | PASS | Baseline/delta views, data tables, reload, clone configuration, visualization overrides, order, and customization persistence passed. |
| T3-05 | PASS | Dataset/tokenizer/benchmark export smoke and the expanded all-eight-form benchmark review plus five-page 1,207-entry tokenizer PDF passed `pdfinfo`, rendering, and visual legibility checks. |
| T4-01 | PASS | Supplied key authentication, public `bert-base-uncased` discovery/download/report/reload/cleanup passed; vocabulary size was 30,522. |
| T4-02 | PASS | Anonymous public Wikitext import persisted 29,119 documents across restart and removed temporary source files. |
| T4-03 | BLOCKED | Three discovered gated candidates all failed download; no access-term acceptance or gated report exists. The external authorization check is DEFERRED by the user and non-blocking. |
| T4-04 | PASS | Isolated PostgreSQL 18.6 reached Alembic 0005; concurrent initialization, migration 10/10, UI equivalence, restart recovery, injected rollback, and valid retry passed. |
| T5-01 | PASS | Restart reconciled active work to addressable terminal failure while preserving completed upload job and its 25,000-document dataset. |
| T5-02 | PASS | Dataset/Settings populated, no-match, empty Keys, four viewport sizes, keyboard navigation, Escape, and focus return passed without document-level horizontal overflow. |
| T5-03 | PASS | Cross Benchmark, Tokenizers empty/loading/error/populated/report states, manager bounds, four viewport sizes, and keyboard/focus behavior passed. |
| T5-04 | PASS | Official launcher streamed visible progress for 10,000 documents, cancellation produced no report, eight backend RSS samples were collected, and a two-document rerun rendered without browser/HTTP errors. Browser heap was not instrumented. |
| T5-05 | OUT_OF_SCOPE | Release support is Windows x64 source-only. Ubuntu was diagnostic; macOS and hosted runtime support are not planned. |
| T5-06 | PASS | Release commit `f8dee1da9084138bd52f3d08437269d3821ba5d6`, hosted CI [36158666979](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/runs/36158666979), synchronized `main`/`develop`, annotated tag `v4.5.0`, and public source-only release were verified. |

## Open issues and validation debt

| ID or component | Status | Durable limitation and follow-up |
| --- | --- | --- |
| ISSUE-002: deployment.network-auth-boundary | OPEN / HIGH | Network-hosted deployment still requires an external authentication and authorization boundary before exposing key-management or destructive routes. This is outside the supported local/source-only release model. Validate the external boundary before any network deployment. See [deployment](runtime/deployment.md#constraints) and [configuration security controls](runtime/configuration.md#security-controls). |
| ISSUE-003: architecture.canonical-ownership | OPEN / MEDIUM | Environment-derived paths, duplicate metric representations, export dictionaries, handwritten frontend contracts, catalog metadata, and toolchain declarations still have planned canonicalization follow-ups. No current regression was reproduced. See [canonical-source remediation](architecture/canonical_source_remediation.md#remaining-canonicalization-work). |
| data.large-file-and-disk-exhaustion | PARTIAL | Controlled `SQLITE_FULL` cleanup and same-name retries passed for upload and dataset-backed import. Host authorization denied bounded-volume creation before a physical filesystem-full scenario could run; no VHD artifact remains. The optional recheck is DEFERRED until a safe host-authorized volume is available. |
| integration.huggingface-discovery-and-download / T4-03 | PARTIAL | Public provider flow passed, but no authorized gated repository completed download/report generation. The gate is BLOCKED and the external recheck is DEFERRED; rerun only when an already-authorized gated repository is explicitly selected. |
| deployment.containerized | NOT_IMPLEMENTED | No container deployment is part of the current product scope. |
| deployment.binary-packaging | NOT_IMPLEMENTED | The supported artifact is the source-only GitHub Release; do not infer installer or executable support. |

## Release gate

The current published release commit is
`f8dee1da9084138bd52f3d08437269d3821ba5d6`. Local release checks and hosted
CI run [36158666979](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/runs/36158666979)
passed for that exact SHA. Annotated tag `v4.5.0` points to the release commit,
and the [public source-only release](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/releases/tag/v4.5.0)
was published. `main` and `develop` were synchronized at that release; the
branches have since advanced past the tag with the post-release data-root
rename, bytecode-cache hardening, and formatting commits (currently
`a84b9c1`). Release readiness for the tag is VALIDATED, with T4-03 and
physical low-disk recovery remaining independently bounded as recorded above;
the post-release delta is tracked as a recorded code/documentation delta and
is not part of the tagged release claim.

## Resolved and historical findings

| Finding | Current state | Durable provenance |
| --- | --- | --- |
| ISSUE-001: active jobs disappeared on restart | Resolved for visibility and lifecycle state. Migration 0005 persists job metadata and startup reconciles pending/running work to failed with an interruption reason; job resumption remains intentionally unsupported. | T5-01 restart evidence and [persistence](architecture/persistence.md). |
| Initial PostgreSQL migration failures | Resolved. The Alembic check-constraint expression and dependent tokenizer-key alteration were corrected; focused migration coverage is 10/10. | T4-04 campaign history and [persistence](architecture/persistence.md). |
| Architecture P1/P2/P3 ownership findings | Resolved and covered by the architecture boundary contract; remaining canonicalization follow-ups are ISSUE-003. | [architecture review](architecture/architecture_review.md#findings) and [canonical-source remediation](architecture/canonical_source_remediation.md). |
| Legacy configuration/cache/tokenizer/report/dashboard paths | Removed or canonicalized; incompatible persisted rows fail explicitly rather than being silently adapted. | [canonical-source remediation](architecture/canonical_source_remediation.md). |
| v4.4.0 report tags, cloning, dashboard persistence, chart controls, and launcher cleanup | Included in the v4.4.0 source-only release and superseded by the v4.5.0 release gate above. | [release procedure](runtime/release.md#v440-release-notes). |

## Documentation consolidation record

The 2026-09-26 consolidation inspected all 2,504 files under the former
`assets/QA` tree, including 27 Markdown reports, JSON/CSV/XLSX evidence,
logs, screenshots, PDFs, SQLite databases, fixtures, and a disposable
PostgreSQL cluster. Meaningful results, limitations, status transitions,
historical failures, and release traceability are represented in this ledger
and the linked topic docs. No standalone QA artifact is intentionally retained
outside `assets/docs`; future test harness output paths may recreate disposable
files only when a new validation run explicitly requests them.
