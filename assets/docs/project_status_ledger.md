# Project Status Ledger
Last updated: 2026-10-30

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

## Current truth

- The previously validated local startup baseline, primary API contract,
  SQLite/Alembic persistence, dataset analysis, custom-tokenizer lifecycle,
  benchmark reports, exercised UI routes, and local/hosted quality gates
  retain the evidence recorded below. The 2026-10-23 audit exercised the
  validation wizard, tokenizer preview, benchmark report, dashboard ordering,
  isolated SQLite persistence, and production launcher on the current tree.
  The pointer-reorder E2E timed out in two full runs before its initial
  preference load and pointer movement were synchronized; the focused test and
  final full runner passed after that test-only correction.
- Windows portable bootstrap and previously validated launcher behavior are
  VALIDATED on the exercised Windows host, including the privilege-isolated
  denied process-termination branch. The current `-Launch` startup and console
  Ctrl+C lifecycle returned launcher exit code 0, stopped its owned backend,
  and freed its port; interactive occupied-port confirmation and redirected
  launch were not re-exercised, so the complete launcher gate remains PARTIAL.
- `v4.8.1` (backend `3.8.1`, frontend `2.8.1`) is the latest published
  release; see the release gate for its record. `v4.8.0` passed hosted CI run
  `40225854600` on release commit `955a151847db2af066d6f0ef2016ab8522b10115`.
- Public Hugging Face discovery, download, report persistence, rendered
  reload, and cleanup are VALIDATED. Authorized gated-repository access is
  still BLOCKED; no gated report was produced.
- Isolated PostgreSQL 18.6 initialization, migration, concurrency,
  SQLite-equivalent UI behavior, restart recovery, and injected-failure
  rollback/retry are VALIDATED. This does not claim coverage of the
  pre-existing host PostgreSQL listener or every deployment topology.
- Physical filesystem exhaustion remains PARTIAL and DEFERRED as optional,
  non-blocking validation debt. Controlled `SQLITE_FULL` cleanup and retry
  passed; the host-authorized bounded-volume scenario was not provisioned and
  is not required to claim the local source-only quality gate.
- Containerized deployment and binary packaging are NOT_IMPLEMENTED. The
  supported release model is source-only Windows x64.

## 2026-10-27 E2E audit (V-20261027, incomplete)

### Follow-up on the current uncommitted working tree

The existing working-tree edits add the Storage tab and update the settings
test inventory assertion; this audit did not change application or test code.
The launcher rebuilt the current frontend and reached readiness in 24.48 s;
a warm restart took 7.62 s. In the in-app browser, pointer selection of Storage
and Down from Runtime both selected and focused Storage and exposed its panel.
The former AUD-20261027-01 is **resolved at this navigation scope** on this
working tree; saving storage paths and restart behavior are not established by
that observation. The former AUD-20261027-02 is **resolved at collection and
settings-suite scope**: all three settings E2Es passed in 33.57 s against
isolated SQLite using installed Chrome. These cover settings boundaries,
persistence and effects on new operations, an SQLite switch, and refusal of
an unavailable PostgreSQL connection. They do not prove a working PostgreSQL
deployment or the complete E2E suite.

The five previously failing independent tests were rerun, all still failed in
117.18 s. Report-tag editing expects `Save tags`, while the rendered template
uses `Save`; benchmark cancellation expects removed page-local progress and
cancellation controls, while the actual controls are in the shared Jobs panel.
The tokenizer lifecycle locator resolves to two matching elements. The
dashboard checkbox and chart-axis expectations remain unresolved. The
benchmark test's cleanup also recorded a socket hang-up, so the rerun does not
establish its full cancellation/rerun workflow. No product failure is inferred
solely from these test failures.

As a separate live observation, a second browser session reloaded the shared
Jobs panel and displayed a running isolated benchmark. Clicking Cancel showed
Cancelling then Cancelled; SQLite recorded job `f9736218` as `cancelled`, no
report with that test run's name existed, and `integrity_check` returned `ok`.
After a backend restart the browser again displayed the cancelled job. This
validates one cancellation and persistence path, not immediate rerun,
interrupted-job reconciliation, or general concurrency. Relevant backend log
warnings were limited to deliberately invalid inputs in the earlier campaign;
the observed socket error has not been reproduced or attributed.

The overall recommendation remains **NO-GO for the requested comprehensive
gate**: the complete suite has not passed, the remaining failures need
diagnosis, and the untested areas below remain open. Re-run the full suite
after test issues are addressed and validate populated chart labels, restart
recovery, concurrency, supported database backends and authorized provider
flows. No application fixes were made during this follow-up.

The subsequent complete Chrome E2E run on a fresh isolated SQLite database
collected successfully and returned **45 passed, six failed, seven skipped**
in 274.67 s, with synthetic benchmark and tokenizer lifecycle opt-ins enabled.
The five failures described above persisted; the playground additionally saw
six tokenizer choices where its test expected five. Whether fixture residue or
application behavior caused the extra choice is unconfirmed. The benchmark
test again reported a socket hang-up in cleanup. Provider/credential and
restart-specific cases were among the seven skips. After the run the isolated
database had no pending/running jobs and no tokenizer rows. A direct invalid
SPA route redirected to Dataset, `/settings` returned HTML, and an unknown
API path and missing JavaScript asset returned 404. These observations do not
replace the missing workflow and environment coverage.

This is an initial, incomplete current-checkout audit, not a full release
certification. Windows 11 Pro 10.0.26300, checkout `3af63a9`, Python 3.14.7,
Node 22.23.1. The production launcher used an isolated `TKBEN_PATHS_CONFIG`
and SQLite data root under `assets/QA/audit-20261027`; it reached health and
SPA readiness in 13.94 seconds. Do not treat older validation as evidence of
the current checkout without revalidation. Ctrl+C through the automated shell
ended the launcher with exit code 1; its backend process exited and ports 5000
and 8000 were no longer listening. This does not verify graceful shutdown in a
normal interactive Windows console.

### Coverage inventory and current results

The four routed feature areas below are the functional denominator: Dataset,
Tokenizers, Cross Benchmark, and Settings. All four rendered, but **0/4 are
fully validated** by this audit. Rendering is not an end-to-end workflow pass.
For the five specifically observed browser scenarios below, four passed and
one failed (4/5); this denominator excludes all unexecuted scenarios and does
not imply coverage of every control or workflow.

| Feature, route, API/data boundary | Executed scenario and result | Remaining coverage |
| --- | --- | --- |
| Dataset `/dataset`; `/api/datasets`, managed jobs, SQLite and source files | PASS: empty catalog renders; add-dataset dialog and custom-upload tab open. | Browser upload, validation, reports, exports, filtering, persistence, failure/recovery and concurrency not manually completed. |
| Tokenizers `/tokenizers`; `/api/tokenizers`, jobs, SQLite and tokenizer artifacts, Hugging Face | PASS: empty catalog and playground entry render. | Browser upload, report, preview, deletion, persistence, external discovery/download and error/recovery not manually completed. |
| Cross Benchmark `/cross-benchmark`; `/api/benchmarks`, jobs, report DB, PDF export | PASS: empty report page and wizard render; Next is disabled without dataset/tokenizer. | Real benchmark, cancellation, reports, dashboard persistence, export, concurrent jobs and restart not manually completed. |
| Settings `/settings`; `/api/settings`, `/api/database`, `/api/keys`, path config and DB | FAIL: Storage has no visible navigation tab; keyboard navigation opens its panel with no selected tab (AUD-20261027-01). | Runtime-setting effects, database switching, key operations, storage save/restart and concurrent edits not manually completed. |
| Lifecycle and jobs; `/api/health`, `/api/jobs` | PASS at startup only; isolated SQLite `integrity_check` returned `ok` and Alembic head `0002_managed_job_created_at_index`. | Deep links, back/forward, shutdown, restart reconciliation, multiple sessions, cancellation and resource cleanup remain unvalidated. |
| API/security/integrations | Independent API-oriented E2E tests ran against isolated backend; details and limitations below. No intrusive exploitation performed. | Authentication boundaries, CORS, dependency advisories, PostgreSQL, failure injection and external services are not comprehensively tested. |
| Visual/performance | Desktop in-app browser inspected empty pages, wizard and settings; no browser console warnings/errors observed in that session. Startup readiness measured above. | Responsive populated states, accessibility, latency, memory, polling and prolonged-load profiling not completed. |

### Findings and remediation

**AUD-20261027-01 — Medium: Storage settings lack a navigation tab.** Open
Settings: Data, Tokenizers, Benchmarks, Runtime, Database and Keys are shown,
but no Storage tab. Click Runtime and press Down: the Storage panel appears,
while Runtime retains focus and no tab is selected. Expected: a visible,
keyboard-accessible Storage tab selects and labels its panel. Actual: pointer
users cannot select Storage and keyboard users enter an unrepresented tab.
The tab array contains `storage` in
`app/client/angular/app/pages/settings-page.component.ts:27`, but the tab
buttons in `settings-page.component.html:9` omit it while the panel exists at
line 154. Add the tab with matching ARIA linkage and test mouse and arrow-key
navigation, selected state, and storage save/restart behavior. Evidence:
`assets/QA/audit-20261027/settings-storage-keyboard.png` while retained.

**AUD-20261027-02 — High: E2E collection aborts before any test runs.** Run
`python -m pytest app/tests/e2e -q` with the locked test dependencies. The
collection of `test_settings_flows.py` fails at line 387: generated
`SETTINGS_FIELDS` contains 17 entries but the hard-coded assertion expects 16.
Expected: the standard E2E suite collects and runs all settings scenarios.
Actual: the entire suite stops at collection. Reconcile the inventory with the
current 17-field contract and add a meaningful contract-driven check; rerun
the full suite including settings. This is a test-gate failure, not evidence
that all 17 UI controls work.

### Checks, limitations and release decision

Backend unit tests: 508 passed, eight deprecation warnings; Ruff passed.
Frontend lint passed, 77 unit tests passed, and production build passed.
BasedPyright reported zero errors and 2,173 warnings. The initial E2E
collection failed as above. With the settings file excluded, the first run
reported 17 passed, seven skipped and 31 setup errors because the locked
Playwright version's Chromium headless executable was missing. Those setup
errors are environment limitations, not confirmed product failures. An
installed Chrome-channel rerun with the backend and test process using the same
isolated paths completed with **43 passed, five failed, seven skipped** in
218 seconds, excluding the uncollectable settings file. The five failures
were: benchmark cancellation test did not observe its progress indicator;
report tag save button locator timed out; dashboard customization checkbox
locator timed out; live report axis-label locator timed out; and tokenizer UI
lifecycle text locator matched two elements. These are unresolved test or
product discrepancies, not independently confirmed defects. Investigate each
against actual rendered state and make fixtures/selectors deterministic before
claiming the affected workflow passes. The first Chrome attempt used a
different test-process path from the isolated backend, yielding ten failures;
its results are not comparable. It created one uniquely named dataset in the
default database; that exact test-owned dataset was removed through the
repository and its document count checked afterward. Provider
and gated Hugging Face cases require credentials/authorization and remain
unverified. SQLite integrity and schema-head checks do not prove transaction
rollback, migration compatibility, or persistence across restart. Browser
console observation covered only the manually visited states.

**NO-GO for the requested comprehensive pre-release gate:** the visible
Settings defect and broken E2E collection require correction, and critical
cross-layer workflows, recovery, security, concurrency and responsive states
remain incompletely validated. Recommended order: (1) repair E2E collection
and Storage navigation (small, independent); (2) rerun all isolated browser
workflows and repair confirmed correctness failures; (3) verify SQLite and
PostgreSQL recovery/concurrency and authorized integrations (larger,
environment-dependent); (4) profile and address evidenced usability and
performance issues. This decision is specific to this audit scope and does
not retroactively change an already published release.

## 2026-10-26 v4.8.1 patch smoke (V-20261026)
- Defect found and fixed: starting a dataset validation and tokenizer reports
  together failed one or more jobs with `sqlite3.OperationalError: database is
  locked` (reproduced in three of three attempts on v4.8.0 code). Cause: SQLite
  sessions began deferred, so a transaction that read and then wrote could not
  upgrade its lock while another job held the writer lock, and SQLite fails
  that upgrade immediately instead of honoring the 30 s timeout. The runtime
  SQLite backend now begins every transaction with `BEGIN IMMEDIATE`; the
  regression test `test_sqlite_backend_serializes_concurrent_read_then_write`
  fails without the change. After the fix, two rounds of one dataset validation
  plus three tokenizer reports started together all completed. Switching the
  database to WAL was tried and rejected: it broke the migration-concurrency
  and untouched-database unit tests.
- Validation: backend unit suite (508), Ruff, BasedPyright (0 errors), OpenAPI
  smoke, Angular lint, unit suite (77) and production build passed; the
  launcher started the production SPA, public Wikitext-2 download, custom
  tokenizer upload, dataset validation, tokenizer reports, a 2,000-document
  three-tokenizer benchmark, and the Help dialog on all three pages ran
  without browser console errors. README screenshots were retaken from this
  running application.
- Not re-run: live Playwright E2E suites, Settings -> Database switching,
  Hugging Face keyed tokenizer download (no key available; tokenizers were
  uploaded as `tokenizer.json` files), and the interactive launcher Ctrl+C
  confirmation. The two earlier-failed jobs in the smoke database were
  disposable. The launcher timing output (b567adc) was observed only through
  successful launches.

## 2026-10-25 storage settings, key initialization, job reliability (unit and build validated; live E2E pending)
- Removed `KERAS_BACKEND`, `MPLBACKEND`, `TKBEN_DATA_DIR`, `TKBEN_LOG_DIR` and
  `HF_KEYS_ENCRYPTION_MATERIAL_FILE` from the environment template. Data and log
  directories persist in `settings/paths-config.json` (Settings → Storage,
  restart required); HF/database encryption keys are created only by the
  explicit Settings → Keys action and are required before adding keys.
- Wizard breadcrumb is a full-width chevron strip; page-level job spinners were
  removed so progress shows only in the Jobs panel.
- Job fixes: frontend polling uses `exhaustMap` with retry and no client-side
  timeout; benchmark progress advances per trial; terminal jobs ignore late
  progress; status reads no longer wait on persistence writes.
- Validation: backend unit suite (507) and the Angular suite (76, including
  the `workflow-pages.spec.ts` run-request test that previously failed) pass
  as V-20261025. Live E2E suites and rehydrating in-flight jobs after a page
  reload were not re-run and remain validation debt.

## 2026-10-23 Cross Benchmark wizard and tokenizer preview (VALIDATED 2026-10-24)

The Cross Benchmark run/clone wizard now scrolls its active step content while
keeping the header, step navigation, and actions visible; changing steps resets
the content scroll position. When a tokenizer report is loaded or generated,
the Tokenizers preview also takes its vocabulary size from that report. The
focused isolated Chromium E2E scenarios passed for wizard navigation/scrolling
and first-report generation with a 1,207-entry vocabulary preview. The
2026-10-23 audit also completed a benchmark with two warmups and eight timed
iterations, saved a baseline, and verified dashboard drag ordering through a
successful settings write and reload. The populated plot cards rendered at
desktop width and the shared step strip remained visible at 390×844. The
pointer-reorder E2E passed in the final full runner after its settings load and
stepped pointer movement were synchronized; two earlier aggregate runs had
timed out at the desktop drag response, while the live browser drag also
passed.

The 2026-10-24 follow-up visually reviewed ordinary and cloned wizard states at
390 px and 375 px (the latter measured 376 CSS px), with no document-level
horizontal overflow. Summary content scrolled inside the dialog, Back reset
Inputs to the top, and actions stayed visible. Warmup `-1` and timed trials `0`
disabled Start benchmark. The form guard now requires a run name and blocks
submission whenever any control is invalid; the Angular production build passed.
An isolated empty data root showed “No datasets available” and kept Next
disabled. The report manager passed search, no-match/clear, open/close, and
focus-return checks at 390 px; the two-report data set did not expose pagination.
Dashboard visualization, baseline, visibility, and ordering preferences saved
through successful UI requests and survived navigation, reload, and backend
restart. The captured baseline preferences were restored and visually checked.
No benchmark was submitted in this follow-up. The broader Cross Benchmark UI
gate is VALIDATED; see the durable closure entry `V-20261024` below.

## 2026-10-23 validation wizard preview (VALIDATED)

`POST /api/datasets/validation/preview` powers a live preview in the Sampling &
Filters step of the dataset validation wizard (counts, length histogram,
matching and expected sampled documents, zero-match guard). The analysis job
now sizes its progress estimate from the filtered document count instead of
the unfiltered total. This audit used a four-row CSV in isolated SQLite: a max
length of 80 previewed 3 of 4 rows; a max length of 0 showed the no-match alert
and disabled Next; restoring 80 enabled navigation and the confirmation step
reported 3 of 4 expected documents. The wizard completed 77 metrics and
persisted a report with three analyzed documents. The focused E2E passed in the
standard runner, and the selected report remained available after reload and
backend restart.

## 2026-10-23 tokenizer playground (VALIDATED)

`POST /api/tokenizers/preview` and the collapsible Tokenizer Playground on
the Tokenizers page add a transient, non-persisted single-pass tokenization
preview over up to 3 local tokenizers. The focused Chromium E2E in
`app/tests/e2e/test_tokenizer_flows.py` passed against isolated SQLite. It
exercised the 2,000-character limit, the three-tokenizer selection cap,
special tokens on and off, token text and IDs, per-tokenizer failure
rendering, collapse/reopen, and 1440px/375px layouts without horizontal
overflow. Preview left tokenizer report flags unchanged. The standard test
runner also passed the production build, Ruff, BasedPyright (0 errors; 2,111
warnings), and 70/70 Angular unit tests. The E2E exposed and fixed a CSS rule
that kept the playground visible despite its `hidden` attribute. After that
run, the selector gained an optional in-widget search, bounded CDK virtual
scrolling, selected-tokenizer chips, and instructions that clarify choices
carry across previews on the current page. The production build succeeded.
An in-app Chromium review using isolated SQLite and 103 placeholder local
catalog entries confirmed that the rows render, search narrows to 1 of 103,
and a selected tokenizer stays visible after filtering and selecting another.
The 2026-10-23 audit then exercised search match/no-match/clear, selected-chip
retention, malformed and valid local tokenizer JSON, special tokens on/off,
and the 390×844 viewport without document overflow. No browser errors occurred
after the valid upload, and the updated focused E2E passed in the standard
runner. Raw
`tokenizer.json` artifacts still report `unknown_count` as unavailable
because `UniversalTokenizerAdapter` reads only `unk_token_id`; this matches
the existing benchmark semantics. Hugging Face downloads and gated access
were not part of this local-tokenizer preview run and remain tracked by their
separate gates.

## 2026-10-22 pre-release lifecycle review (pending validation)

The post-v4.6.1 FastAPI SPA deployment delta was reviewed against the
Windows launcher and the canonical test runner. This review identified a
release-relevant safety risk: the test runner selected listener PIDs by port
at cleanup time, potentially targeting unrelated processes after a race.
Its live test path also reused an arbitrary existing `dist` directory
without confirming that the output matched current sources. The corrective
draft PR introduces test-owned PID/start-time/executable verification, a
fresh locked frontend build for live tests, strict preview-port behavior,
and stronger production build/readiness checks. Windows CI is being
expanded to parse the PowerShell scripts and smoke the owned preview process
lifecycle. These are code changes awaiting hosted CI and manual Windows
console validation, not evidence that T0-02 has passed.

Before release, run `start_on_windows.ps1 -Launch` in an ordinary Windows
PowerShell console on both a clean and warm installation. Verify same-origin
`/api/health`, `/`, `/settings`, and static asset delivery; Ctrl+C,
graceful Uvicorn exit, exact process ownership and port release; occupied
port refusal/confirmation; redirected launch; and managed-job restart
reconciliation. Preserve a PARTIAL status if any path remains unverified.

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
| V-20261017 | Current-baseline recheck; post-release delta closure; v4.5.1 release-gate verification | T0-01, T0-02 (delta), T5-06 (v4.5.1) | PASS at `e7f4f9b` on the exercised Windows host: launcher clean bootstrap (pinned Python 3.14.7/uv 0.12.17/Node 22.23.1, Development-profile venv with test extras), warm relaunch with setup skipped, and the documented port-occupied refusal branch; full local CI-equivalent suite passed (compileall, Ruff, BasedPyright with 0 errors, Alembic `0005` initialization at the `data` root, 518 backend tests passed with 13 intentionally skipped provider/credential/live-benchmark cases, OpenAPI smoke with 35 paths, frontend lint, 63/63 frontend unit tests, Angular production build at 2.5.1, and the changed single-lane startup-loading E2E slice). This run closed the previously unexecuted V-20260928 delta validation and provided the v4.5.1 release-gate evidence recorded under T5-06. Legacy ACL-trapped disposable pytest state under `app/tests` remains an environment limitation and is now excluded from collection by `norecursedirs` in `app/tests/pytest.ini`; the tagged `19/19` and `21/21` launcher contract-suite records remain the release-tagged evidence. |
| V-20261020 | Current-tip baseline recheck; launcher warm path; README surface verification | T0-01, T0-02 (warm delta), T1-01 | PASS at `f49ac4b` on the exercised Windows host: compileall, Ruff, BasedPyright with 0 errors/2,018 warnings, Alembic `0005` at the `data` root, 474/474 unit tests, 43/43 live E2E tests with 13 opt-in provider/credential/live-benchmark skips, 1/1 performance test (518 passed total), OpenAPI smoke with 35 paths, frontend lint, 63/63 unit tests, and the production build at backend 3.5.1/frontend 2.5.1 passed. The launcher warm relaunch reached backend and preview readiness (health/proxy 200) and `-KillAll` stopped both process roots and freed ports 5000/8000; the automation harness terminated the foreground launcher wrapper after spawn, so service logs and direct readiness checks provide the evidence. README surface verification passed: the 512x512 logo renders directly below the title, all six shields.io badges and their target links returned HTTP 200, and GitHub GFM rendering contains one logo plus six badge images. T4-03 remains BLOCKED; physical disk exhaustion remains optional PARTIAL/DEFERRED. |
| V-20261020 | Database configuration and live switching delta | T1-04 (delta), T4-04 (delta) | PASS on the current working tree: 223 focused backend unit tests, the browser-driven SQLite settings-isolation E2E, Ruff, compileall, BasedPyright with 0 errors, and the Angular production build passed. The in-app browser authenticated to an isolated PostgreSQL 18.6 cluster, migrated it to `0006_database_runtime_settings`, saved a PostgreSQL-specific setting, switched back to SQLite and confirmed its separate setting, then restarted the backend and confirmed SQLite configuration and settings persisted without database environment variables. A closed PostgreSQL port also returned a connection-test error and an activation rollback while SQLite stayed connected; earlier in this task, an unavailable startup was recovered through the Database panel. Port 5432 was not used. |
| V-20261021 | Automatic settings/dashboard persistence; release candidate validation | T0-01, T0-02 (delta), T1-02 (delta), T1-04 (delta), T4-04 (delta), T5-03 (delta), T5-06 (candidate) | PASS locally on the candidate working tree: backend unit suite 487 passed with the default-root case separately passing (488 total); frontend 69/69, lint and production build; compileall, Ruff, BasedPyright (0 errors), OpenAPI smoke, SQLite migration to `0007`, and seven live browser E2Es passed. The UI exercised all 16 runtime settings, database test-before-switch and failure no-change, SQLite settings isolation, dashboard autosave/reload, and a real synthetic Cross Benchmark report workflow. An isolated PostgreSQL 18.6 cluster on port 55432 migrated to `0007`; `dashboard_revision` and `dashboard_preferences` were present and database-backed preference JSON round-tripped. The launcher cleanup routine passed isolated SQLite selected-path/sidecar deletion with inactive DB and `.gitkeep` preservation, PostgreSQL DB/key preservation, and malformed-configuration fail-closed checks. The app health and frontend returned HTTP 200, and the Database panel was visually inspected in the in-app browser. The first main smoke at candidate commit `0de5fa6f7c167975733e6b9cdda1871d2cfa0ca5` rebuilt the frontend and served backend health plus `/settings` at HTTP 200. Its four-file E2E run passed 7/7, but bootstrap `settings/.env` overrode the shell data-root setting, so that run used the repository default `data` root. Test cleanup assertions restored settings and database configuration and removed synthetic reports, datasets, and tokenizers; the default SQLite database/config files were touched, and no pre-run snapshot is available to prove byte-for-byte restoration. On exact main smoke commit `fbe03383c858c68234a8dff921845cff1ad3a027`, the frontend was rebuilt and isolated SQLite startup, health, `/settings` HTTP 200, and in-app visual review passed. The four-file E2E suite then passed 7/7 in 53.86s with `settings/.env` temporarily set to the QA root; the backend log confirmed the isolated path, the environment file was restored byte-for-byte, and default database/config timestamps did not change during this corrected run. The opt-in E2E first failed because its helper read the repository-default `.env` data root while the live app used the isolated candidate root; rerunning all four relevant E2E files with the same isolated `TKBEN_DATA_DIR` passed 7/7. This was test-process setup, not an application failure. The original `settings/.env` was restored byte-for-byte; services and the isolated PostgreSQL cluster were stopped, and their ports were confirmed clear. Findings were consolidated here and in release notes; disposable logs, screenshots, scripts, databases, and cluster files were removed afterward. Hosted validation completed on release commit `2ecd12119b86864d617367bd6b0712bb9a1b2b92` in main CI run `39582304529`; tag `v4.6.0` was published from that `main` commit. |
| V-20261021 | Shim and legacy compatibility removal | T0-01 (delta), T1-04 (delta), T4-04 (delta) | PASS locally on the cleanup working tree: removed the deprecated `/api/tokenizers/settings` endpoint, legacy `DATABASE_*` environment import, legacy `runtime-settings.json` import, and the pre-baseline Alembic chain; the current schema is created by `0001_current_schema`. Verification passed with 475/475 backend unit tests, 69/69 Angular unit tests, frontend lint and production build, Ruff, BasedPyright (0 errors/2,095 warnings), compileall, OpenAPI generation (38 paths), a fresh SQLite baseline initialization, and Alembic `check` with no remaining model diff. The previous local `data/database.db` was backed up under `assets/QA/legacy-baseline-backup-20261021` and recreated. Live browser E2E and PostgreSQL were not re-run in this cleanup slice. |
| V-20261021 | Strict shim cleanup follow-up | T0-01 (delta), T1-04 (delta), T4-04 (delta) | PASS on the current working tree: removed the tokenizer adapter's encode-only fallback and duplicate dataset-report aggregate fallbacks; deleted the user-approved archived database snapshot; corrected the `/api/settings` warning description; updated tokenizer test doubles to exercise the supported callable interface. The standard runner passed 518 Python tests, with 13 skipped and one isolated-default-path test deselected (that assertion passed separately); Ruff, BasedPyright, compileall, Angular lint, 69/69 Angular unit tests, and the production build passed. Browser E2Es are part of the Python run; no `test:e2e` npm script is configured. OpenAPI exposes 38 paths and excludes `/api/tokenizers/settings`. Fresh SQLite and isolated PostgreSQL 18.6 initialization both reached `0001_current_schema`; PostgreSQL database-backed settings round-tripped. The isolated environment was restored, test services and PostgreSQL stopped, and ports 5000, 8000, and 55432 were confirmed free. Manual review retains synthetic Zipf/word-cloud generation when metrics are absent and The Pile's unsupported-source “legacy dataset scripts” guard; historical release notes remain records. |
| V-20261021 | Environment-owned CORS origins | T1-02 (settings boundary delta) | PASS on the current working tree: the focused environment configuration, startup CORS, and settings-route suite passed 185/185. Coverage includes explicit `CORS_ORIGINS` loading, normalization and deduplication, blank-value fallback, malformed-origin rejection, and exclusion from the Settings API. Ruff and `git diff --check` passed; BasedPyright reported 0 errors and 2,089 warnings. No database schema or HTTP contract change was introduced. |
| V-20261021 | Final whole-ledger and settings-bundle validation | T0-01, T1-02, T1-04, T4-04, T5-06 (candidate) | PASS for the runnable local gates on the current working tree plus the schema-rebase guard: 486 backend unit tests; 43 browser E2Es passed with 13 opt-in provider/credential/live-benchmark skips; 69/69 frontend unit tests, lint, and production build; compileall, Ruff, BasedPyright (0 errors), and OpenAPI smoke. In-app browser review confirmed the Settings Data and Runtime controls and Database panel render; `/api/settings` still exposes the 16 application fields and excludes `CORS_ORIGINS`, whose allow/deny behavior was checked separately. Fresh SQLite initialization and isolated PostgreSQL 18.6 settings/dashboard JSON round-trips passed. An isolated SQLite database created by replaying the v4.6.0 Alembic chain reached `0007_dashboard_ui_preferences`; current startup schema-compared and rebased it to `0001_current_schema` while retaining runtime and dashboard settings. PostgreSQL 18.6 separately passed the matching-schema `0007` marker rebase and settings round-trip; a drifted schema was refused without marker change. The first combined runner attempt had one setup-only failure because isolated `settings/.env` conflicted with a default-root assertion; rerunning unit tests under the repository default and E2E tests under the isolated root passed. Environment and QA fixtures were restored, services stopped, and ports 5000, 8000, and 55432 confirmed free. Candidate hosted CI and release actions remain pending. |

| V-20261021 | v4.6.1 versioned patch-candidate smoke | T0-01, T1-01, T1-02, T1-04, T5-06 | PASS on the versioned working tree: focused backend migration/CORS/settings/tokenizer tests passed 204/204; compileall, Ruff, BasedPyright (0 errors/2,093 warnings), OpenAPI smoke (38 paths), frontend lint, 69/69 frontend unit tests, and the production build passed. The first frontend test attempt hit sandbox-temp `EPERM` before collection; rerunning with the project cache as TEMP/TMP passed. Isolated live startup returned `/api/health` status `ok`; `/settings` returned HTTP 200; the in-app browser rendered Settings → Database (SQLite connected), Datasets, Tokenizers, and Cross Benchmark. OpenAPI excluded the deprecated tokenizer-settings route. The live smoke used isolated SQLite data/logs under `assets/QA`; the standalone OpenAPI import empty default-root log was removed. The two ignored E2E screenshots were reviewed and removed because their settings-UI and T2-03 findings are already recorded here and the tests regenerate them. `settings/.env` was restored byte-for-byte, temporary outputs were removed, processes stopped, and ports 5000/8000 were free. Hosted CI remains pending. |
| V-20261021 | FastAPI production SPA and launcher lifecycle delta | T0-01 (delta), T0-02 (delta) | PARTIAL on the current working tree: seven focused FastAPI route/startup tests passed. Direct Uvicorn served `/`, `/settings`, the built JavaScript bundle, `/api/health`, `/api/settings`, and dataset API requests; the in-app browser rendered the SPA and loaded assets/API calls from the same origin. Unknown `/api` paths and missing `.js` assets returned 404. A synthetic parent/child process-tree check stopped only the supplied tree and preserved the unrelated UI-port listener. The actual `start_on_windows.ps1 -Launch` smoke timed out before readiness in the nested PowerShell host; a temporary faulthandler hook captured Python 3.14's Windows Proactor loop blocked in `socket._fallback_socketpair()` during loop initialization. Direct Uvicorn starts normally. The original `settings/.env` was restored by matching SHA-256; scratch data was removed and ports 64076, 64077, and 8000 were free. Launcher readiness and normal launcher Ctrl+C shutdown remain unverified on a standard Windows console. |
| V-20261022 | PR #8 Windows SPA and launcher validation follow-up | T0-01 (delta), T0-02 (delta), T5-06 (candidate) | PARTIAL at PR head `75eab8962cec3cdaf484df4a4c3ed94900282207`. Hosted backend, frontend, and Windows lifecycle jobs passed. The Windows PowerShell 5.1 launch served `/api/health`, `/`, `/settings`, and the production bundle; warm build reuse and source-triggered rebuild passed. Uvicorn shut down and ports were released, but the Codex PTY routed Ctrl+C to the child instead of the launcher's cancel handler, so the launcher exited 1; standard interactive-console Ctrl+C remains unverified. A same-session listener on port 5000 was visible to the exact PR launcher invoked noninteractively: it exited 1 with the listener PID and no-termination message, and the listener remained bound. The isolated `test_default_runtime_paths_use_repository_data` rerun passed 1/1 with the repository default path; the earlier aggregate was 532 passed, 1 setup-induced failure, and 13 skipped because that aggregate used the isolated path override. Frontend tests passed 69; Ruff passed; BasedPyright reported 0 errors. Keep PR #8 draft until standard-console shutdown is verified; the interactive listener-confirmation/termination path was not exercised. |
| V-20261022 | v4.7.0 release candidate smoke | T0-01, T0-02 (delta), T5-06 | PASS on `develop` after the version bump: backend compileall, Ruff, BasedPyright (0 errors/2,101 warnings), database initialization, and 489/489 unit tests passed; OpenAPI generates (37 paths) and a duplicate-operation-ID warning from the SPA routes was fixed by excluding them from the schema; frontend lint, 69/69 unit tests, and the production build passed. Direct Uvicorn served `/`, `/settings`, `/api/health`, `/api/settings`, and `/api/datasets/list` with HTTP 200 and returned 404 for unknown `/api` paths; the server was stopped and port 8000 freed. Consolidated the former `assets/QA` FastAPI static-hosting and PR #8 reports into the V-20261021/V-20261022 rows above; the launcher `-Launch` readiness and standard-console Ctrl+C (PR #8, draft, excluded from v4.7.0) remain unverified. Hosted CI pending. |
| V-20261022 | v4.7.0 post-release full E2E revalidation (`afbf637`) | T0-01, T0-02 (delta), T1-01 through T3-05, T5-06 | PASS for the runnable gates; T0-02 stays PARTIAL. Launcher: `start_on_windows.ps1 -Launch` run three times from the app Terminal panel (an interactive console). The first run rebuilt a stale Angular output, then `/api/health`, `/`, `/settings`, `/api/settings` and `/api/datasets/list` returned 200 and unknown `/api` paths returned 404; warm relaunches reached health without a rebuild. Ctrl-C sent to the tab ended the launcher shell, freed port 5000, and left no TKBEN Python/Node process; the console transcript was not captured, so graceful-exit text and launcher exit code are unverified. The interactive occupied-port prompt was not exercised because no tool can type into the panel. Standard runner after installing the locked `--all-extras` environment and Playwright Chromium into `runtimes/cache/playwright`: Ruff clean, BasedPyright 0 errors/2,101 warnings, 533 passed with 13 opt-in skips, 69/69 frontend tests. Opt-in synthetic E2Es (`E2E_RUN_BENCHMARKS`, tokenizer report flow and UI lifecycle) ran against the production launcher origin on port 5000: 49 passed, 7 skipped (HF key, provider, restart-name opt-ins). Public Hugging Face discovery requires an active key (400 "No active Hugging Face access key"), so T4-01/T4-03 were not re-run (BLOCKED, no key). Defect found and fixed: the E2E suite left nine `custom/` datasets (`e2e_sample`, `e2e_xlsx_import_*`, `qa_row_*`) and the `CUSTOM_e2e_tiny_tokenizer` in the live default database; the new `app/tests/e2e/conftest.py` session fixture removes datasets and tokenizers created during the run, and a rerun left both catalogs empty. The leaked rows were deleted through the API; `settings/.env` hash unchanged. Managed-job restart reconciliation was not re-run live (unit/earlier evidence only). Hosted CI for this delta is pending. |
| V-20261022 | Launcher lifecycle and restart-reconciliation follow-up | T0-02 (delta), T5-01 | PASS for managed-job reconciliation, PARTIAL for launcher shutdown. Live: a 24,000-document upload completed, a dataset analysis job was started, and the backend (the exact listener PID) was force-killed mid-job. The launcher, run in the app Terminal panel under `Start-Transcript`, reported `FastAPI exited with code -1 after startup`, logged the application log path, returned exit code 1, and left no TKBEN process or port 5000 listener. After relaunch, the interrupted job was `failed` with "Job interrupted because the application restarted while it was running." at its original ID; the completed dataset was retained and then deleted. The Ctrl-C graceful-exit text and exit code remain unverified: the panel's stop action terminates the wrapper before it can log them, although the backend stopped and released the port each time. A `CTRL_BREAK_EVENT` to a launcher in its own process group (not equivalent to Ctrl+C) made the launcher exit 2 and orphaned the exact TKBEN backend listener; `start_on_windows.ps1 -KillAll` then stopped that process tree and freed the port. Injecting console input or Ctrl+C events into a separate console was declined by the permission system, so the interactive occupied-port prompt is NOT TESTED. Hosted CI for these commits has not been run because nothing was pushed. |
| V-20261022 | v4.7.1 patch release smoke and screenshot refresh | T0-01, T0-02 (delta), T5-06 | PASS on `develop` after the version bump. Delta since v4.7.0 is the E2E cleanup fixture plus the chart-label fix below. Ruff clean, 489/489 backend unit tests (run with `PYTHONPATH=app`), frontend lint, 69/69 unit tests, and the production build passed. `start_on_windows.ps1 -Launch` (PowerShell 5.1) reached `/api/health` 200 and served `/`, `/settings`, `/api/datasets/list` and `/api/tokenizers/list` with 200. Live workload: a 2,521-paragraph technical-docs CSV upload, full dataset analysis, three tokenizer reports, and a 3-tokenizer benchmark (custom BPE 8k/32k and WordPiece 16k trained locally with `tokenizers`) all completed. Defect found and fixed: the cross-benchmark bar chart rotated its x-axis tokenizer labels past the SVG bottom edge, so they were clipped and overlapped the axis title; the viewBox height and axis title offset were adjusted and verified visually. The README screenshots were retaken at 1600 px width from that workload and visually inspected. The seeded data was deleted afterwards and the launcher stopped (port 5000 free). Standard-console Ctrl+C, the interactive occupied-port prompt, and Hugging Face-gated checks are unchanged validation debt. |
| V-20261022 | Test suite consolidation | T0-01 (validation tooling delta), T5-06 (candidate recheck) | PASS on the current working tree: consolidated 45 Python unit modules plus 1 performance module into 7 unit suites, 13 browser E2E modules into 6 suites, and 14 Angular specs into 5 suites; preserved all 615 test cases and the manual Windows launcher scenario. The default Windows runner passed 533 Python tests with 13 opt-in skips (546 collected), 69/69 Angular tests, Ruff, BasedPyright (0 errors; 2,102 warnings), frontend lint, and the production build. Python completed in 126.21s and Angular tests in 2.80s; a separately measured E2E-only run passed 43 with 13 opt-in skips in 93.18s. The suite used an isolated test-data root, cleanup completed, and ports 5000/8000 were free. Pre-change E2E execution time was not measured, so no E2E speedup comparison is claimed.

| V-20261023 | Tokenizer Playground live browser validation | T2-08, T5-07 | PASS on the exercised Windows host with isolated SQLite: the focused Chromium E2E passed 1/1, covering four custom WordLevel tokenizers, the 2,000-character boundary, selection cap, exact text/ID chips, special-token toggle, per-item error state, collapse/reopen, responsive layout at 1440px and 375px, and no report creation. The production build, Ruff, BasedPyright (0 errors; 2,111 warnings), and 70/70 Angular unit tests passed. Fixed the playground collapse CSS issue found by the browser run; E2E cleanup removed its test tokenizers and the runner freed ports 5000/8000. Raw tokenizer JSON `unknown_count` remains unavailable by the existing adapter contract; external Hugging Face download/access was not tested. |
| V-20261023 | Comprehensive end-to-end validation audit | T0-02 (delta), T2-02, T2-08, T5-07, local quality gate | PARTIAL overall on the current `develop` working tree. Two full runs initially timed out waiting for the desktop pointer-reorder `PUT /api/settings/dashboard`; the exact isolated rerun passed. The E2E now waits for dashboard preferences to load on navigation/reload and uses a 10-step pointer movement; the focused runner passed and the next full runner passed with 539 Python tests, 13 skipped, and no failures. The production build, Ruff, BasedPyright (0 errors; 2,149 warnings), and 71/71 Angular tests passed. In the live app against isolated SQLite, a four-row CSV completed the validation wizard with 80/0/80 max-length preview, zero-match recovery, 77 metrics, and a persisted three-document report; dataset/report/tokenizer state survived reload and backend restart. Tokenizer flows covered malformed and valid JSON, search match/no-match/clear, selection retention, special-token on/off, and 390×844 layout without overflow. Cross Benchmark completed two warmups/eight timed iterations, rendered 34 metrics, persisted a baseline, and saved/reloaded dashboard order. A normal visible Windows console ran `start_on_windows.ps1 -Launch`, returned `/api/health` 200, handled a generated Win32 console Ctrl+C event with launcher exit code 0, stopped the exact owned backend, and freed the port; `.env` was restored to its original SHA-256. Remaining: interactive occupied-port confirmation, redirected launch, current hosted CI, live Hugging Face credentials/gated access, isolated PostgreSQL recheck, and physical disk exhaustion were not exercised in this audit. `npm ci` reported 15 dependency audit findings (4 moderate, 9 high, 2 critical); advisories were not triaged. Temporary fixtures and isolated database were removed. |

| V-20261024 | Cross Benchmark Phase 2 closure | T5-03, T3-04 (delta), local UI form guard/build | PASS on the current working tree. The ordinary and cloned Cross Benchmark wizards were reviewed at 390 px and 375 px, and the ordinary wizard at the five planned viewport sizes; dialogs had no document-level horizontal overflow, internal scrolling and step reset worked, and actions stayed visible. Warmup `-1` and timed trials `0` disabled Start benchmark; the required-name/full-form guard fix passed the Angular production build. The empty-dataset state kept Next disabled in an isolated disposable root. Report-manager search/open/close/focus checks passed at 390 px; pagination was unavailable with two reports. Dashboard visualization, baseline, visibility, and order saves returned HTTP 200 and persisted through navigation, reload, and backend restart; the captured preferences were restored exactly. No benchmark was submitted and Phase 3 was not started. |
| V-20261024 | Remaining-validation Phases 0-1 (preflight and production launcher) | T0-02 (delta) | PARTIAL at `bc1ade5` on Windows 10.0.26200 (Python 3.14.7, Node 22.23.1, uv 0.12.17) against the active local SQLite database. Phase 0 preflight confirmed matching SHAs, free ports 5000/8000, and synthetic fixtures (4-row CSV, WordLevel tokenizers with 3/12/1,207 tokens, one invalid JSON). Phase 1 results: clean `-Launch` PASS (frontend build, `/api/health` 200, one port-5000 listener, routes `/`, `/dataset`, `/tokenizers`, `/cross-benchmark`, `/settings` rendered); warm launch skipped setup and reached readiness; interactive occupied-port decline and redirected occupied-port refusal both PASS (the occupying QA process was left intact). FAIL/gaps: Ctrl+C delivered through the automation PTY stopped Uvicorn gracefully but returned launcher exit code 1 with the failure-cleanup warning (the 2026-10-23 generated Win32 console Ctrl+C returned 0, so this is attributed to the PTY harness, not reproduced in a normal console); a redirected free-port launch exited during stale-environment `uv sync` with `Get-FileHash` unrecognized in that restricted command context; interactive occupied-port acceptance was not run. See open issues. |
| V-20261025 | v4.8.0 release candidate validation | T0-01, T0-02 (delta), T5-06 | PASS on `develop` at the release candidate: compileall, Ruff, BasedPyright (0 errors, 2,172 warnings), OpenAPI generation (43 paths), 507/507 backend unit tests, frontend lint, 76/76 Angular unit tests (including the `workflow-pages.spec.ts` run-request test that failed before the 2026-10-25 changes), and the production build. `start_on_windows.ps1 -Launch` from PowerShell 5.1 reached readiness, served the SPA and API from port 5000 and was stopped with `-KillAll` (ports freed). README screenshots (dataset, tokenizer overview, cross-benchmark) were retaken at 1600 px from the running app and visually inspected; the cross-benchmark capture shows an empty exact round-trip chart because that metric is not available for the stored Hugging Face tokenizer report. Live E2E suites and in-flight job rehydration after reload were not re-run for this release. Hosted CI for the release commit is recorded in the release gate. |
| V-20261029 | PowerShell launcher cleanup and local-data removal audit | T0-02 (delta), T0-03 (delta) | PASS for the isolated cleanup harness: 11 checks passed under Windows PowerShell 5.1 and PowerShell 7, including four launcher `-DryRun` actions from a different working directory against an external configured data root; nested backend/frontend and Python package build caches including Vitest coverage output; temporary spill-file naming; persistent database, key, settings, and tokenizer preservation; repeated cleanup; missing-after-preview and locked files; unsafe data-root rejection; ACL error reporting; standard uninstall including interrupted writes; complete SQLite/PostgreSQL local-data handling; incompatible `-DryRun -Launch` and `-DryRun -KillAll` rejection; and active-process snapshot filtering/refusal. Both PowerShell versions passed the outside-sentinel junction check and live process-table refusal check. The current checkout has no frontend `node_modules` or `dist`, so the existing owned-preview smoke could not run locally. Hosted CI and Python unit tests for the new managed-temp paths were not run in this audit. The actual repository cache dry run found no targets and made no changes. |

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
| runtime.startup.local-webapp | PARTIAL | The previous Windows bootstrap and Linux diagnostic startup reached health, proxy, route, and restart-recovery checks. This delta's direct Uvicorn SPA/API smoke passed, but the production launcher did not reach readiness in the nested PowerShell host; standard Windows-console startup and Ctrl+C remain to be checked. Populated dashboards and responsive coverage are separate gates. | 2026-10-21 |
| runtime.windows-launcher | PARTIAL | Historical clean/warm bootstrap, stamps, build repair, port conflict refusal, maintenance menu, cleanup, quoted-process handling, and localized denied termination remain validated as recorded. In the 2026-10-23 audit, `start_on_windows.ps1 -Launch` reached `/api/health` 200 in a normal visible Windows console; a generated Win32 console Ctrl+C event returned launcher exit code 0, stopped the exact owned backend, and freed its port. The launcher now follows the repository's established PowerShell supervision pattern, catching `PipelineStoppedException` and cleaning up in `finally`; `.env` matched its original SHA-256 after the test. Keep PARTIAL because interactive occupied-port confirmation and redirected launch were not re-exercised in this audit. | 2026-10-23 |
| configuration.runtime-settings | WORKING | All 16 application settings, bounds, sparse persistence, restart, conflict, reset, and new-work effects passed. Sparse overrides persist in the active database; all UI controls autosave with debouncing and serialize edits, surface write failures, and flush a valid pending edit on navigation. Browser coverage exercised all 16 controls and an actual backend restart. Network bootstrap values, including API base path, listener addresses/ports, and CORS origins, remain `.env`-owned and excluded from the Settings API. `CORS_ORIGINS` overrides the UI-derived default and is validated at startup. SQLite/PostgreSQL independence and reload after switching passed. The component deliberately retains broader WORKING scope. | 2026-10-21 |
| configuration.database-connection | VALIDATED | SQLite and PostgreSQL controls, schema validation, atomic file persistence, encrypted PostgreSQL password handling, environment-free startup, connection testing, runtime activation, and rollback/recovery were exercised. | 2026-10-20 |
| runtime.managed-job-lifecycle | VALIDATED | Job persistence, terminal failure on restart, cancellation conflicts, and retained completed data passed. Resumption/checkpointing is intentionally unsupported. | 2026-09-22 |
| backend.api-contracts | VALIDATED | Route/unit/OpenAPI coverage and restart-addressable job behavior passed. Provider-dependent routes remain separately bounded. | 2026-09-25 |
| persistence.sqlite-alembic | VALIDATED | Alembic head `0001_current_schema`, the schema-verified rebaseline from exact published v4.6.0 revision `0007_dashboard_ui_preferences`, rejection of drifted/unknown schema markers, schema/rollback/cascade contracts, restart reconciliation, active-database runtime and dashboard preference persistence, and controlled `SQLITE_FULL` cleanup/retry passed. | 2026-10-21 |
| persistence.postgresql-runtime | VALIDATED | Isolated PostgreSQL 18.6 concurrency and baseline schema initialization, authenticated UI connection testing, runtime switching, database-local settings, restart, rollback, and retry passed. A matching current schema stamped with the published v4.6.0 revision rebaselined after schema comparison; dashboard preference columns and JSON roundtrip passed on port 55432. The actual tagged v4.6.0 migration chain was separately replayed and rebaselined on isolated SQLite. The existing host listener was untouched. | 2026-10-21 |
| data.dataset-import-and-analysis | VALIDATED | CSV/XLSX import, `.xls` rejection, size boundary, analysis metrics, reload, and public Wikitext import passed. Physical host exhaustion is separate debt. | 2026-09-24 |
| data.custom-tokenizer-storage | VALIDATED | Canonical artifact replacement, catalog identity, restart persistence, report/vocabulary rendering, and deletion passed. | 2026-09-24 |
| data.tokenizer-playground-preview | VALIDATED | The transient preview accepted three local tokenizers, applied or omitted special tokens, returned ordered token text/IDs and per-tokenizer failures, and created no reports. Raw `tokenizer.json` unknown counts remain unavailable. | 2026-10-23 |
| integration.huggingface-discovery-and-download | PARTIAL | Public `bert-base-uncased` flow passed with vocabulary 30,522; three gated candidates failed download, so no gated report exists. | 2026-09-25 |
| benchmark.execution-and-reporting | VALIDATED | Admission, reports, bounded worker concurrency, progress/cancel, immediate rerun, metadata, reload, and deletion passed. Performance samples are host-specific. | 2026-09-24 |
| benchmark.dashboard-and-pdf-export | VALIDATED | Dashboard data/customization and all eight supported benchmark PDF chart forms plus the five-page tokenizer PDF passed rendered review. Dashboard layout, visibility, visualization, and baseline preferences now autosave to the active database; dashboard browser tests cover immediate save and reload. | 2026-10-21 |
| ui.route-shell-and-empty-states | VALIDATED | Dataset, Tokenizers, Cross Benchmark, Settings, loading/error/empty, and populated states passed the exercised route and responsive checks. | 2026-09-25 |
| ui.startup-readiness | VALIDATED | Offline, slow, terminal-failure, retry, and recovered frontend-first states passed in the in-app browser, and the 2026-10-17 baseline recheck passed the single-lane tokenizer startup-flow E2E including reduced-motion and narrow-viewport assertions. The 2026-10-20 full live E2E recheck passed the same 2/2 startup slice at `f49ac4b`. | 2026-10-20 |
| ui.dataset-dashboard | VALIDATED | Metric families, persisted dashboard reload, malformed optional histogram fallback, and export action passed. | 2026-09-23 |
| ui.settings-page | VALIDATED | Typed settings, all 16 runtime fields, debounced autosave, conflict reload, persistence across reload and backend restart, reset, key lifecycle, masked credential storage, keyboard navigation, and responsive states passed. The Database tab passed test-before-switch, failed-target no-change, SQLite switching, settings isolation, and restored connection configuration. | 2026-10-21 |
| ui.cross-benchmark-workflow | VALIDATED | Wizard, report manager, tags, baseline/delta, clone/customize, tables, ordering, database-backed preference persistence, and reload passed historically. A real pointer drag changed widget order and persisted after reload. The 2026-10-24 follow-up completed ordinary and cloned wizard review at 390 px and 375 px, report-manager search/open/close/focus checks at 390 px, invalid-trial form-guard checks, and dashboard preference save/navigation/reload/backend-restart persistence. The form guard fix passed the Angular production build; baseline dashboard preferences were restored. | 2026-10-24 |
| ui.tokenizer-report-and-vocabulary | PARTIAL | 1,207-entry report persistence, contiguous API paging, rendered pagination, restart report, and four-viewport route states passed historically. The 2026-10-23 first-report E2E generated a 1,207-entry tokenizer report and verified the catalog preview updated from unknown to 1,207. | 2026-10-23 |
| ui.tokenizer-playground | VALIDATED | The focused Chromium E2E and live isolated-SQLite workflow passed text/selection boundaries, search match/no-match/clear, selection retention across filtering, special tokens on/off, token text/IDs, per-item error rendering, collapse/reopen, and 1440px/390px/375px layouts without document overflow. Invalid JSON showed an inline error; a valid local WordLevel tokenizer created a report. No browser errors occurred after valid upload. Raw tokenizer JSON `unknown_count` remains unavailable by the existing adapter contract. | 2026-10-23 |
| deployment.source-only-local | VALIDATED | Published v4.7.1 source-only release with hosted CI run `39800902959` passing frontend, backend, and Windows launcher jobs at the release SHA; tag and publication traceability are recorded below. The 2026-10-23 working-tree audit is separate from that release evidence, and hosted CI for the current tree was not run. | 2026-10-23 |
| deployment.windows-portable-bootstrap | VALIDATED | Pinned Python/Node/uv bootstrap, dependency repair, stamped build reuse, and migration remain validated historically. The current static-SPA launcher readiness limitation is tracked under `runtime.windows-launcher`; it does not change the bootstrap evidence. | 2026-10-20 |
| deployment.cross-platform-manual | OUT_OF_SCOPE | Ubuntu manual execution was diagnostic only; macOS and hosted runtime support are not release targets. | 2026-09-22 |
| runtime.windows-launcher / T0-02 | PARTIAL | Unresolved from V-20261024 Phase 1: (1) Ctrl+C through a non-console PTY returned launcher exit code 1 (a generated Win32 console Ctrl+C returned 0 on 2026-10-23); (2) a redirected `-Launch` in a restricted command context failed at `Get-FileHash` during stale-environment sync, so redirected free-port launch is unverified; (3) interactive occupied-port acceptance was never run. Hosted CI run [41023891665](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/runs/41023891665) passes PowerShell parsing, the 11-check cleanup harness, and owned-preview startup, root/deep-link readiness, and cleanup. The local PS5.1 and PS7 reruns also passed; host policy skipped the junction and live process-table branches. Interactive launcher lifecycle and occupied-port behavior remain unverified. Keep this gate PARTIAL. | 2026-10-30 |
| deployment.containerized | NOT_IMPLEMENTED | No active container deployment configuration exists. | — |
| deployment.binary-packaging | NOT_IMPLEMENTED | Releases are source-only; no installer or binary workflow exists. | — |
| test-infrastructure.local-quality-gates | VALIDATED | The 2026-10-23 final standard runner passed the production build, Ruff, BasedPyright (0 errors; 2,149 warnings), 539 Python tests with 13 opt-in/provider/credential skips, and 71/71 Angular tests. The pointer-reorder E2E timed out in two preceding full runs; explicit preference-load waits and 10-step pointer movement made both the focused test and final full suite pass. Hosted CI run [41023891665](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/runs/41023891665) passed backend compile/Ruff/type/database/unit/OpenAPI checks, frontend install/lint/unit/build, and Windows parse/cleanup/preview checks. Local PS5.1 and PS7 cleanup-harness reruns passed; junction and live process-table branches were skipped by host policy. PostgreSQL was not re-exercised. The 2026-10-23 `npm ci` audit findings (4 moderate, 9 high, 2 critical) remain untriaged. | 2026-10-30 |
| test-infrastructure.database-settings-delta | PASS | The tagged v4.6.0 Alembic chain was replayed into isolated SQLite; current startup rebaselined `0007_dashboard_ui_preferences` to `0001_current_schema` and retained runtime settings and dashboard preferences. PostgreSQL 18.6 separately passed the schema-matching marker rebase. Current SQLite/PostgreSQL baseline, test-before-switch, switching, rollback/no-change, and per-database isolation checks passed. The full live browser E2E run passed 43 tests with 13 expected opt-in skips. | 2026-10-21 |
| test-infrastructure.automatic-settings-persistence | PASS | Runtime settings and dashboard preferences save to the active database with independent revisions. The candidate passed reload/restart, serialized edits, visible failed-write handling, stale-revision recovery, all 16 runtime controls, dashboard reload, and a real synthetic report workflow. Full relevant browser E2E set passed 7/7. | 2026-10-21 |
| test-infrastructure.hosted-ci-and-release-evidence | VALIDATED | Hosted CI run [41023891665](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/runs/41023891665) passed all three jobs on commit `123b483abd610254831ef1ffcced1d632f2fac17`: backend validation, frontend validation, and Windows launcher contract. Historical release runs remain recorded above. | 2026-10-30 |
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
| T0-01 | PASS | Published v4.7.1 is the latest release. Hosted CI run `39800902959` passed frontend, backend, and Windows launcher jobs on release commit `b8ae9fe5fdb11126387cc440903ef73f0a555b22`; current working-tree local results are tracked separately under V-20261023. |
| T0-02 | PARTIAL | Historical bootstrap, conflict refusal, maintenance, and process-ownership evidence remains as recorded. In the 2026-10-23 normal visible Windows console run, `start_on_windows.ps1 -Launch` reached `/api/health` 200; a generated Ctrl+C console event returned launcher exit code 0, stopped the exact backend PID, and freed the port, with `.env` restored byte-identically by SHA-256. The 2026-10-29 cleanup delta passed the isolated 11-check harness and PowerShell parsing, but interactive occupied-port confirmation and redirected launch remain unverified; both PowerShell versions passed the cleanup junction/CIM checks. The owned-preview smoke was unavailable locally because frontend dependencies/build output were absent. Keep this gate PARTIAL. |
| T0-03 | PASS | All 13 maintenance-menu routes, install profiles, expected update refusal, destructive decline/approval, cleanup, uninstall, and Kill All passed. The 2026-10-29 delta adds a dedicated cleanup harness; 11 checks passed under Windows PowerShell 5.1 and 7, including live junction and CIM branches. |
| T1-01 | PASS | Startup E2E 2/2 and shell recovery after persistence restart passed; the 2026-10-20 full live E2E recheck passed the same 2/2 startup slice at `f49ac4b`. |
| T1-02 | PASS | Existing boundary, conflict, reset, and downstream-effect coverage remains current. Browser coverage exercised all 16 controls, debounced autosave, serialized edits, failed-write visibility, stale-revision recovery, reload, and persistence through backend restart. `CORS_ORIGINS` remains environment-owned and outside the 16-field Settings API bundle; explicit allowed and denied origins were checked. |
| T1-03 | PASS | At `09805b7`, synthetic Settings Keys lifecycle, masking, duplicate/active-delete/reveal policies, single-active switching, ciphertext separation, and cleanup passed; live supplied-key coverage was SKIPPED because no test key was available. |
| T1-04 | PASS | Focused migration/persistence/settings contracts and runtime override restart coverage passed. The candidate confirmed SQLite/PostgreSQL settings isolation, dashboard preference autosave, database-backed settings, configuration-free startup, and persistence after restart. A database created by the published v4.6.0 migration chain was schema-compared and safely rebaselined on SQLite; drift was refused. |
| T1-05 | PASS | At `993e52e8`, populated Dataset/Tokenizer filter matrices, exact numeric boundaries, combined/no-match/reset, completion-order races, and stale discovery result/error handling passed. |
| T2-01 | PASS | CSV/XLSX upload and persistence, UI catalogue visibility, `.xls` rejection, 25 MiB/25 MiB+1 boundary, and ready-state import passed. Physical disk exhaustion remains separate debt. |
| T2-02 | PASS | At `0993e900`, all six dataset metric families were selected, analyzed, persisted, rendered, and restored; metric unit tests 78/78 and API E2E 1/1 passed. The 2026-10-23 browser audit also exercised filtered preview 3/4, zero-match alert/Next disablement, recovery to 3/4, 77-metric completion, persistence of the three-document report, and reload/restart recovery. |
| T2-03 | PASS | At `de1aaee6`, a controlled duplicate/URL/email/HTML/line-break/empty-document dataset produced the expected quality, structure, and compression metrics and reloaded correctly. |
| T2-04 | PASS | Same-name custom-tokenizer replacement, one catalog row, canonical seven-token artifact, launcher restart, report/vocabulary rendering, and deletion passed. |
| T2-05 | PASS | The 1,207-entry tokenizer report reopened; API pages 500/500/207 were contiguous; first/middle/final UI paging and four viewport rendering passed. |
| T2-06 | PASS | At `f736f938`, the populated Cross Benchmark wizard created a local report, verified persistence/reload, and cleaned its dataset/tokenizer/report; combined E2E 9/9. |
| T2-07 | PASS | At `cc91fec1`, API progress reached at least 20%, cancellation became terminal with no report, and an immediate two-document rerun completed and rendered. Browser sampling showed 5% at one capture; the API and terminal evidence remain the gate basis. |
| T2-08 | PASS | Live browser/API preview round trips with three isolated custom WordLevel tokenizers returned two tokens without special tokens and four with `[CLS]`/`[SEP]`; rendered token IDs matched. Deleting one selected artifact produced an inline error result while the other two succeeded. Preview did not create reports. `unknown_count=null` for raw JSON artifacts remains expected. |
| T3-01 | PASS | Three comparable 1,000-document runs recorded throughput, latency percentiles from 80 observations per run, phase timing, RSS, and memory delta. Samples describe this host and are not a general performance guarantee. |
| T3-02 | PASS | At `b087db60`, requested/effective parallelism 1/1 and 2/2, two-worker overlap, persisted configuration, ordering, observations, per-document samples, and report reload passed. |
| T3-03 | PASS | Report search, 25/1 pagination, tags, confirmation, and physical deletion passed against populated current-schema reports. |
| T3-04 | PASS | Baseline/delta views, data tables, reload, clone configuration, visualization overrides, order, and customization persistence passed. The 2026-10-24 UI save changes also persisted through navigation, reload, and backend restart; original dashboard preferences were restored afterward. |
| T3-05 | PASS | Dataset/tokenizer/benchmark export smoke and the expanded all-eight-form benchmark review plus five-page 1,207-entry tokenizer PDF passed `pdfinfo`, rendering, and visual legibility checks. |
| T4-01 | PASS | Supplied key authentication, public `bert-base-uncased` discovery/download/report/reload/cleanup passed; vocabulary size was 30,522. |
| T4-02 | PASS | Anonymous public Wikitext import persisted 29,119 documents across restart and removed temporary source files. |
| T4-03 | BLOCKED | Three discovered gated candidates all failed download; no access-term acceptance or gated report exists. The external authorization check is DEFERRED by the user and non-blocking. The 2026-10-20 environment had no supplied key or authorized gated repository, so no gated flow was attempted. |
| T4-04 | PASS | Isolated PostgreSQL 18.6 baseline initialization, prior concurrency, authenticated UI connection testing, live switching, settings isolation, restart recovery, injected rollback, and retry evidence passed. A matching schema stamped `0007_dashboard_ui_preferences` rebaselined to `0001_current_schema` after metadata comparison; runtime and dashboard preference JSON round-tripped on port 55432. The exact tagged v4.6.0 migration chain was also replayed and rebaselined on isolated SQLite. The host PostgreSQL listener was not used. |
| T5-01 | PASS | Restart reconciled active work to addressable terminal failure while preserving completed upload job and its 25,000-document dataset. |
| T5-02 | PASS | Dataset/Settings populated, no-match, empty Keys, four viewport sizes, keyboard navigation, Escape, and focus return passed without document-level horizontal overflow. |
| T5-03 | PASS | Cross Benchmark ordinary and cloned wizard states passed at narrow widths, including internal scroll, fixed actions, step reset, empty-dataset Next guard, and invalid-trial Start guard. Report-manager search/open/close/focus checks passed at 390 px; pagination beyond the first page was unavailable with two reports. Existing Tokenizers empty/loading/error/populated/report states, manager bounds, and viewport/keyboard/focus coverage remain passing. |
| T5-04 | PASS | Official launcher streamed visible progress for 10,000 documents, cancellation produced no report, eight backend RSS samples were collected, and a two-document rerun rendered without browser/HTTP errors. Browser heap was not instrumented. |
| T5-05 | OUT_OF_SCOPE | Release support is Windows x64 source-only. Ubuntu was diagnostic; macOS and hosted runtime support are not planned. |
| T5-06 | PASS | Releases `v4.5.0`, `v4.5.1`, `v4.6.0`, `v4.6.1`, and `v4.7.1` were verified. v4.7.1 (backend 3.7.1, frontend 2.7.1) was published from annotated tag `v4.7.1` at main commit `b8ae9fe5fdb11126387cc440903ef73f0a555b22`; hosted CI run `39800902959` passed frontend, backend, and Windows launcher jobs. Release URL and prior release evidence remain recorded above. The 2026-10-24 Phase 2 follow-up is PASS under V-20261024; T4-03 gated Hugging Face access and physical low-disk recovery remain validation debt. |

| T5-07 | PASS | The full browser flow passed on Chromium: 2,000-character input capping, max-three selection, request flags for special tokens on/off, rendered token text and IDs, per-item error card, collapse/reopen, and 1440px/375px responsive checks without document overflow. The 2026-10-23 audit additionally verified search match/no-match/clear, selected-chip retention, malformed/valid local tokenizer upload, and a 390×844 viewport without document overflow. No browser errors occurred after valid upload; external Hugging Face access is tracked separately. |

## Open issues and validation debt

| ID or component | Status | Durable limitation and follow-up |
| --- | --- | --- |
| ISSUE-002: deployment.network-auth-boundary | OPEN / HIGH | Network-hosted deployment still requires an external authentication and authorization boundary before exposing key-management or destructive routes. This is outside the supported local/source-only release model. Validate the external boundary before any network deployment. See [deployment](runtime/deployment.md#constraints) and [configuration security controls](runtime/configuration.md#security-controls). |
| ISSUE-003: architecture.canonical-ownership | OPEN / MEDIUM | Environment-derived paths, duplicate metric representations, export dictionaries, handwritten frontend contracts, catalog metadata, and toolchain declarations still have planned canonicalization follow-ups. No current regression was reproduced, and the architecture boundary test passed in the 2026-10-20 suite. See [canonical-source remediation](architecture/canonical_source_remediation.md#remaining-canonicalization-work). |
| data.large-file-and-disk-exhaustion | PARTIAL | Controlled `SQLITE_FULL` cleanup and same-name retries passed for upload and dataset-backed import in the 2026-10-20 suite. Physical host exhaustion was not re-attempted; no `fsutil` file or bounded-volume test was run because this is optional, non-blocking validation debt for the local source-only quality gate. The gate remains PARTIAL/DEFERRED and is not claimed as PASS; rerun only if a safe host-authorized volume is available. |
| integration.huggingface-discovery-and-download / T4-03 | PARTIAL | Public provider flow passed, but no authorized gated repository completed download/report generation. The gate is BLOCKED and the external recheck is DEFERRED; rerun only when an already-authorized gated repository is explicitly selected. No supplied key or gated repository was available on 2026-10-20. |
| deployment.containerized | NOT_IMPLEMENTED | No container deployment is part of the current product scope. |
| deployment.binary-packaging | NOT_IMPLEMENTED | The supported artifact is the source-only GitHub Release; do not infer installer or executable support. |

## Release gate

The previous published release was `v4.5.1`, pointing to commit
`ec74b7567f967c1948d362c761fa9b12438d5a10`. Hosted CI run
[36840557312](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/runs/36840557312)
passed both jobs for that SHA; the annotated tag and
[public source-only release](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/releases/tag/v4.5.1)
published on 2026-10-01 were verified. The pre-candidate remote refs are
`develop`=`3b09ba0` and `main`=`d49b247`: develop is four commits ahead and
main has no unique commits. Those four commits add database connection
management and database-scoped runtime settings, plus the dashboard
preference feature; the full release diff also contains README/documentation
consolidation, license headers, test updates, and formatting. No dependency
changes or API/report schema version changes were found.

The MINOR release `v4.6.0` (backend `3.6.0`, frontend `2.6.0`) was published as a source-only GitHub Release on 2026-10-21. Annotated tag `v4.6.0` was created while checked out on `main`, pushed, and verified to peel to release commit `2ecd12119b86864d617367bd6b0712bb9a1b2b92`; the public release is https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/releases/tag/v4.6.0. Before tagging, `main` and `develop` were aligned at that commit. Candidate/develop CI run `39557902883` and final main CI run `39582304529` passed both jobs; the final run validates the exact release commit. Main smoke included the production rebuild, isolated SQLite startup and UI checks, and the focused live E2E suite (7/7 on identical application code; subsequent changes were documentation-only). After publication, the release documentation update was committed to `develop`, fast-forwarded to `main`, and pushed to both branches. The first candidate E2E attempt used the repository default data root due to bootstrap `.env` precedence; cleanup assertions restored logical settings/config and removed synthetic test data, but there is no pre-run binary database snapshot to verify byte-for-byte restoration.
T4-03
gated Hugging Face access and physical low-disk recovery remain independently
bounded, non-blocking validation debt as described above.

The PATCH release `v4.6.1` (backend `3.6.1`, frontend `2.6.1`) was published
as a source-only GitHub Release on 2026-10-21:
[v4.6.1](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/releases/tag/v4.6.1).
Annotated tag `v4.6.1` was created from `main`, pushed, and verified to peel
to release commit `d198f203525bb57603060db7283f7d01432269b3`; remote `main`
and `develop` both pointed to that commit before tagging. Focused local checks
passed 204 backend tests and 69 frontend tests, frontend lint/build, backend
compileall, Ruff, BasedPyright (0 errors; 2,093 warnings), OpenAPI smoke (38
paths), lock verification, and isolated startup/browser smoke. Hosted main CI
run [39658313798](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/runs/39658313798)
passed both frontend and backend jobs on the release commit. GitHub reports no
uploaded binary assets; its source archives are available from the release
page. The post-publication documentation update was synchronized to both
branches. The first frontend test invocation failed before collection with a
sandbox temporary-directory `EPERM`; rerunning with the project cache as
`TEMP`/`TMP` passed. The live smoke restored `settings/.env` byte-for-byte,
stopped its processes, freed ports 5000/8000, and removed its temporary output
and reviewed disposable screenshots/logs. T4-03 gated Hugging Face access and
physical low-disk recovery remain non-blocking validation debt as described
above.

The PATCH release `v4.7.1` (backend `3.7.1`, frontend `2.7.1`) was published
as a source-only GitHub Release on 2026-10-22:
[v4.7.1](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/releases/tag/v4.7.1).
Annotated tag `v4.7.1` was created from `main` and verified to peel to release
commit `b8ae9fe5fdb11126387cc440903ef73f0a555b22`; `main` and `develop` pointed
to that commit before tagging. Hosted CI run
[39800902959](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/runs/39800902959)
passed the frontend, backend, and Windows launcher jobs on that commit. The
release has no uploaded binary assets. Before publication, the merged
`README-design`, `fix/tkben-spa-lifecycle-20261022`, and detached-WIP branches
were verified to hold no unique commits and were removed, leaving only `main`
and `develop`.

The previous gate record remains historical: the v4.5.0 release commit
`f8dee1da9084138bd52f3d08437269d3821ba5d6` passed hosted CI run
[36158666979](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/runs/36158666979),
and the post-release data-root rename and bytecode-cache hardening delta that
had been recorded as unvalidated code/documentation changes at `a84b9c1` was
closed by the V-20261017 campaign.

The MINOR release `v4.8.0` (backend `3.8.0`, frontend `2.8.0`) was published
as a source-only GitHub Release on 2026-10-25:
[v4.8.0](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/releases/tag/v4.8.0).
Annotated tag `v4.8.0` was created on `main` after a fast-forward from
`develop` and peels to `955a151847db2af066d6f0ef2016ab8522b10115`. Hosted CI
run `40225854600` passed all three jobs. The release has no uploaded binary
assets. Validation evidence is V-20261025; launcher validation debt remains
listed under open issues.

The PATCH release `v4.8.1` (backend `3.8.1`, frontend `2.8.1`) was published
as a source-only GitHub Release on 2026-10-26:
[v4.8.1](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/releases/tag/v4.8.1).
Annotated tag `v4.8.1` was created on `main` after a fast-forward from
`develop` and peels to `45b7b0fc7db94a84b0b587526e4f21842a5e228b`. Hosted CI
run `40308567067` passed all three jobs on that commit. The release has no
uploaded binary assets. Validation evidence is V-20261026.

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
