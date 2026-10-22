# Release Procedure
Last updated: 2026-10-21

## Release model

TKBEN publishes source-only GitHub Releases. The repository has no Tauri,
installer, executable, portable-app, or other binary packaging workflow. Do not
add packaging as part of a source release.

The public release version and component versions use the existing repository
convention:

| Surface | `v3.9.0` | `v4.0.0` | `v4.1.0` | `v4.2.0` | `v4.3.0` | `v4.4.0` | `v4.5.0` | `v4.5.1` | `v4.6.0` | `v4.6.1` | `v4.7.0` public |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Public Git tag and GitHub Release | `3.9.0` | `4.0.0` | `4.1.0` | `4.2.0` | `4.3.0` | `4.4.0` | `4.5.0` | `4.5.1` | `4.6.0` | `4.6.1` | `4.7.0` |
| Backend package (`app/server/pyproject.toml`) | `2.4.0` | `3.0.0` | `3.1.0` | `3.2.0` | `3.3.0` | `3.4.0` | `3.5.0` | `3.5.1` | `3.6.0` | `3.6.1` | `3.7.0` |
| Frontend package (`app/client/package.json`) | `1.4.0` | `2.0.0` | `2.1.0` | `2.2.0` | `2.3.0` | `2.4.0` | `2.5.0` | `2.5.1` | `2.6.0` | `2.6.1` | `2.7.0` |

The latest published release is `v4.6.1` (backend `3.6.1`, frontend `2.6.1`);
`v4.7.0` (backend `3.7.0`, frontend `2.7.0`) was published on 2026-10-22 from
annotated tag `v4.7.0` on `main` commit `f628688e325bcb7d903fbe49e830b0f32ddb9248`
(hosted CI run `39750287164` passed both jobs; source-only, no binary assets).
Its annotated tag points to validated `main` commit
`d198f203525bb57603060db7283f7d01432269b3`; publication and validation evidence
are recorded in the [project status ledger](../project_status_ledger.md#release-gate).

## v4.6.0 release notes

The v4.6.0 delta is based on verified repository changes after `v4.5.1`:

- add a Database panel to configure, test, and switch SQLite or PostgreSQL
  connections, with encrypted PostgreSQL password storage and rollback when
  activation fails; legacy database environment values are imported once;
- save application runtime settings and dashboard layout, visibility, chart,
  and report-baseline preferences automatically to the active database, with
  separate revisions and database-local values; and
- add Alembic revisions `0006_database_runtime_settings` and
  `0007_dashboard_ui_preferences` for those persisted settings.

API version `1.2.0`, benchmark schema version `3`, and report version `5`
remain unchanged. The public release status and final commit are recorded in
the [project status ledger](../project_status_ledger.md#release-gate).

## v4.7.0 release notes

The v4.7.0 delta is based on verified repository changes after `v4.6.1`:

- serve the production Angular build from FastAPI at `/` with SPA fallback
  (unknown `/api` paths and missing assets return 404), and keep these
  frontend routes out of the OpenAPI schema;
- update the Windows launcher to use the single-origin FastAPI SPA flow and
  drop the obsolete UI-port configuration; and
- revamp the README overview, features, and design.

API version `1.2.0`, benchmark schema version `3`, and report version `5`
remain unchanged. The Windows lifecycle hardening in draft PR #8 is not part
of this release; its open standard-console Ctrl+C verification remains
validation debt in the [project status ledger](../project_status_ledger.md#open-issues-and-validation-debt).
Validation evidence is recorded as `V-20261022` (release smoke).

## v4.6.1 patch release notes

The v4.6.1 release increments the public release, backend, and frontend
patch versions to `4.6.1`, `3.6.1`, and `2.6.1`. The verified changes since
v4.6.0 include:

- allow `CORS_ORIGINS` to be configured through bootstrap environment settings,
  with normalization and startup validation; it remains outside the 16-field
  application Settings API bundle;
- remove the deprecated `/api/tokenizers/settings` endpoint and obsolete
  database/runtime-settings import paths; `/api/settings` is the settings
  contract;
- consolidate schema creation at `0001_current_schema`, while safely
  rebaselining the exact published v4.6.0 head only when its schema matches
  current metadata; and
- remove unused tokenizer and report compatibility fallbacks.

API version `1.2.0`, benchmark schema version `3`, and report version `5`
remain unchanged.

Focused local regression tests passed (204 backend tests and 69 frontend
tests); frontend lint/build, backend compileall, Ruff, BasedPyright, OpenAPI,
dependency-lock verification, and isolated startup/browser smoke passed.
Hosted CI run [39658313798](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/runs/39658313798)
passed frontend and backend jobs on release commit
`d198f203525bb57603060db7283f7d01432269b3`. Annotated tag `v4.6.1` and the
source-only GitHub Release were published from `main` on 2026-10-21; the tag
peels to that commit. `main` and `develop` were synchronized before tagging
and again after the publication record update. The release has no uploaded
binary assets; GitHub provides the source archives. Remaining bounded
validation debt is recorded in the [project status ledger](../project_status_ledger.md#open-issues-and-validation-debt).

## v4.5.1 release notes

The current release delta is based on verified repository changes after
`v4.5.0`:

- redesign the startup tokenizer flow into a single-lane pipeline where source
  words enter a schematic tokenizer and leave as `##`-prefixed subword tokens,
  with updated loading component, style, unit, and browser coverage;
- move the canonical data root to the top-level `data` directory and align the
  launcher contract, upload contract, and documentation with the post-release
  layout;
- clear backend Python bytecode caches during launcher cleanup; and
- trim low-value and duplicate tests from the suite.

API version `1.2.0`, benchmark schema version `3`, and report version `5`
remain unchanged. The current validation outcome, remaining limitations, and
release traceability are recorded in the [project status
ledger](../project_status_ledger.md#release-gate).

## v4.5.0 release notes

The current release delta is based on verified repository changes after
`v4.4.0`:

- add durable managed-job lifecycle metadata, restart reconciliation, and the
  Alembic `0005_managed_job_lifecycle` migration;
- apply configured benchmark parallelism and strengthen progress, cancellation,
  resource-observation, and immediate-rerun behavior;
- complete dataset and tokenizer catalog filtering, metric dashboards, custom
  tokenizer persistence, vocabulary reporting, and cross-benchmark report
  management;
- expand dashboard visualization and source-only PDF export coverage; and
- harden the Windows launcher readiness, cache/process cleanup, and local
  source-distribution validation path.

API version `1.2.0`, benchmark schema version `3`, and report version `5`
remain unchanged. The current validation outcome, remaining limitations, and
release traceability are recorded in the [project status
ledger](../project_status_ledger.md#release-gate).

## v4.4.0 release notes

The current release delta is based on verified repository changes after
`v4.3.0`:

- add persisted benchmark report tags, report management, and benchmark cloning;
- add report-scoped baseline selection and dashboard layout persistence;
- improve catalog filtering, settings validation, benchmark defaults, chart
  controls, histogram/CDF views, token-shape metrics, and bounded word clouds;
- centralize disposable caches under `runtimes/cache` and add deterministic
  Windows launcher cleanup and `-KillAll` process maintenance;
- keep the launcher’s source-stale frontend detection, local health checks, and
  actionable redirected backend diagnostics in the supported source-only flow.

API version `1.2.0`, benchmark schema version `3`, report version `5`, and
Alembic revision `0004_benchmark_report_tags` remain unchanged. Its durable
validation history is summarized in the [project status
ledger](../project_status_ledger.md#resolved-and-historical-findings).

## v4.3.0 release notes

The current release delta is based on verified repository changes after
`v4.2.0`:

- centralize configuration ownership between `.env` and structured JSON;
- load the environment before configuration and database imports;
- reject invalid boolean aliases, missing structured settings, and unknown
  configuration blocks;
- remove legacy configuration, namespace, cache, and tokenizer compatibility
  paths in favor of the canonical implementations;
- keep custom tokenizer identity and its canonical `tokenizer.json` artifact
  together across service recreation and persistence;
- make the Windows launcher build missing or source-stale Angular production
  output before preview, verify the required `index.html` and build stamp, and
  fail when an existing port listener cannot be stopped; redirected launches
  now capture backend logs for actionable health-check failures.

API version `1.2.0`, benchmark schema version `3`, report version `5`, and
Alembic revision `0003_canonical_state_cleanup` remain unchanged. Its durable
validation history is summarized in the [project status
ledger](../project_status_ledger.md#resolved-and-historical-findings).

## Preparation and validation

1. Start from the current `develop`, inspect `git status`, the previous release
   tag, and the commits on `develop` since that tag. Preserve unrelated local
   work and do not make release-preparation edits directly on `main`.
2. Update the README, `assets/docs`, and this release procedure before the
   final branch synchronization. Keep documentation version references
   consistent with the release candidate.
3. Run the CI-equivalent checks for the intended release surfaces: backend
   compileall, Ruff, BasedPyright, unit tests, and OpenAPI smoke; frontend
   `npm run lint`, `npm run test:unit`, and `npm run build` from `app/client`.
4. Launch with `start_on_windows.ps1 -Launch`, verify the backend health
   endpoint and frontend production entry, then exercise Dataset, Tokenizers,
   Settings → Database, and Cross Benchmark routes. Prioritize the changed
   database test/switch/rollback behavior, per-database runtime settings,
   automatic dashboard preference persistence, and real report workflows.
5. Run the focused live API/UI tests when the local services are available.
   Inspect browser console output and application logs; classify expected
   test-injected failures separately from release-blocking errors.
6. Record the current status, evidence summary, skipped checks, unavailable
   providers, and unresolved limitations in the [project status
   ledger](../project_status_ledger.md). Do not claim skipped, unavailable-
   provider, or unverified checks as passed; retain raw logs only as temporary
   run output when they are needed for diagnosis.

## Versioning and synchronization

After validation is release-ready, apply the coordinated version bump to the
public tag version, backend package, frontend package, README, and relevant
documentation. Commit the candidate and its evidence on `develop` before
synchronizing branches.

Synchronize `main` from the validated `develop` commit so the branches point to
the same tree. Verify both branch tips and the clean worktree before creating
an annotated tag named `vX.Y.0` with `main` checked out at the synchronized
commit. Never create a release tag from `develop` or another branch.

## Publication

Create the GitHub Release from the annotated tag and use the release summary to
describe the reviewed delta and validation evidence. The source archive is the
only release artifact. After publication, verify that the tag and release point
to the synchronized `main` commit and that `develop` remains aligned with it.
