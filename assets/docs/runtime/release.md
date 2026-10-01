# Release Procedure
Last updated: 2026-10-01

## Release model

TKBEN publishes source-only GitHub Releases. The repository has no Tauri,
installer, executable, portable-app, or other binary packaging workflow. Do not
add packaging as part of a source release.

The public release version and component versions use the existing repository
convention:

| Surface | `v3.9.0` | `v4.0.0` | `v4.1.0` | `v4.2.0` | `v4.3.0` | `v4.4.0` | `v4.5.0` | `v4.5.1` public |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Public Git tag and GitHub Release | `3.9.0` | `4.0.0` | `4.1.0` | `4.2.0` | `4.3.0` | `4.4.0` | `4.5.0` | `4.5.1` |
| Backend package (`app/server/pyproject.toml`) | `2.4.0` | `3.0.0` | `3.1.0` | `3.2.0` | `3.3.0` | `3.4.0` | `3.5.0` | `3.5.1` |
| Frontend package (`app/client/package.json`) | `1.4.0` | `2.0.0` | `2.1.0` | `2.2.0` | `2.3.0` | `2.4.0` | `2.5.0` | `2.5.1` |

The latest published release is `v4.5.1`. Its component versions are backend
`3.5.1` and frontend `2.5.1`.

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
   endpoint and frontend production entry, then exercise Dataset, Tokenizers, and Cross
   Benchmark routes. Prioritize flows changed since the previous release,
   including current schema-3/report-5 persistence, catalog filtering, custom
   tokenizer handling, vocabulary-shape metrics, histogram/CDF views, bounded
   word-cloud layout, and dashboard visualization controls.
5. Run the focused live API/UI tests when the local services are available.
   Inspect browser console output and application logs; classify expected
   test-injected failures separately from release-blocking errors.
6. Record the current status, evidence summary, skipped checks, unavailable
   providers, and unresolved limitations in the [project status
   ledger](../project_status_ledger.md). Do not claim skipped, unavailable-
   provider, or unverified checks as passed; retain raw logs only as temporary
   run output when they are needed for diagnosis.

## Versioning and synchronization

After validation is release-ready, apply the coordinated minor bump to the
public tag version, backend package, frontend package, README, and relevant
documentation. For the current preparation, the public version is `v4.5.1`,
the backend package is `3.5.1`, and the frontend package is `2.5.1`. Commit all
release-preparation changes on `develop` before synchronizing branches.

Synchronize `main` from the validated `develop` commit so the branches point to
the same tree. Verify both branch tips and the clean worktree before creating
an annotated tag named `vX.Y.0` from the synchronized `main` commit.

## Publication

Create the GitHub Release from the annotated tag and use the release summary to
describe the reviewed delta and validation evidence. The source archive is the
only release artifact. After publication, verify that the tag and release point
to the synchronized `main` commit and that `develop` remains aligned with it.
