# TKBEN T5-06 current-candidate validation

Date: 2026-09-25

Implementation under test: v4.5.0 release commit `f8dee1da9084138bd52f3d08437269d3821ba5d6`, with backend package `3.5.0` and frontend package `2.5.0`.

## Scope and selection

The current ledger had closed the Tier 0 through Tier 5 functional slices except
for the current-candidate hosted-CI/release gate (`T5-06`). This task selected
that gate from the clean `develop` checkout, validated the release candidate,
published the source-only release, and reconciled the ledger. The gated Hugging
Face route (`T4-03`) and physical filesystem-exhaustion recovery remain
external/environment-dependent limits and were not silently treated as passed.

## Local release-suite evidence

The CI-equivalent checks ran against the current implementation with the
repository-managed Python environment and an isolated disposable cache root.

| Check | Result |
| --- | --- |
| `python -m compileall app/server` | PASS |
| Ruff for `app/server` and `app/tests` | PASS; non-failing access-denied warnings came from an existing protected test-cache path |
| BasedPyright | PASS; 0 errors and 2,018 warnings |
| Embedded database initialization | PASS; Alembic head `0005_managed_job_lifecycle` |
| Backend unit suite | PASS; 517 passed, 23 warnings |
| OpenAPI smoke | PASS; 35 paths generated |
| Frontend lint | PASS |
| Frontend unit suite | PASS; 14 files and 63 tests |
| Frontend production build | PASS |
| `git diff --check` and post-run worktree check | PASS; no tracked changes from validation |

The database initialization check did not change the tracked checkout. No
provider credential or live external provider was used for this local suite;
the official launcher was used separately for the rendered browser captures.

## Rendered screenshot refresh

The current rendered Dataset, Tokenizers, and Cross Benchmark captures were
shown for review before replacing the README figure assets. The refreshed
assets are:

- [Dataset](../../figures/dataset.png): four-document quality, structure, and
  compression dashboard;
- [Tokenizers](../../figures/tokenizers-overview.png): the reviewed 1440x900
  populated 1,207-entry vocabulary report capture from the responsive
  validation run; and
- [Cross Benchmark](../../figures/cross-benchmark.png): populated report
  dashboard and chart grid from the current wizard workflow.

The full-page tokenizer candidate with duplicated sticky navigation was not
embedded and was removed as a redundant exploratory capture.

## Hosted and release boundary

The release-preparation commit was pushed to `develop` as
`f8dee1da9084138bd52f3d08437269d3821ba5d6`. Hosted CI run
[36158666979](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/runs/36158666979)
completed successfully for that exact SHA, with both backend and frontend jobs
successful. Remote `main` and `develop` were then verified after release
closure and remained aligned through the documentation-only evidence updates;
the tag and release remain anchored to the tested release commit.

The annotated tag `v4.5.0` was pushed with tag-object SHA
`4a49670c25a224f8218d7feb2fe07ec8e8690ae2`, peeling to the release commit.
The non-draft, non-prerelease source-only [GitHub Release v4.5.0](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/releases/tag/v4.5.0)
was published on 2026-09-25. T5-06 and source-only release readiness are now
`PASS`/`VALIDATED` respectively.

## Remaining limits

- `T4-03` remains `BLOCKED`: the supplied Hugging Face key authenticated, but
  no already-authorized gated repository completed download/report generation.
- `data.large-file-and-disk-exhaustion` remains `PARTIAL`: SQLite-full cleanup
  and retry behavior are covered, but host authorization denied creation of the
  bounded volume needed for physical exhaustion/recovery.
- `T5-05` remains `OUT_OF_SCOPE` for the Windows x64 release target.
