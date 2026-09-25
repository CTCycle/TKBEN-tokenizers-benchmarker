# TKBEN T5-06 current-candidate validation

Date: 2026-09-25

Implementation under test: application revision `21ed4ff6d32fceb5b8e6521e96b1ac9dcd0c75d2`, the clean `develop` tip before this evidence record was added.

## Scope and selection

The current ledger had closed the Tier 0 through Tier 5 functional slices except
for the current-candidate hosted-CI/release gate (`T5-06`). This task selected
that gate because it is actionable from the clean checkout. The gated Hugging
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
provider credential, live external provider, or application service was used
for this local suite.

## Hosted and release boundary

The evidence commit was pushed to `develop` as `4c5cbc98805adf249e4a83b0605c0bf2d5a9ad30`.
Hosted CI run [36148106588](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/runs/36148106588)
completed successfully for that exact SHA. Both
[`backend-validation`](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/runs/36148106588/job/108114229311)
and
[`frontend-validation`](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/runs/36148106588/job/108114229626)
completed with `success`. No `main` synchronization, annotated tag, or GitHub
Release publication is part of this task; therefore `T5-06` and source-only
release readiness remain `PARTIAL` until that separate release workflow is
completed.

## Remaining limits

- `T4-03` remains `BLOCKED`: the supplied Hugging Face key authenticated, but
  no already-authorized gated repository completed download/report generation.
- `data.large-file-and-disk-exhaustion` remains `PARTIAL`: SQLite-full cleanup
  and retry behavior are covered, but host authorization denied creation of the
  bounded volume needed for physical exhaustion/recovery.
- `T5-05` remains `OUT_OF_SCOPE` for the Windows x64 release target.
