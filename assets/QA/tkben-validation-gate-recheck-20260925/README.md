# TKBEN validation-gate re-audit

Date: 2026-09-25. Checkout: `develop` at `f83424f4c8cdc7e3499d99f23dfc1b9ba10b4977`. The application implementation remains the current candidate recorded by the ledger at `21ed4ff6d32fceb5b8e6521e96b1ac9dcd0c75d2`; the checkout changes after that implementation are validation and documentation records only.

## Scope and result

The current validation ledger was reviewed before selecting this slice. All
functional Tier 0 through Tier 3 gates and the PostgreSQL/runtime gate are
already closed in the ledger. The actionable re-audit covered:

- a fresh local quality and startup check against the current checkout;
- rendered empty/default route states for Dataset, Tokenizers, Cross Benchmark,
  and Settings;
- the remaining Hugging Face gated-provider boundary;
- the physical low-disk recovery boundary; and
- the current-candidate hosted-CI/release boundary.

No application defect was reproduced and no production-code change was needed.
The residual statuses remain honest: T4-03 is `BLOCKED`, physical disk
exhaustion is `PARTIAL`, and T5-06/source release readiness is `PARTIAL`.

## Current validation-gate inventory

| Gates | Final status | Current boundary |
| --- | --- | --- |
| T0-01 through T0-03 | PASS | Revision/quality baseline, launcher contract, and maintenance menu evidence remain valid; the fresh local checks below found no regression. |
| T1-01 through T1-05 | PASS | Startup, settings, key management, persistence/configuration, and catalogue-filter/race slices remain closed. The broader `configuration.runtime-settings` component remains `WORKING` by ledger scope, while T1-02 itself is PASS. |
| T2-01 through T2-07 | PASS | Dataset, metric, tokenizer, report, wizard, and cancellation slices remain closed. |
| T3-01 through T3-05 | PASS | Benchmark measurements, parallelism, report workflows, dashboard/PDF export, and the expanded chart/vocabulary review remain closed. |
| T4-01 | PASS | Public Hugging Face discovery/download/report flow is validated. |
| T4-02 | PASS | Public Wikitext dataset download, restart persistence, and source cleanup are validated. |
| T4-03 | BLOCKED | No authorized gated-repository download/report is evidenced; see the provider boundary below. |
| T4-04 | PASS | Isolated PostgreSQL runtime equivalence and rollback/retry evidence is recorded in the ledger. |
| T5-01 through T5-04 | PASS | Restart recovery, responsive/accessible route coverage, populated workflows, and progress/cancellation evidence remain closed. |
| T5-05 | OUT_OF_SCOPE | macOS/hosted-Linux release validation is excluded from the supported Windows x64 source-only target. |
| T5-06 | PARTIAL | Current-candidate hosted CI is green, but no intentional `main` synchronization, annotated tag, or public release was performed. |

## Fresh local evidence

| Check | Result |
| --- | --- |
| Backend unit suite: `app/server/.venv/Scripts/python.exe -m pytest -c app/tests/pytest.ini app/tests/unit --confcutdir=app/tests/unit --basetemp=runtimes/cache/pytest-gate-recheck-20260925 -p no:cacheprovider -q` | PASS: 517 passed, 24 warnings, 47.07 seconds. The expected warning reports that `cache_dir` is unknown because the cache plugin was disabled for the isolated run. |
| Ruff: `app/server/.venv/Scripts/python.exe -m ruff check app/server app/tests` | PASS. A non-failing access-denied warning came from the existing protected test-cache subtree. |
| BasedPyright: `app/server/.venv/Scripts/python.exe -m basedpyright -p app/server/pyrightconfig.json` | PASS: 0 errors, 2,018 warnings. |
| Frontend unit tests: `npm --prefix app/client run test:unit` | PASS: 14 files, 63 tests. |
| Frontend lint: `npm --prefix app/client run lint` | PASS: all files pass linting. |
| Frontend production build: `npm --prefix app/client run build` | PASS: Angular production bundle generated. |
| `git diff --check` | PASS. |

The supported launcher then started the backend and production preview. Direct
checks returned HTTP 200 from `http://127.0.0.1:5000/api/health` and
`http://127.0.0.1:8000/`. The Codex in-app browser rendered these current
checkout states:

- `/dataset`: Dataset filters and the empty state `No datasets match the current filters.`;
- `/tokenizers`: Tokenizer filters, empty preview, and the disabled empty dashboard;
- `/cross-benchmark`: Reports (0), no selected report, and `No benchmark reports available.`;
- `/settings`: the Data section with default controls and disabled `SAVE CHANGES`.

The task-owned backend and preview process tree was stopped after inspection;
ports 5000 and 8000 were clear. Only the five launcher log files created by
this smoke run were removed; existing application data under `app/resources`
was not removed or modified.

## Remaining boundaries

### T4-03: gated Hugging Face provider

The current environment had no opt-in `TKBEN_TEST_HF_KEY`,
`TKBEN_TEST_HF_USE_STORED_KEY`, or `TKBEN_TEST_HF_GATED_REPO` setting. No
credential was read, entered, or transmitted, and the live provider test was
not rerun after the earlier isolated credential cleanup. The same-day provider
record remains the current evidence: Hub authentication and the public
`bert-base-uncased` flow passed, but the three discovered gated candidates all
failed with `provider_denied_or_download_failed`; no access terms were
accepted and no gated report was generated.

Final status: `BLOCKED`.

Next action: rerun the opt-in flow only when a user-authorized key and an
already-authorized gated repository are available. A gated report must be
downloaded, persisted, rendered after reload, and cleaned up before T4-03 can
be promoted.

### Physical disk exhaustion

The same-day fixed-size VHD attempt was denied by the host OS before a volume
was created. Consequently no full-disk upload/import scenario ran, and no new
claim is made beyond the already validated simulated `SQLITE_FULL` cleanup and
retry behavior.

Final status: `PARTIAL`.

Next action: repeat the upload and dataset-backed import failure/recovery flow
on a host-authorized bounded volume, then verify incomplete-row cleanup and a
same-name retry after capacity is restored.

### T5-06/source release

The current-candidate hosted-CI result is already recorded as successful for
the exact evidence commit. This re-audit did not synchronize `main`, create an
annotated tag, or publish a GitHub Release because that is a separate
intentional release operation, not an implementation validation step.

Final status: `PARTIAL`.

Next action: perform the coordinated release procedure only when publication
is explicitly selected, including version consistency, `main` synchronization,
annotated tag creation, release publication, and post-publication verification.

## Cleanup and retained evidence

No credential, database, tokenizer, dataset, or report was created by this
re-audit. The existing same-day open-debt record remains the source for the
provider, VHD, and release audit artifacts:

- [open validation debt](../tkben-open-validation-debt-20260925/README.md)
- [current-candidate release validation](../tkben-t5-06-current-candidate-20260925/README.md)
