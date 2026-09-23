# T3-03 / T3-04 Populated Cross Benchmark Workflow Validation

- Date: 2026-09-23
- Result: PASS for T3-03 and T3-04; PARTIAL for the Cross Benchmark portion of T5-03
- Environment: Windows, official `start_on_windows.ps1 -Launch`, isolated SQLite/data and log roots under `assets/QA/tkben-t3-03-04-live-20260923`, backend port 54321, production preview port 54322, installed Chrome, and Codex in-app Browser review.

## Scope and live scenario

The opt-in Playwright scenario in [test_cross_benchmark_report_workflows.py](../../app/tests/e2e/test_cross_benchmark_report_workflows.py) used local inputs only and left report/job APIs unmocked. It uploaded a two-document CSV and two small local WordLevel tokenizers, then created one-tokenizer and two-tokenizer reports through the rendered benchmark wizard. It seeded 24 lightweight current-schema manager rows in the isolated database, retaining both live wizard reports for the populated dashboard scenario.

The scenario exercised:

- Report-manager server search, 25/1 pagination, inline tag save and reload, delete confirmation/cancel, confirmed deletion, and persisted-row `404` verification.
- Baseline selection against a real two-tokenizer report, comparison-strip and data-table deltas, browser-storage persistence, report reload, and restored selector rendering.
- Clone eligibility and copied dataset/tokenizer/trial configuration.
- Visualization switching, keyboard widget reordering, hidden-widget customization, reset/default restoration, and persistence across reload.
- Dashboard horizontal-overflow checks at 1920x1080, 1440x900, 1024x768, and 390x844. The report-manager, clone, and customization dialogs were also checked against each viewport's bounds; Escape dismissal and focus return were checked for keyboard operation.
- The one-tokenizer chart tick label geometry against its `Tokenizer` axis title, browser console/page errors, and HTTP error responses.

The API-backed setup and cleanup left no matching test reports. The isolated API returned zero reports, datasets, and tokenizers after the run. All created report rows, tag edits, custom tokenizer files, and the test dataset were deleted. The original `settings/.env` was restored byte-for-byte after the temporary isolated launcher configuration. The verified launcher backend and preview process tree were stopped; ports 5000, 8000, 54321, and 54322 were then clear. The disposable SQLite/data root and launcher logs were removed after verification; the reviewed screenshots and this durable QA record remain.

## Findings fixed

- The one-tokenizer chart's rotated, end-anchored label crowded the `Tokenizer` axis title. A single point now uses a centered horizontal tick label above the axis title; multi-point charts keep their rotated labels. The live scenario asserts the label geometry and accessible tokenizer name.
- Baseline preference storage restored the comparison strip, but the rendered `<select>` showed `None` after reload. The generated options now reflect the stored baseline; a store unit test covers restoration from browser storage.
- The populated responsive dialog review found that the startup shell's animation fill left an identity transform on an ancestor, changing fixed dialog positioning. The shell animation now returns to its normal non-transformed state after completion.
- Angular CDK's `Dialog closed` live-announcer text appeared visibly at the bottom of the page. Its visually hidden class now follows the existing one-pixel clipped accessibility pattern; the E2E verifies the announcement remains accessible and visually hidden.

## Checks and evidence

The live E2E was run against the official launcher with the isolated `.env` values and these PowerShell commands:

```powershell
$env:E2E_RUN_BENCHMARKS = '1'
$env:PYTHONPATH = 'app'
& .\app\server\.venv\Scripts\python.exe -m pytest app/tests/e2e/test_cross_benchmark_report_workflows.py --browser chromium --browser-channel chrome --base-url http://127.0.0.1:54322
```

Result: **1 passed in 18.88s**. The launcher reported backend health and preview readiness. The browser observed no page exceptions, console errors, or HTTP responses at status 400 or above.

Other focused checks:

- Backend: `& .\app\server\.venv\Scripts\python.exe -m pytest app/tests/unit/server/services/test_benchmark_reports.py app/tests/unit/server/api/test_benchmarks_routes.py`; **16 passed in 3.96s**.
- Frontend unit: `& .\runtimes\nodejs\npm.cmd --prefix app/client run test:unit`; **14 files, 62 tests passed**.
- Frontend lint and production build: `& .\runtimes\nodejs\npm.cmd --prefix app/client run lint` and `& .\runtimes\nodejs\npm.cmd --prefix app/client run build`; both passed after the chart, baseline, and CSS fixes.
- Ruff: `& .\app\server\.venv\Scripts\python.exe -m ruff check app/tests/e2e/test_cross_benchmark_report_workflows.py`; passed.
- BasedPyright: with `$env:PYTHONPATH = 'app'`, `& .\app\server\.venv\Scripts\python.exe -m basedpyright --level error app/tests/e2e/test_cross_benchmark_report_workflows.py`; **0 errors, 0 warnings**. The default warning-level invocation reports 96 dynamic Playwright/JSON typing warnings and zero errors; the repository gate is configured not to fail on warnings.
- `git diff --check`; passed. Git emitted its existing LF-to-CRLF working-copy normalization notice for `styles.css`.

Visual evidence captured from the populated report:

- [Single-tokenizer chart after label fix](tkben-t3-03-single-tokenizer-chart-20260923.png)
- [Cross Benchmark at 1920x1080](tkben-t3-03-04-cross-benchmark-1920x1080-20260923.png)
- [Cross Benchmark at 1440x900](tkben-t3-03-04-cross-benchmark-1440x900-20260923.png)
- [Cross Benchmark at 1024x768](tkben-t3-03-04-cross-benchmark-1024x768-20260923.png)
- [Cross Benchmark at 390x844](tkben-t3-03-04-cross-benchmark-390x844-20260923.png)
- [Report manager at 390x844](tkben-t3-03-report-manager-390x844-20260923.png)

## Gate results and remaining limitations

| Gate | Final status | Evidence or remaining work |
| --- | --- | --- |
| T3-03 report management | PASS | Live persisted search, pagination, tags, confirmation, and deletion across 26 reports. |
| T3-04 dashboard persistence | PASS | Baseline/delta table, reload, clone settings, visualization/order/customization persistence. |
| T5-03 Cross Benchmark responsive/keyboard slice | PARTIAL | Populated Cross Benchmark page and three dialogs reviewed at four sizes; Tokenizers responsive and keyboard coverage remains. |
| T3-01 / T3-02 | PARTIAL | Controlled performance/resource campaign and advanced benchmark-flag semantics remain. |
| T3-05 | PARTIAL | Dataset, tokenizer, and benchmark PDF download/render parity remains uninspected. |
| T4-01 / T4-03 | BLOCKED | Public/gated Hugging Face provider checks need approved network/credentials. |
| T4-02 | UNTESTED | Public dataset download and disk validation remain. |
| T4-04 | BLOCKED | PostgreSQL equivalence requires a disposable database and credentials. |
| T5-02 | UNTESTED | Dataset/Settings responsive and keyboard matrix remains. |
| T5-04 | UNTESTED | Resource, streaming, and responsiveness campaign remains. |
| T5-05 | PARTIAL | Linux manual startup is recorded; macOS and hosted `ubuntu-latest` remain. |
| T5-06 | PARTIAL | Hosted CI passed for implementation commit `dc8cae26ea81b1c621c800168d5b3581c9fab04f`; release publication evidence remains unchecked. |

The broader responsive visual matrix is PARTIAL: this report flow passed its populated-state viewport checks, while Tokenizers, Dataset, and Settings routes and empty/loading/error/long-identifier states remain. The dashboard/PDF component remains WORKING until populated-report PDF download and rendered parity are checked. Containerized deployment and binary packaging remain NOT_IMPLEMENTED by design.

Hosted workflow [run 35875638854](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/runs/35875638854) completed successfully for pushed implementation commit `dc8cae26ea81b1c621c800168d5b3581c9fab04f`. Both `frontend-validation` and `backend-validation` jobs succeeded. This establishes CI evidence only; release publication remains outside this validation scope, so T5-06 remains PARTIAL.
