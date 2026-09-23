# T2-07 Benchmark Cancellation and Immediate Rerun

Last updated: 2026-09-23
Date: 2026-09-23
Result: PASS for local long-running cancellation, no-report, and immediate-rerun behavior
Validated implementation commit: `cc91fec1b7307ae59091dc9e522627d8b69c6924`
Environment: Windows, official `start_on_windows.ps1 -Launch`, isolated SQLite data and log roots under `runtimes/cache/t2-07-validation-data-20260923` and `runtimes/cache/t2-07-validation-logs-20260923`, API on port 5000, production preview on port 8000, installed Chrome.

## Coverage

- The opt-in browser test uploaded a generated 10,000-document CSV and a local tokenizer, then started a 10,000-document benchmark with 200 timed trials from the populated Cross Benchmark wizard.
- API polling observed the managed job as `running` at 20% or greater before the test used the page-level **Cancel benchmark** action. The cancellation request was accepted, the job reached terminal `cancelled`, no report was returned for that run name, and the page restored the enabled **Run benchmark** action.
- The test immediately started a two-document run with one timed trial. It completed, returned a persisted report, and rendered the matching report name and chart in the page.
- No browser page exceptions, console errors, or failed HTTP responses were observed. The [running screenshot](tkben-t2-07-benchmark-cancellation-running-20260923.png) shows the accessible cancellation control while the job is active; its displayed 5% is the page's last progress poll, while the API gate had already observed at least 20%. The [rerun screenshot](tkben-t2-07-benchmark-cancellation-20260923.png) shows the populated report.

## Checks

- From `app/client`, `npm run test:unit`: 14 files and 61 tests passed, including the cancellation-request guard and immediate rerun state test.
- Angular lint (`npm run lint`) and production build (`npm run build`): passed.
- Focused backend job and cancellation tests (`test_benchmark_jobs.py`, `test_jobs_manager.py`): 9 passed.
- From the repository root, the opt-in Chrome E2E (`test_benchmark_can_be_cancelled_and_immediately_rerun`) ran with `PYTHONPATH` set to `app`, `E2E_RUN_BENCHMARKS=1`, and `app/server/.venv/Scripts/python.exe -m pytest app/tests/e2e/test_benchmark_cancellation.py -q -c app/tests/pytest.ini --browser-channel=chrome`: 1 passed in 13.73 seconds. The bundled Playwright Chromium executable was unavailable, so the installed Chrome channel was used.
- Ruff check, Ruff format check, and `git diff --check`: passed.

## Cleanup and Boundary

- The test deleted its rerun report and verified it was absent, then deleted the custom tokenizer and dataset and verified both were absent from their list endpoints.
- The successful run used a dedicated SQLite data root. The temporary `TKBEN_DATA_DIR` override in `settings/.env` was restored byte-for-byte with its original timestamp. The isolated data and task-specific launcher log roots were removed.
- Two earlier validation runs had written six task-created managed-job rows to the configured default database because `.env` overrode the shell data-root value. The exact IDs `6d0d8fdc`, `5296d639`, `40cfebbc`, `8c183c2a`, `8d34eeaa`, and `ebd20196` were deleted transactionally; a follow-up query confirmed none remain. Other database rows and files were preserved.
- The official launcher stopped both application process trees; ports 5000 and 8000 were verified clear.
- This closes only the T2-07 local cancellation and rerun slice. The broader Cross Benchmark workflow remains PARTIAL. Hosted CI is still unvalidated, Hugging Face and PostgreSQL gates remain blocked, and the responsive visual matrix remains untested. No public API, schema, or database schema changes were made.
