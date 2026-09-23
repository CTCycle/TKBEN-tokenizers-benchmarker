# T2-06 Populated Cross Benchmark Wizard Validation

Date: 2026-09-23
Result: PASS for the local wizard-to-report generation, rendering, and reload slice
Validated E2E implementation commit: `f736f9386a119df0bec330cd7a6e7212fd6a61e7`
Environment: Windows, official `start_on_windows.ps1 -Launch`, isolated SQLite data and log roots under `runtimes/cache/t2-06-validation-data-20260923` and `runtimes/cache/t2-06-validation-logs-20260923`, API on port 5000, production preview on port 8000, installed Chrome.

## Coverage

- The test uploaded a unique two-document CSV and local WordLevel tokenizer through the API. It used no Hugging Face provider or credential.
- In the rendered Cross Benchmark page, the test selected `eff.encode_tokens_per_second_mean`, chose the local dataset and tokenizer, set a two-document limit with zero warmup trials and one timed trial, and started the benchmark through the wizard.
- The managed job completed successfully. The API returned a persisted report with the selected dataset and metric; the browser rendered its populated summary and charts. Reload fetched the same report ID and rendered the report again.
- The browser reported no page exceptions, console errors, or failed HTTP responses. The screenshot shows the report after reload: [populated Cross Benchmark report](tkben-t2-06-cross-benchmark-wizard-20260923.png).

## Checks

- `test_cross_benchmark_wizard_runs_and_reloads_local_report`: passed.
- Focused Cross Benchmark dashboard and benchmark API E2E set: 9 passed in 11.61 seconds, including the new wizard flow, the existing dashboard interaction cases, and the benchmark API report round trip.
- Ruff check, Ruff format check, and `git diff --check`: passed.
- Launcher readiness checks: backend health returned 200 and the production preview returned 200.

## Cleanup and Boundary

- The test deleted its report and verified the report endpoint returned 404. It deleted its tokenizer and dataset and verified both were absent from their list endpoints.
- Test inputs, API fixtures, SQLite state, and launcher logs were under the isolated data/log roots. The original `settings/.env` bytes were preserved and restored after the launcher processes were stopped.
- This closes T2-06's local wizard-to-report flow only. The broader Cross Benchmark workflow remains PARTIAL: live report-manager, baseline, clone, inline-tag, customization, data-table, and responsive interactions remain separate validation work. At the captured desktop viewport, the single-tokenizer tick label sits close to the `Tokenizer` axis title; this evidence does not close the responsive visual matrix.
