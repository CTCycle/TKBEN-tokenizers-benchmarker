# T2-05 Local Tokenizer Report and Vocabulary Validation

Date: 2026-09-23
Result: PASS for the local custom-tokenizer report and vocabulary-paging slice
Validated application revision: `5cf39a0a49e7f51e57438520901bda3551e96600`
Environment: Windows, official `start_on_windows.ps1 -Launch`, isolated SQLite data and log roots under `runtimes/cache/t2-05-validation-data-20260923`, API on port 5000, production preview on port 8000, installed Chrome through Playwright.

## Coverage

The focused report-flow E2E now uploads a unique local WordLevel `CUSTOM_` tokenizer with exactly 1,207 vocabulary entries. It no longer depends on previously downloaded tokenizers or Hugging Face access. The Tokenizers page opened the new report and completed its managed generation job.

The report endpoint returned the same persisted report ID after browser reload. API vocabulary reads returned 500 entries at offset 0, 500 at offset 500, and 207 at offset 1,000, with contiguous token IDs 0 through 1,206. In the rendered dashboard, Next and Previous moved through the first, middle, and short final pages; navigation was disabled at both ends. Reloading and reopening restored the same report and first vocabulary page.

The browser emitted the expected 404 for the latest-report lookup before the new report existed. The test asserts that this is the sole failed HTTP response and that there are no page exceptions or other console errors.

## Checks

- `test_tokenizer_report_flow_supports_paged_vocabulary`: 1 passed in Chrome.
- `test_tokenizer_report_contract.py`, `test_tokenizer_vocabulary_metrics.py`, `test_tokenizers_routes.py`, and `test_tokenizers_service.py`: 34 passed. Three existing AnyIO deprecation warnings remain.
- Ruff check, Ruff format, and `git diff --check`: passed.
- Screenshot: [populated tokenizer report and middle vocabulary page](tkben-t2-05-tokenizer-report-vocabulary-20260923.png).
- Codex in-app Browser automation could not initialize because the Windows sandbox helper failed to apply its read-deny ACL. The focused Playwright Chrome E2E provided the browser interaction and rendered screenshot instead.

## Cleanup and Boundary

The test deleted its unique tokenizer and cascading report/vocabulary rows, then confirmed the latest-report endpoint returned 404. The official launcher `-KillAll` stopped its two owned process roots; ports 5000 and 8000 were no longer listening. The temporary `settings/.env` override was restored byte-for-byte, and the isolated data root, test caches, and temporary launcher logs were removed.

This closes T2-05 and validates the local tokenizer report dashboard and vocabulary paging. Hugging Face discovery, PostgreSQL, hosted CI, PDF export, populated Cross Benchmark dashboards, and the responsive visual matrix remain separate gates.
