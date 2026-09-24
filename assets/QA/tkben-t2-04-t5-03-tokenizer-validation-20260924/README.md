# TKBEN T2-04, T2-05, and T5-03 Tokenizers Validation

**Date:** 2026-09-24

**Last updated:** 2026-09-24

**Result:** T2-04 PASS; T2-05 revalidation PASS; Tokenizers portion of T5-03 PASS

**Application revision exercised:** `3150c1603fc3a7cc74bfdc91d88068b80dd651c0`

## Environment

- Windows local run through `start_on_windows.ps1 -Launch` under PowerShell 7.
- Frontend `http://127.0.0.1:59539`; API `http://127.0.0.1:59538`.
- SQLite and tokenizer artifacts used an isolated root at `data-root/`.
- Hugging Face discovery was intercepted with a deterministic empty response; this run made no provider-backed claim.

## Results

- **T2-04, upload and persistence:** the UI uploaded a valid custom tokenizer twice under the same filename. Both upload requests completed before assertions ran; the canonical catalog contained one entry and the second seven-token vocabulary replaced the first five-token vocabulary in the stored artifact.
- **T2-04, restart and deletion:** the official launcher was stopped and restarted with the same isolated root. The catalog and seven-token artifact survived restart. The UI generated a report, the completed report and vocabulary preview showed the replacement vocabulary, and UI deletion removed the catalog entry, latest report, and artifact directory.
- **T2-05, report and vocabulary:** the 1,207-entry report flow passed again. The persisted report reloaded, and the API returned contiguous 500/500/207 vocabulary pages while the UI navigated the first, middle, and final pages.
- **T5-03, Tokenizers route:** empty, loading, controlled error, populated long-identifier, and report states were checked. Empty/loading/error/populated route states and the report view were measured at 1920×1080, 1440×900, 1024×768, and 390×844 with no horizontal overflow. The Tokenizer Manager remained within the viewport at each size. ArrowRight/End tab navigation, Escape dismissal, and focus return passed at each size.
- **Cleanup:** the post-restart UI deletion returned the catalog to empty and removed the persisted report and canonical artifact. The temporary application data root and launcher processes are removed as part of task cleanup.

## Verification Evidence

- `tokenizer-e2e-final.txt`: focused tokenizer API/UI run, **7 passed, 3 skipped**. Skips were the two external Hugging Face discovery cases and the separately executed restart case.
- `tokenizer-restart-e2e-final.txt`: post-restart report and UI deletion, **1 passed, 9 deselected**.
- `screenshots/`: rendered report and long-identifier views at four viewport sizes, plus empty, loading, error, and Tokenizer Manager states. The images were reviewed after capture.
- Ruff check passed for `app/tests/e2e/test_tokenizers_api.py`.

## Remaining Boundaries

- Hugging Face discovery/download and gated-provider behavior remain **BLOCKED** pending an approved provider credential/network run.
- PostgreSQL runtime equivalence remains **BLOCKED** pending a disposable PostgreSQL target and credentials.
- The aggregate responsive matrix remains **PARTIAL**: Dataset and Settings routes have not received their responsive state/keyboard matrix. T5-02 remains **UNTESTED**.
- T5-05 is **OUT_OF_SCOPE** for the supported Windows x64 release target; the Ubuntu manual run is diagnostic evidence only, and no macOS or hosted Ubuntu runtime validation is planned. T5-06 remains **PARTIAL** because release publication was not checked.
- The report export matrix remains **WORKING** pending manual review of other visualization overrides and vocabulary exports larger than 23 entries.
