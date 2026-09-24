# TKBEN T2-01, T4-02, and T5-02 Dataset/Settings Validation

**Date:** 2026-09-24

**Result:** T2-01 PASS; T4-02 PASS for the public ordinary-disk path; T5-02 PASS. The Dataset/Settings portion of the responsive visual matrix is complete.

**Application revision exercised:** `develop` at `668ef675816e60e5d69267fb96899a96aef678e5` with the dataset import changes in this validation run's working tree. The implementation and this record are committed together.

## Environment

- Windows local run through `start_on_windows.ps1 -Launch` under PowerShell 7, using an isolated data/cache root under this QA folder.
- Frontend `http://127.0.0.1:59639`; API `http://127.0.0.1:59638`.
- Embedded SQLite database was 16,625,664 bytes. At the disk check, G: had 430.74 GB free.
- Public Hugging Face dataset access was anonymous. The task environment disabled implicit Hub token use; no application key was configured for this validation.

## Results

### T2-01: CSV/XLSX import and analysis

- Focused route tests passed **5/5**; dataset API E2E passed **7/7**. The E2E cases exercised CSV and generated XLSX upload, managed upload completion, persisted row counts and histograms, analysis behavior, and unsupported upload rejection.
- A three-row XLSX fixture was generated with the declared runtime dependency `openpyxl==3.1.5`. The server and UI now accept CSV and XLSX consistently; `.xls` is rejected with an explicit supported-format message.
- The rendered Dataset manager accepted a synthetic CSV file through its file picker and showed the resulting `custom/dataset-csv` row with three documents. A previously imported three-row XLSX dataset remained visible in the catalogue.
- The Dataset manager dialog described CSV/XLSX consistently. Its file input advertised `.csv,.xlsx`.

### T4-02: public dataset download and disk persistence

- The official launcher downloaded the public `wikitext/wikitext-2-v1` dataset without an application key. After a full launcher restart using the same isolated data root, the API and Dataset page still showed **29,119 documents**.
- The temporary `sources/` directory contained no files after import. The isolated SQLite database retained the dataset after restart.
- This validates one public Hugging Face source and an ordinary free-space import/cleanup path only. Low-disk exhaustion, interrupted-download recovery, and other public dataset source formats were not exercised.

### T5-02: Dataset and Settings responsive/keyboard matrix

Rendered populated Dataset, no-match Dataset, Settings > Data, and empty Settings > Keys states were reviewed at 1920x1080, 1440x900, 1024x768, and 390x844. `document.documentElement.scrollWidth` was at or below the viewport width for every state:

| Rendered state | Document width at 1920 / 1440 / 1024 / 390 viewport widths |
| --- | --- |
| Dataset populated and no-match | 1910 / 1430 / 1015 / 380 px |
| Settings > Data | 1920 / 1440 / 1024 / 380 px |
| Settings > Keys (empty) | 1920 / 1440 / 1024 / 390 px |

- Dataset manager `ArrowRight` selected Predefined → Add by name → Custom dataset. `Escape` closed the dialog and returned focus to the Add dataset button.
- Settings `ArrowRight` moved from Data to Tokenizers; `End` moved to Keys and displayed the empty-state message, “No keys stored.”
- At 1024 px the Settings Data body measured approximately 1025 CSS px because of fractional layout rounding; the document root remained 1024 px and no document-level horizontal overflow appeared.
- This matrix covers page states and route-tab keyboard behavior. It does not repeat the full key-management lifecycle or credential-backed actions.

## Verification Evidence

- `logs/dataset-routes-unit.log`: **5 passed**, one pytest warning.
- `logs/dataset-api-e2e.log`: **7 passed**, one pytest warning.
- Ruff passed for the changed Python sources and tests; `uv lock --check --project app/server`, Angular `npm run lint`, Angular production build, and `git diff --check` passed.
- Both pytest invocations emitted `PytestConfigWarning: Unknown config option: cache_dir`; this warning did not fail either suite.
- Browser measurements and keyboard behavior above were observed in the Codex in-app Browser against the official launcher. No screenshot files were retained.

## Remaining Boundaries

- Large/near-limit upload behavior and low-disk failure/recovery remain **PARTIAL** and are retained as a separate validation-debt item.
- T4-01/T4-03 Hugging Face discovery and gated/private access remain **BLOCKED** pending approved provider credentials/network. This run used only one public dataset and does not close those gates.
- T4-04 PostgreSQL remains **BLOCKED** pending a disposable target and credentials.
- Full Settings key management and supplied-credential behavior remain outside T5-02; see the separate T1-03 record and its credential limitation.
- T3-05 remains WORKING pending inspection of other PDF visualization overrides and vocabulary exports beyond 23 entries.
- T5-06 remains PARTIAL because hosted CI success does not establish release publication evidence.
