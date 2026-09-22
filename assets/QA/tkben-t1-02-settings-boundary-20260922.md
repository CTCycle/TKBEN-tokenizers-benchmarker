# TKBEN T1-02 Settings boundary closure

Last updated: 2026-09-22

## Scope

- Revision: `2966daf1732e6604570bab43b0728d8015c947ad`
- Branch: `develop`
- Environment: Windows 11 build `10.0.26200`, Python `3.14.7`, Node.js
  `22.23.1`, npm `10.9.8`
- Launcher: `start_on_windows.ps1 -Launch`
- Browser: installed Chrome channel through pytest-Playwright
- Evidence policy: text-only; no screenshots, videos, browser bundles, or
  secrets are retained in this artifact.

## Result

`T1-02` is **PASS** for the campaign slice.

- The Settings browser campaign passed `2/2`: the new all-field matrix and the
  existing round-trip/conflict/reset/new-operation lifecycle test.
- All `16/16` runtime fields were exercised through the rendered Angular form:
  `5` Data, `5` Tokenizers, `4` Benchmarks, and `2` Runtime.
- The matrix passed `38` invalid cases, `25` valid boundary cases, `3`
  legitimate decimal cases, and all `3` tokenizer relationship states
  (default > maximum, maximum > candidates, and both relationships invalid).
- The combined valid save sent all 16 fields in one `PATCH /api/settings`,
  asserted an exact revision increment, asserted the complete
  `overridden_keys` set, verified both upload limits were converted from MiB
  to bytes, and reloaded `/settings` to verify all 16 controls rehydrated.
- The focused backend contract set passed `56` tests with `16` existing
  deprecation warnings.
- Angular unit tests passed `60/60` across `14` files, Angular lint passed,
  and the production build completed through the official launcher.
- The in-app browser rendered `/settings` after the rebuilt launch with a
  non-empty page, no framework error overlay, and no captured console errors.

## Deterministic field matrix

Numeric bounds are read from the backend Pydantic model metadata by
`app/tests/e2e/test_settings_ui.py`. Upload values are UI MiB values; the
persisted API values are bytes.

| Tab | Accessible label | API key | Semantics | Valid boundaries | Invalid values exercised |
| --- | --- | --- | --- | --- | --- |
| Data | Histogram bins | `datasets.histogram_bins` | integer, 5..100 | 5, 100 | 4, 101, 5.5 |
| Data | Dataset upload limit (MiB) | `datasets.max_upload_bytes` | integer, UI >=1 MiB | 1 | 0, 1.5 |
| Data | Dataset download timeout (seconds) | `datasets.download_timeout_seconds` | decimal, >=1 | 1 | 0.99 |
| Data | Download retry attempts | `datasets.download_retry_attempts` | integer, 1..10 | 1, 10 | 0, 11, 1.5 |
| Data | Download retry backoff (seconds) | `datasets.download_retry_backoff_seconds` | decimal, 0..60 | 0, 60 | -0.01, 60.01 |
| Tokenizers | Default discovery limit | `tokenizers.default_discovery_limit` | integer, 1..250 and <= maximum | 1, 250 | 0, 251, 1.5 |
| Tokenizers | Maximum discovery limit | `tokenizers.max_discovery_limit` | integer, 1..250 and >= default | 1, 250 | 0, 251, 1.5 |
| Tokenizers | Discovery candidate cap | `tokenizers.max_discovery_candidates` | integer, >=1 and >= maximum | 1 | 0, 1.5 |
| Tokenizers | Metadata candidate multiplier | `tokenizers.metadata_candidate_multiplier` | integer, 1..10 | 1, 10 | 0, 11, 1.5 |
| Tokenizers | Tokenizer upload limit (MiB) | `tokenizers.max_upload_bytes` | integer, UI >=1 MiB | 1 | 0, 1.5 |
| Benchmarks | Default document cap | `benchmarks.default_max_documents` | integer, 1..100000 | 1, 100000 | 0, 100001, 1.5 |
| Benchmarks | Default tokenizer batch size | `benchmarks.default_batch_size` | integer, 1..4096 | 1, 4096 | 0, 4097, 1.5 |
| Benchmarks | Default parallelism | `benchmarks.default_parallelism` | integer, 1..128 | 1, 128 | 0, 129, 1.5 |
| Benchmarks | Benchmark streaming batch size | `benchmarks.streaming_batch_size` | integer, >=100 | 100 | 99, 100.5 |
| Runtime | Dataset streaming batch size | `datasets.streaming_batch_size` | integer, >=100 | 100 | 99, 100.5 |
| Runtime | Job polling interval (seconds) | `jobs.polling_interval` | decimal, >=0.25 | 0.25 | 0.24 |

For every invalid case the rendered control set `aria-invalid="true"`, showed
the expected inline error, and left Save disabled; a valid value was restored
before the next case. The integer-only fractional cases for the two streaming
batch fields are included in the matrix as `100.5`.

The all-fields persistence values were:

- Data: histogram `25`, upload `12 MiB` (`12582912` bytes), timeout `12.5`,
  attempts `4`, backoff `2.5`, streaming batch `11000`.
- Tokenizers: default `7`, maximum `11`, candidates `17`, multiplier `4`,
  upload `12 MiB` (`12582912` bytes).
- Benchmarks: documents `1234`, batch `24`, parallelism `2`, streaming batch
  `1100`.
- Runtime: polling `1.25`.

## Commands and outcomes

```text
.\start_on_windows.ps1 -Launch                                      PASS; rebuilt stale Angular production output and started ports 5000/8000
$env:PYTHONPATH='app'; .\app\server\.venv\Scripts\python.exe -m pytest -c app/tests/pytest.ini app/tests/e2e/test_settings_ui.py -q --browser-channel=chrome  PASS (2 passed)
$env:PYTHONPATH='app'; .\app\server\.venv\Scripts\python.exe -m pytest -c app/tests/pytest.ini app/tests/unit/server/configurations/test_runtime_settings.py app/tests/unit/server/api/test_settings_routes.py app/tests/unit/server/api/test_keys_routes.py app/tests/unit/server/repositories/test_database_initialization.py app/tests/unit/server/repositories/test_database_migrations.py app/tests/unit/server/repositories/test_persistence_contract.py -q  PASS (56 passed, 16 warnings)
.\runtimes\nodejs\npm.cmd --prefix .\app\client run test:unit       PASS (14 files, 60 tests)
.\runtimes\nodejs\npm.cmd --prefix .\app\client run lint            PASS
.\app\server\.venv\Scripts\ruff.exe check --no-cache app/tests/e2e/test_settings_ui.py  PASS
.\app\server\.venv\Scripts\ruff.exe format --check app/tests/e2e/test_settings_ui.py  PASS
```

The production bundle build was part of the final launcher run and reported
`Application bundle generation complete`; it produced the rebuilt Settings
chunk before the final browser campaign.

## Fixes made

- Added the backend-derived 16-field rendered boundary matrix, MiB-to-byte
  persistence assertions, all-field reload checks, and final snapshot equality
  checks to `app/tests/e2e/test_settings_ui.py`.
- Updated the Settings component so a dynamic HTML maximum cannot hide the
  authoritative `default <= maximum` field error.
- Updated tokenizer form validation to retain both relationship errors when
  both relationships are invalid, with focused frontend regression coverage.

## Cleanup

- Final `GET /api/settings` showed the original defaults, an empty
  `overridden_keys` set, and revision `12`.
- `app/resources/runtime-settings.json` was absent after restoration. The
  persistent `app/resources/hf-key-material.json` remained in place and was
  not modified by this slice.
- The launcher `-KillAll` command again hit the host command-line inspection
  `Access denied` condition. The exact launcher-owned backend/frontend trees
  and an orphaned preview child were identified and stopped explicitly; ports
  `5000` and `8000` were then confirmed free.

This record closes the T1-02 campaign slice only. It does not promote the
broader `configuration.runtime-settings` component or claim the separate key,
provider, PostgreSQL, responsive, hosted-CI, or release gates.
