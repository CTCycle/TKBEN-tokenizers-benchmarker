# TKBEN T1-02 Runtime Settings validation follow-up

Last updated: 2026-09-22

## Scope

- Validated source revision: `41f265af004fd9dd99eaa0445a3c04000d6b106c`
- Branch: `develop`
- Environment: Windows, repository-managed Python environment, installed Chrome
  channel, local FastAPI and Angular production preview
- This record supplements
  [the earlier T1-02 boundary closure](tkben-t1-02-settings-boundary-20260922.md).
  That historical evidence is retained unchanged.
- The source commit adds tests only; no production implementation or database
  migration changed.

## Runtime setting matrix

The editable key set is `RuntimeSettingsStore._EDITABLE_KEYS`. Types, defaults,
and base bounds come from `ServerSettings` and the strict runtime patch schema;
the backend test derives the public field list and verifies all 16 keys and
numeric bounds against those current models. Angular control bindings are in
`settings-page.component.ts` and accessible labels are in
`settings-page.component.html`.

| Persisted key | Default; type and allowed domain | Persistence and backend validation | Angular control | Runtime consumer |
| --- | --- | --- | --- | --- |
| `datasets.histogram_bins` | `20`; integer, 5–100 | Sparse integer at `overrides.datasets.histogram_bins`; strict integer and bounds checked before write | `histogramBins` — Histogram bins | New dataset and tokenizer report histograms (`DatasetService`, `TokenizerReportingService`) |
| `datasets.max_upload_bytes` | `25 × 1024²` bytes; integer, ≥1 byte | Sparse byte count; strict integer and positive-byte lower bound | `datasetMaxUploadMiB` — Dataset upload limit; UI converts MiB to bytes | Dataset upload stream limit (`api/datasets.py`, `read_upload_limited`) |
| `datasets.download_timeout_seconds` | `180.0`; finite float, ≥1 | Sparse finite number; lower bound checked before write | `downloadTimeoutSeconds` — Dataset download timeout | Timeout for newly started dataset download workers (`DatasetService`) |
| `datasets.download_retry_attempts` | `3`; integer, 1–10 | Sparse strict integer with inclusive bounds | `downloadRetryAttempts` — Download retry attempts | Retry count for subsequent dataset downloads (`DatasetService`, `dataset_operations.py`) |
| `datasets.download_retry_backoff_seconds` | `2.0`; finite float, 0–60 | Sparse finite number with inclusive bounds | `downloadRetryBackoffSeconds` — Download retry backoff | Delay between subsequent dataset download attempts (`DatasetService`) |
| `tokenizers.default_discovery_limit` | `50`; integer, 1–250 and ≤ maximum | Sparse strict integer; tokenizer model also checks `default ≤ maximum` | `defaultDiscoveryLimit` — Default discovery limit | Default for new discovery requests that omit a limit (`_build_tokenizer_discovery_query`) |
| `tokenizers.max_discovery_limit` | `250`; integer, 1–250 and ≥ default | Sparse strict integer; tokenizer model checks it is ≤ candidate cap | `maxDiscoveryLimit` — Maximum discovery limit | Rejects requests above the configured discovery limit (`api/tokenizers.py`) |
| `tokenizers.max_discovery_candidates` | `750`; integer, ≥1 and ≥ maximum | Sparse strict integer; tokenizer model checks it is ≥ maximum | `maxDiscoveryCandidates` — Discovery candidate cap | Caps candidates returned by tokenizer discovery (`TokenizersService`) |
| `tokenizers.metadata_candidate_multiplier` | `3`; integer, 1–10 | Sparse strict integer with inclusive bounds | `metadataCandidateMultiplier` — Metadata candidate multiplier | Multiplies discovery limit for metadata overfetch (`TokenizersService`) |
| `tokenizers.max_upload_bytes` | `10 × 1024²` bytes; integer, ≥1 byte | Sparse byte count; strict integer and positive-byte lower bound | `tokenizerMaxUploadMiB` — Tokenizer upload limit; UI converts MiB to bytes | Tokenizer upload stream limit (`api/tokenizers.py`, `read_upload_limited`) |
| `benchmarks.default_max_documents` | `1,000`; integer, 1–100,000 | Sparse strict integer with inclusive bounds | `benchmarkDefaultMaxDocuments` — Default document cap | Prefills the document count for newly opened cross-benchmark runs |
| `benchmarks.default_batch_size` | `16`; integer, 1–4,096 | Sparse strict integer with inclusive bounds | `benchmarkDefaultBatchSize` — Default tokenizer batch size | Prefills the batch size for newly opened cross-benchmark runs |
| `benchmarks.default_parallelism` | `1`; integer, 1–128 | Sparse strict integer with inclusive bounds | `benchmarkDefaultParallelism` — Default parallelism | Prefills parallelism for newly opened cross-benchmark runs |
| `benchmarks.streaming_batch_size` | `1,000`; integer, ≥100 | Sparse strict integer with lower bound | `benchmarkStreamingBatchSize` — Benchmark streaming batch size | Database row batch size for newly constructed `BenchmarkService` instances |
| `datasets.streaming_batch_size` | `10,000`; integer, ≥100 | Sparse strict integer with lower bound | `datasetStreamingBatchSize` — Dataset streaming batch size | Batch size for newly constructed `DatasetService` processing streams |
| `jobs.polling_interval` | `1.0`; finite float, ≥0.25 | Sparse finite number with lower bound | `jobPollingInterval` — Job polling interval | Poll interval returned for new jobs and used by the frontend job poller |

The persisted document is `<TKBEN_DATA_DIR>/runtime-settings.json` (by default
`app/resources/runtime-settings.json`) with `schema_version`, `revision`, and
sparse nested `overrides`. Omitted fields/groups resolve from the typed
defaults. Upload limits are byte-valued integers in persistence and the API;
only their Settings controls use MiB. The strict patch schema rejects nulls,
malformed types, non-finite numbers, unknown fields, and invalid bounds. Tokenizer
relationship rules require `default_discovery_limit ≤ max_discovery_limit ≤
max_discovery_candidates`. Candidate state is validated before persistence and
before the in-memory snapshot changes; resetting the final override removes the
runtime settings file.

## Acceptance evidence

| Criterion | Result and evidence |
| --- | --- |
| All 16 settings mapped to current implementation | PASS. `test_public_runtime_matrix_matches_typed_backend_schema` derives request fields, cross-checks `ServerSettings` constraints, compares the keys with `RuntimeSettingsStore._EDITABLE_KEYS`, and asserts 16 fields. The rendered E2E matrix checks all 16 Settings controls and their backend-derived boundaries. |
| Validation boundaries and atomic rejection | PASS. API cases cover numeric minimum/maximum, malformed types, empty/null values, out-of-range values, and fractional integers. Each rejected mixed update leaves both the effective snapshot and persisted bytes unchanged. Tokenizer cross-field failures and non-finite stored floats are also covered. |
| Sparse persistence, reload, and backend restart | PASS. The store restart test writes only a dataset histogram override, recreates the store, and verifies every omitted setting returns to its typed default. In the browser, five saved overrides survived Settings reload and a backend restart. |
| Revision conflict and reset semantics | PASS. Stale updates return HTTP 409 without overwriting newer settings. The browser reset of `datasets.histogram_bins` restored 20 while retaining the unrelated `benchmarks.streaming_batch_size=1001` override. Reset-all removed all overrides; after another backend restart, the API and Settings page returned defaults and `runtime-settings.json` was absent. |
| New work uses settings without rewriting existing artifacts | PASS. Backend service/API tests observe updated dataset histogram, upload limits, download timeout/retry/batching, tokenizer discovery cap/multiplier, benchmark stream batching, and job poll interval. The Settings E2E uploads a uniquely named synthetic dataset and tokenizer, observes histogram/job settings, tokenizer discovery limit, and benchmark wizard document/batch/parallelism prefills, then removes those dataset/tokenizer inputs. The completed upload job remains in the normal terminal-job retention lifecycle; existing artifacts were not rewritten. |
| Regression gates | PASS with one runner collection limitation. The full backend unit folder passed 504/504 using a fresh cache directory inside `runtimes/cache`; Ruff passed; BasedPyright reported 0 errors and 1,953 warnings; Angular unit tests passed 60/60 across 14 files; lint and production build passed. The repository runner's Python phase separately failed while traversing an inaccessible generated `app/tests/cache/pytest` directory; the detailed runner result is below. |

## Commands and results

```text
app/tests/run_tests.bat
  Live server: PASS; Ruff: PASS; BasedPyright: PASS (0 errors, 1,953 warnings)
  Python phase: collection failed with WinError 5 Accesso negato at
    app/tests/cache/pytest
  Frontend bootstrap: PASS; Angular unit tests: PASS (14 files, 60 tests)
  Frontend E2E: SKIPPED by the repository runner

set PYTHONPATH=app & app/server/.venv/Scripts/python.exe -m pytest -c app/tests/pytest.ini app/tests/unit --confcutdir=app/tests/unit --basetemp=runtimes/cache/pytest-t1-02-20260922 -p no:cacheprovider -q
  PASS: 504 passed, 24 warnings

set PYTHONPATH=app & app/server/.venv/Scripts/python.exe -m pytest -c app/tests/pytest.ini app/tests/e2e/test_settings_ui.py --browser-channel=chrome --tb=short -q
  PASS: 2 passed; one warning reports a pre-existing pytest cache-path collision

ruff check and ruff format --check on the three changed test files
  PASS
npm --prefix app/client run lint
  PASS
npm --prefix app/client run build
  PASS (production bundle generated)
```

The isolated backend command skips the root E2E `conftest.py` so pytest does
not reuse the ACL-protected `pytest-basetemp-current`; its fresh base remains
inside the canonical `runtimes/cache` root, so the cache-layout unit test also
passes. The temporary directory was removed after the run. The Settings E2E
used the installed Chrome channel because the repository-local Playwright
Chromium binary is unavailable.

## In-app browser lifecycle

The Codex in-app browser rendered `http://127.0.0.1:8000/settings`. Starting
from defaults, the browser changed histogram bins to `21`, benchmark default
batch size to `17`, parallelism to `2`, benchmark streaming batch size to
`1001`, and job polling interval to `2`; Save Changes completed successfully.
Reloading Settings and restarting the backend preserved all five values. The
API response showed the exact five overridden keys and the documented default
snapshot alongside them.

The browser then reset histogram bins individually and confirmed the unrelated
benchmark streaming override remained. Reset-all cleared the remaining
override. After a second backend restart and Settings reload, the browser
showed documented defaults and the API returned an empty `overridden_keys`
list. The final screenshot was visually reviewed in the browser session; this
text-only QA record does not retain a screenshot. Final API checks showed
defaults active and no `runtime-settings.json` file.

This closes T1-02 at the stated source revision. It does not promote the broader
`configuration.runtime-settings` component or the separate provider,
PostgreSQL, populated report, hosted CI, responsive, or release gates.
