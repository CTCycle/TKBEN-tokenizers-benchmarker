# T1-05 Catalog Filtering and Races

Last updated: 2026-09-22

## Validation identity

- Repository: `CTCycle/TKBEN-tokenizers-benchmarker`
- Branch: `develop`
- Validated implementation revision: `993e52e8f0617d4bf98a60d62b4c5d5588541218`
- Validation date: 2026-09-22 (Europe/Rome)
- Scope: local Dataset and Tokenizer catalogue filtering, stale-response
  ownership, and deterministic tokenizer discovery sequencing.
- Provider/network scope: no Hugging Face network access or credential was
  required.

## Environment

- Windows host, PowerShell.
- Python `3.14.7` from `app/server/.venv`.
- Node `v22.23.1`, npm `10.9.8`, Git `2.55.0.windows.3`.
- Installed Chrome channel `153.0.8010.53`.
- Backend `http://127.0.0.1:5000`, frontend production preview
  `http://127.0.0.1:8000`; backend docs and frontend root returned HTTP 200.
- Embedded SQLite at Alembic head `0005_managed_job_lifecycle`.

The E2E fixture helpers captured the pre-test local catalogue, inserted four
ready Dataset rows with document counts `2`, `5`, `5`, and `8`, and inserted
four local Tokenizer rows with sources `huggingface`, `huggingface`, `custom`,
and `custom` and vocabulary sizes `2`, `5`, `5`, and `8`. Tokenizer artifacts
were local copies of the existing small JSON fixture; no provider download was
performed. The expected rows include any pre-existing baseline catalogue rows,
so the test does not delete user data.

## Exact validation commands and results

All commands were run from the repository root. The focused Playwright runs
used the installed Chrome channel. Python commands used
`$env:PYTHONPATH=(Join-Path (Get-Location) 'app')`; frontend commands used
`$env:NPM_CONFIG_CACHE=(Join-Path (Get-Location) 'runtimes\cache\npm')` and
`$env:NG_CLI_ANALYTICS='false'`.

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'app')
$env:APP_TEST_FRONTEND_URL='http://127.0.0.1:8000'
$env:APP_TEST_BACKEND_URL='http://127.0.0.1:5000'
$env:UI_BASE_URL=$env:APP_TEST_FRONTEND_URL
$env:API_BASE_URL=$env:APP_TEST_BACKEND_URL
$env:PLAYWRIGHT_BROWSERS_PATH=(Join-Path (Get-Location) 'runtimes\cache\playwright')
app\server\.venv\Scripts\python.exe -m pytest -c app/tests/pytest.ini app/tests/e2e/test_app_flow.py -k 'catalog_filter_matrix or catalog_race or discovery_race' --browser-channel=chrome -q --tb=short
```

Result: **5 passed, 12 deselected**. This includes both the existing Dataset
stale-response scenario and the new Dataset/Tokenizer populated matrices,
Tokenizer catalogue race, and Tokenizer discovery race.

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'app')
app\server\.venv\Scripts\python.exe -m pytest -c app/tests/pytest.ini app/tests/e2e/test_app_flow.py --browser-channel=chrome -q --tb=short -o cache_dir=runtimes/cache/pytest-state/t1-05-app-flow
```

Result: **17 passed**.

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'app')
app\server\.venv\Scripts\python.exe -m pytest -c app/tests/pytest.ini app/tests/unit/server/api/test_datasets_routes.py app/tests/unit/server/api/test_tokenizers_routes.py app/tests/unit/server/repositories/test_persistence_contract.py app/tests/unit/server/services/test_tokenizers_service.py -k 'dataset_list or tokenizer_list or catalog_filters or dataset_catalog' -q --tb=short -o cache_dir=runtimes/cache/pytest-state/t1-05-backend
```

Result: **4 passed, 30 deselected**.

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'app')
app\server\.venv\Scripts\python.exe -m pytest -c app/tests/pytest.ini app/tests/unit/server/api/test_datasets_routes.py app/tests/unit/server/api/test_tokenizers_routes.py -q --tb=short -o cache_dir=runtimes/cache/pytest-state/t1-05-backend-routes
```

Result: **15 passed**. Three existing AnyIO deprecation warnings were emitted;
none was a test failure.

```powershell
$env:NPM_CONFIG_CACHE=(Join-Path (Get-Location) 'runtimes\cache\npm')
$env:NG_CLI_ANALYTICS='false'
runtimes\nodejs\npm.cmd --prefix app/client run test:unit -- --watch=false
runtimes\nodejs\npm.cmd --prefix app/client run lint
runtimes\nodejs\npm.cmd --prefix app/client run build
```

Results: **14 frontend test files / 60 tests passed**, lint passed, and the
production build completed successfully.

```powershell
app\server\.venv\Scripts\python.exe -m ruff check app/tests/e2e/test_app_flow.py
```

Result: **All checks passed**.

The in-app browser smoke check rendered `/dataset` and `/tokenizers` with the
expected empty states before fixture seeding. The local backend `/docs` and
frontend `/` readiness checks both returned HTTP 200.

## Filter matrix

Each case asserted the request query parameters and the rendered catalogue
rows after the response settled.

| Catalogue | Case | Contract asserted | Result |
| --- | --- | --- | --- |
| Dataset | Unfiltered | `/api/datasets/list` with no optional query parameters | PASS |
| Dataset | Trimmed search | `search=beta-<fixture>` | PASS |
| Dataset | Public source | `source=public` | PASS |
| Dataset | Custom source | `source=custom` | PASS |
| Dataset | At-least boundary | `document_count_operator=at_least&document_count=5`, including count 5 | PASS |
| Dataset | At-most boundary | `document_count_operator=at_most&document_count=5`, including count 5 | PASS |
| Dataset | Combined filters | Trimmed search + public + at-least + count 5 | PASS |
| Dataset | No match | Combined filters with an absent search term | PASS |
| Dataset | Reset | Clearing search, source, and count returned the complete catalogue | PASS |
| Tokenizer | Unfiltered | `/api/tokenizers/list` with no optional query parameters | PASS |
| Tokenizer | Trimmed search | `search=beta-<fixture>` | PASS |
| Tokenizer | Hugging Face source | UI `hugging_face` serialized as backend `source=huggingface` | PASS |
| Tokenizer | Custom source | `source=custom` | PASS |
| Tokenizer | At-least boundary | `vocabulary_size_operator=at_least&vocabulary_size=5`, including size 5 | PASS |
| Tokenizer | At-most boundary | `vocabulary_size_operator=at_most&vocabulary_size=5`, including size 5 | PASS |
| Tokenizer | Combined filters | Trimmed search + Hugging Face + at-least + size 5 | PASS |
| Tokenizer | No match | Combined filters with an absent search term | PASS |
| Tokenizer | Reset | Clearing search, source, and vocabulary returned the complete catalogue | PASS |

## Race scenarios

- Existing Dataset browser race: request A and request B were issued through
  the rendered Dataset search control; the newer response retained ownership
  of rows and loading state. **PASS, 1/1**.
- Tokenizer catalogue, B completes first: A was delayed 2,000 ms and B 200 ms;
  B's rows became final and A could not overwrite them. **PASS**.
- Tokenizer catalogue, A completes first after B was issued: A was delayed
  600 ms and B 2,000 ms; loading remained owned by B, the stale A rows were
  not rendered, and B became final. **PASS**.
- Tokenizer discovery, stale error: delayed `old-error` followed by a faster
  `new-success` left the success result visible and no stale error banner.
  **PASS**.
- Tokenizer discovery, stale result: delayed `old-success` followed by a
  faster `new-error` left the current error state visible and ignored the old
  result. **PASS**.

## Cleanup and defects

- The new Dataset and Tokenizer fixture rows and local tokenizer artifacts were
  removed in `finally` cleanup blocks.
- The full app-flow regression leaves three unrelated row-lifecycle test
  datasets by design; the exact generated names were deleted afterward through
  the Dataset delete API.
- Final checks returned `{"datasets":[],"count":0}` and
  `{"tokenizers":[],"count":0}` for the local catalogues.
- No T1-05 production defect was reproduced, so no application source fix was
  needed. The existing `switchMap` catalogue ownership and
  `discoverySequence` protections were validated by live browser evidence.
- No generated fixture, cache, screenshot, or secret was added to the commit.

## Closure decision

All supported local Dataset and Tokenizer filters, exact numeric boundaries,
combined/no-match/reset states, catalogue races, and deterministic discovery
races passed with regression gates green. T1-05 is promoted from **PARTIAL**
to **PASS** for validated implementation revision
`993e52e8f0617d4bf98a60d62b4c5d5588541218`.
