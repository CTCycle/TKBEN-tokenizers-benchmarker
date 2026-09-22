# TKBEN T1-03 key-management validation

Last updated: 2026-09-22

## Scope

- Revision: `09805b7062325b6180ef956419c1d5db1b4d2e89`
- Branch: `develop`
- Environment: Windows, Python 3.14.7, Node.js 22.23.1, Chrome channel
- Application path: `start_on_windows.ps1 -Launch`
- Validation ports: `5001` (backend) and `8001` (frontend), selected because
  the pre-existing port-5000 backend was already listening
- Validation data: disposable isolated `TKBEN_DATA_DIR`, removed after the run
- Reveal policy: `ALLOW_KEY_REVEAL=false`
- Secret handling: `TKBEN_TEST_HF_KEY` was unavailable, so no supplied
  credential was entered. The live-secret-specific test was skipped; synthetic
  keys were generated in the test process and were never written to evidence.

## Results

| Check | Result | Evidence |
| --- | --- | --- |
| Clean API/database baseline | PASS | Isolated `/api/keys` returned zero rows and SQLite `hf_access_keys` returned zero rows before the browser run. The pre-existing backend on port 5000 also returned zero keys. |
| Official launcher startup | PASS | Temporary `.env` override launched the current checkout successfully on ports 5001/8001; the wrapper restored `settings/.env` byte-for-byte. |
| Rendered Settings > Keys lifecycle | PASS | `test_hf_keys_ui.py`: 1 passed. Empty state, add, masked row, duplicate `409`, activation/deactivation/reactivation, active-delete `400`, inactive deletion, single-active switching, and final row removal passed in the rendered UI. |
| Reveal policy | PASS | The rendered reveal action returned `403`; the masked preview and API list remained unchanged. |
| Direct SQLite ciphertext | PASS | The test queried only newly created IDs and asserted a stored value existed, differed from the generated plaintext, and did not contain it. The encryption-material file existed separately from SQLite. Values were not printed. |
| Public API masking | PASS | Create/list responses and every post-action list response contained no `key_value` field and no generated plaintext. |
| Cleanup | PASS | Validation-created rows were removed; the isolated API and SQLite key counts returned to zero; the Keys tab rendered `No keys stored.`; validation ports had no listeners. |
| Supplied credential path | SKIPPED | `TKBEN_TEST_HF_KEY` was not present in the process environment. No live credential was substituted or logged. |

## Regression gates

```text
app/server/.venv/Scripts/python.exe -m pytest -c app/tests/pytest.ini -o "cache_dir=<task cache>" app/tests/e2e/test_hf_keys_ui.py -q --browser-channel=chrome  PASS (1 passed, 1 skipped)
app/server/.venv/Scripts/python.exe -m pytest -c app/tests/pytest.ini -o "cache_dir=<task cache>" app/tests/e2e/test_settings_ui.py -q --browser-channel=chrome  PASS (2 passed)
app/server/.venv/Scripts/python.exe -m pytest -c app/tests/pytest.ini -o "cache_dir=<task cache>" app/tests/unit/server/api/test_keys_routes.py app/tests/unit/test_hf_access_keys_service.py -q  PASS (10 passed)
runtimes/nodejs/npm.cmd --prefix app/client run test:unit  PASS (14 files, 60 tests)
runtimes/nodejs/npm.cmd --prefix app/client run lint  PASS
app/server/.venv/Scripts/python.exe -m ruff check --config app/server/pyproject.toml app/server app/tests/e2e/test_hf_keys_ui.py  PASS
app/server/.venv/Scripts/python.exe -m basedpyright -p app/server/pyrightconfig.json  PASS (0 errors, 1,953 warnings)
git diff --check  PASS
```

No product-code defect was reproduced. The only source addition is the focused
browser test at `app/tests/e2e/test_hf_keys_ui.py`. The dynamic synthetic-key
prefix did not occur in application logs or the non-sensitive evidence files.
