# TKBEN partial-gate closure validation

Date: 2026-09-22
Repository: `CTCycle/TKBEN-tokenizers-benchmarker`
Branch and base revision: `develop`, `4b608b7d44a22767e665669cfc420409ffd2f006`
Change state: uncommitted working-tree changes based on the revision above.
Validation roots: disposable Windows and Linux checkouts; no user project data
was used.

## Windows portable bootstrap

Host: Windows 11 Pro build 26200, Windows PowerShell 5.1.26100.9444.

The final clean run began after removing managed Python, Node.js, uv, the backend
`.venv`, frontend `node_modules` and `dist`, all dependency/build stamps, the
SQLite database, and generated `settings/.env`. The supported launcher path
created `.env` from its example, downloaded and verified Python 3.14.7, Node.js
22.23.1, and uv 0.12.17, installed the locked backend dependencies (77 packages)
and frontend dependencies (501 packages), migrated SQLite to
`0005_managed_job_lifecycle`, built Angular, wrote the three expected stamps,
and started the backend and preview. `/api/health` returned 200; Chrome rendered
the Dataset page at the preview URL.

The clean and warm runs used these stamps:

| Stamp | SHA-256 | Warm relaunch result |
| --- | --- | --- |
| `app/server/.venv/.tkben-dependencies.json` | `28CADCC348F61C2A7FCB49099E0381078FB59C05DF932B950F934256F527240B` | Hash and timestamp unchanged |
| `app/client/node_modules/.tkben-dependencies.json` | `3398197C4A95D69BE8220A0DD492DD1C838C4A27516E4E4422FD6BC8891CF122` | Hash and timestamp unchanged |
| `app/client/dist/tkben-angular/.tkben-build.json` | `B0D7DF68E8582FE5B5E62E66768FDC4096B06D9D5AB802E6B1FFD3F484D63B59` | Hash and timestamp unchanged |

The warm launcher reported that environments and frontend output were ready and
skipped setup. I also installed uv 0.12.16 into the disposable managed-runtime
directory; the next launch detected the mismatch, replaced it with pinned uv
0.12.17, synchronized the backend, and reused the frontend build. The official
uv 0.12.17 release and Windows archive are listed by the
[Astral uv release](https://github.com/astral-sh/uv/releases/tag/0.12.17).

## Windows maintenance menu

All 13 menu routes were exercised in the disposable checkout. Standard and
Development installation profiles both completed. Rebuild, database
initialization, and the test-suite menu route passed. Check for Updates
completed. Update on `develop` refused with a clear message and did not switch
branches or change files.

For Remove Logs, Clear Cache, Remove All Data, Uninstall, and Kill All, both
decline and approval paths were exercised. Sentinel checks showed:

- Declines retained the selected log, cache marker, database, source files,
  runtime settings, managed runtimes, dependencies, build output, and `.env`.
- Approved log cleanup removed `.log` files and preserved a non-log file,
  database, and dataset fixture.
- Approved cache cleanup removed the disposable cache marker and cache entries
  while retaining installed runtimes, dependencies, build output, and user data.
- Approved Uninstall removed managed runtimes, the backend environment,
  frontend dependencies, and build output while preserving the database, source
  fixtures, runtime settings, `.env`, templates, and lockfiles.
- Approved Remove All Data removed the database, source fixtures, runtime
  settings, and log contents while preserving `.env`, `.env.example`, source
  code, and dependency manifests.
- Kill All refusal preserved both live service owners and backend health.
  Approval stopped only the two TKBEN process-tree roots; ports 5000 and 8000
  were then free. An initial run exposed a quoted npm command-line match and
  duplicate nested process roots; the launcher filter was corrected and the
  route then completed successfully.

Clear Cache encountered confirmation prompts for test-created directory links
under the disposable runtime cache. The links inspected at that time pointed
into that cache; the route completed and no links remained. No system or
unrelated process was targeted.

## Windows failure-path checks

| Scenario | Result |
| --- | --- |
| Malformed backend dependency stamp | Repaired by locked backend sync; frontend dependencies and build were reused. |
| Missing frontend dependency stamp | Reinstalled with `npm ci`; valid Angular build was reused. |
| Stale frontend build stamp | Rebuilt Angular without reinstalling frontend dependencies. |
| Backend-only stale dependency fingerprint | Repaired backend state; frontend dependency and build stamps stayed unchanged. |
| uv 0.12.16 in managed runtime | Replaced with pinned uv 0.12.17. |
| Invalid `FASTAPI_PORT` | Launch exited nonzero with a validation error; the existing healthy service was untouched and the original `.env` bytes were restored. |
| Redirected launch with occupied ports | Launch exited nonzero, named listener owners, and reported that no process was terminated. |
| Listener reacquired port 5000 after the initial check | A delayed synthetic listener bound during a forced frontend rebuild. The second check aborted launch, reported its PID, left that listener running, and did not start port 8000. The helper was then stopped by its verified process ID. |
| Permission-denied process termination | Not forced. This host did not provide a safe, isolated non-owned process/privilege boundary for that case; no foreign or system process was used. |

## Linux manual startup and restart E2E

Environment: disposable `ubuntu:latest` container, resolved as Ubuntu 26.04;
system Python 3.14.4, Node.js 22.22.3, uv 0.11.30. The repository CI workflow
uses `ubuntu-latest`, Python 3.14.7, Node.js 22.22.3, and uv 0.11.30. This was a
containerized manual run, not a hosted CI result or an exact Python-patch match.

Followed the fresh-checkout flow in `README.md`,
`assets/docs/runtime/startup.md`, and `assets/docs/runtime/deployment.md`:

```bash
cd app/server
uv sync
uv run python -m uvicorn server.app:app --app-dir .. --host 127.0.0.1 --port 5000

cd app/client
npm ci
npm run build
npm run preview -- --host 127.0.0.1 --port 8000 --strictPort
```

Startup applied Alembic migration `0005_managed_job_lifecycle`; health returned
200 and the preview proxy served `/api/datasets/list`. The browser rendered
Dataset, Tokenizers, Cross Benchmark, and Settings. A long dataset analysis was
started for dataset `custom/restart-e2e` (25,000 documents). After stopping and
restarting the backend through the documented manual command:

- Active job `3fa194eb` remained addressable and returned `failed` with
  `Job interrupted because the application restarted while it was running.`
- Completed upload job `c079b263` remained `completed`.
- Dataset `custom/restart-e2e` and all 25,000 documents remained in SQLite.
- The post-restart database held two managed-job rows and one analysis session.

The disposable database had no populated pre-existing benchmark reports,
reports tags, tokenizer artifacts, or saved runtime overrides before this test.
The evidence proves preservation of the completed test upload and its dataset,
not preservation of populated user artifacts that were absent at baseline. The
browser routes were checked again after restart. The backend, preview, Linux
container, Docker Desktop service, and test listeners were stopped afterward;
ports 5000, 8000, 16500, and 18000 were free.

## Code and quality evidence

- `app/tests/unit/server`: 212 passed, 19 warnings.
- Launcher contract tests: 17 passed.
- Ruff: all checks passed (the existing ignored pytest-cache path emitted a
  non-failing access-denied warning).
- BasedPyright: 0 errors, 1,953 warnings.
- Frontend lint passed; frontend unit tests passed: 14 files, 60 tests.
- The Windows menu test-suite run passed 378 tests with 4 skips before the
  final launcher contract additions; its E2E phase was explicitly skipped.
- Windows PowerShell parser reported no syntax errors. `git diff --check`
  passed.
- Hosted CI and macOS were not run.

## T0-02 permission-denied follow-up

Date: 2026-09-22
Repository revision: `4a7f875eb9fcfe3b5eaf69e2cf9f82f672e03bf8`
Change state: the launcher contract additions and privilege harness are
uncommitted working-tree changes based on that revision.

The production launcher was not changed. The new
`app/tests/integration/windows/test_launcher_port_permissions.ps1` harness is
explicitly safety-gated: it refuses redirected input/output, refuses an
elevated launcher token, requires `-AllowDisposableEnvironment`, clones a
disposable checkout, starts only synthetic listeners, and cleans up the
privileged listener through its own elevated helper context. The real
privilege-isolated run was not executed on this managed host because it does
not provide a disposable Windows VM, Windows Sandbox, or equivalent isolated
boundary. No system or foreign process was targeted.

Local regression evidence:

- The exact requested focused command was attempted and stopped during pytest
  collection because this checkout does not add `app` to `PYTHONPATH`
  (`ModuleNotFoundError: No module named 'server'`).
- With the repository runner's `PYTHONPATH=app` setup, the focused launcher
  contract suite passed `19 passed`.
- The isolated backend/unit run passed `349 passed, 19 warnings`.
- `app/tests/run_tests.bat` completed Ruff, BasedPyright (`0 errors`), live
  service readiness, frontend unit tests (`14 files, 60 tests`), and cleanup,
  but its Python phase failed during collection on the existing ACL-protected
  `app/tests/runtimes/cache/pytest` path with `WinError 5: Access denied`.
  This is recorded as an environment residue; permissions were not widened or
  the path deleted. No TKBEN listener remained on ports 5000 or 8000.

Permission-denied acceptance fields remain unobserved: protected PID/port,
real `Stop-Process` denial, listener survival, launcher output, and
backend/frontend non-start evidence. T0-02 and `runtime.windows-launcher`
therefore remain `PARTIAL` until the harness is run in the required isolated
Windows boundary.
