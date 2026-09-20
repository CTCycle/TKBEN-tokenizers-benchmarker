# Testing and Quality
Last updated: 2026-09-18

## Tooling and Tests
- Lint and format with Ruff, or the project-standard equivalent if it changes in the future.
- Type check expectations are Pylance-compatible typing.
- Run `app/server/.venv/Scripts/python.exe -m basedpyright -p app/server/pyrightconfig.json` from the repository root; the configured gate requires zero errors and does not fail on warnings.
- Tests use pytest.
- Minimum test impact coverage:
  - `tests/unit`
  - relevant `tests/e2e` when behavior crosses API and UI boundaries
- Browser/live validation uses the in-app browser for a quick visual and interaction smoke check, and uses Playwright or pytest-playwright for repeatable route/API coverage.

The repository CI gate currently runs backend source-only compileall, Ruff,
BasedPyright, unit tests, an OpenAPI smoke import, and the AST architecture
boundary test, plus frontend `npm ci`, lint, unit tests,
and production build.
The backend job creates `app/server/.venv` before installing the test extra,
initializes the embedded SQLite schema, and then runs the checks from that
environment so CI matches the repository Pyright and runtime configuration.
The Windows `app/tests/run_tests.bat` runner additionally supports live
backend/frontend startup, the configured pytest target, and optional frontend
test scripts.

Provider and external-database checks are conditional: run them only when a
usable Hugging Face key or PostgreSQL target is configured, and record an
unavailable gate explicitly. Live validation should correlate browser state,
API responses, application logs, and persisted records; a successful HTTP
status or build alone is not release evidence.

## Development Cache and Artifact Locations
- `runtimes/cache` is the single disposable cache root. Pytest’s collection
  cache and deterministic basetemp are `runtimes/cache/pytest-state` and
  `runtimes/cache/pytest-basetemp-current`, including when pytest is invoked
  directly. The pytest hook also routes bytecode, coverage, Matplotlib, and
  Playwright state below this root.
- Ruff, mypy, uv, pip, npm, Angular, and other development-tool caches resolve
  to named children of `runtimes/cache`; the launcher, batch runner, CI, and
  tool configuration use the same paths.
- Downloaded datasets and Hugging Face tokenizer artifacts are persistent
  application data under `<TKBEN_DATA_DIR>/sources/datasets` and
  `<TKBEN_DATA_DIR>/sources/tokenizers` (defaulting to `app/resources`). They
  are not disposable caches and are never removed by cache cleanup.
- Generic ignore rules for accidental legacy cache names are defensive only;
  no tool is configured to use an alternative cache root.
- `app/client/dist`, `app/server/.venv`, and `app/client/node_modules` remain in
  their established locations because they are the runtime build output or
  installed dependency trees rather than tool caches.
- The complete `runtimes/cache` hierarchy can be deleted and recreated without
  affecting the persistent database, datasets, tokenizer artifacts, logs, or
  templates.

## Cross-language Quality Gates
- Keep architecture layering intact: API -> contracts/services -> repository.
- Treat `test_architecture_boundaries.py` as the executable ownership and
  removed-symbol contract; schema changes must be represented by a tracked
  Alembic revision rather than runtime adoption logic.
- Do not bypass contract validation models.
- Do not duplicate business logic across backend and frontend without necessity.
- Add or adjust tests when changing behavior, contracts, or data schemas.
