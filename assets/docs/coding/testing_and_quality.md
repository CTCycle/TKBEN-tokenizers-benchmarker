# Testing and Quality
Last updated: 2026-08-20

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
boundary test, plus frontend `npm ci`, lint, unit tests, and production build.
The backend job creates `app/server/.venv` before installing the test extra so
the CI type-checking environment matches the repository Pyright configuration.
The Windows `app/tests/run_tests.bat` runner additionally supports live
backend/frontend startup, the configured pytest target, and optional frontend
test scripts.

## Development Cache and Artifact Locations
- Pytest’s collection cache and temporary test directory are under
  `assets/cache/pytest` and `assets/cache/pytest-basetemp`.
- Ruff, mypy, Python bytecode, coverage, Playwright, uv, pip, npm, and Angular
  persistent-build caches are under their respective `assets/cache` folders.
- `app/client/dist`, `app/server/.venv`, and `app/client/node_modules` remain in
  their established locations because they are the runtime build output or
  installed dependency trees rather than tool caches.

## Cross-language Quality Gates
- Keep architecture layering intact: API -> contracts/services -> repository.
- Treat `test_architecture_boundaries.py` as the executable ownership contract;
  module-only refactors do not require a schema migration.
- Do not bypass contract validation models.
- Do not duplicate business logic across backend and frontend without necessity.
- Add or adjust tests when changing behavior, contracts, or data schemas.
