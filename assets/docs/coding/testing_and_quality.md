# Testing and Quality
Last updated: 2026-08-02

## Tooling and Tests
- Lint and format with Ruff, or the project-standard equivalent if it changes in the future.
- Type check expectations are Pylance-compatible typing.
- Run `app/server/.venv/Scripts/python.exe -m basedpyright -p app/server/pyrightconfig.json` from the repository root; the configured gate requires zero errors and does not fail on warnings.
- Tests use pytest.
- Minimum test impact coverage:
  - `tests/unit`
  - relevant `tests/e2e` when behavior crosses API and UI boundaries
- Browser/live validation uses Playwright or pytest-playwright; the in-app browser is not a validation path.

The repository CI gate currently runs backend compileall, Ruff, BasedPyright,
unit tests, and an OpenAPI smoke import, plus frontend `npm ci`, lint, and
production build. The Windows `app/tests/run_tests.bat` runner additionally
supports live backend/frontend startup, the configured pytest target, and
optional frontend test scripts.

## Cross-language Quality Gates
- Keep architecture layering intact: API -> service -> repository.
- Do not bypass domain validation models.
- Do not duplicate business logic across backend and frontend without necessity.
- Add or adjust tests when changing behavior, contracts, or data schemas.
