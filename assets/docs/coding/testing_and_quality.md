# Testing and Quality
Last updated: 2026-06-02

## Tooling and Tests
- Lint and format with Ruff, or the project-standard equivalent if it changes in the future.
- Type check expectations are Pylance-compatible typing.
- Tests use pytest.
- Minimum test impact coverage:
  - `tests/unit`
  - relevant `tests/e2e` when behavior crosses API and UI boundaries

## Cross-language Quality Gates
- Keep architecture layering intact: API -> service -> repository.
- Do not bypass domain validation models.
- Do not duplicate business logic across backend and frontend without necessity.
- Add or adjust tests when changing behavior, contracts, or data schemas.
