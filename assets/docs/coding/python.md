# Python Rules
Last updated: 2026-08-20

## Runtime and Dependencies
- Python target version: `>=3.14` from `pyproject.toml`.
- Use `app/server/.venv` as the canonical backend environment.
- Do not create new virtual environments in this project.
- Keep dependency state aligned with:
  - `pyproject.toml`
  - `app/server/uv.lock`
- Use `uv` as the dependency and runtime manager used by project scripts.

## Typing
- Type annotations are required for public APIs and non-trivial internal logic.
- Prefer built-in generics: `list[str]`, `dict[str, Any]`.
- Prefer `|` unions over `typing.Union`.
- Use `collections.abc` abstract types where appropriate.
- Treat typing as required quality, not optional documentation.

BasedPyright runs from the repository root with `pyrightconfig.json`, targets
Python 3.14 in recommended mode, and must report zero errors. Warnings are
reported for visibility but do not fail the gate.

## Validation and API Contracts
- Use Pydantic contract models for request, response, and settings validation.
- Avoid ad-hoc validation when a model can encode the contract.
- Use explicit HTTP status codes for success and error paths.
- Keep response models stable and consistent.
- Handle errors safely and preserve traceability through job IDs and logs.

## Import and Ownership Boundaries

- Keep API routers at the HTTP boundary; they may depend on contracts,
  configurations, and services, but not repositories directly.
- Keep contract modules implementation-free: contracts must not import API,
  service, or repository modules.
- Keep repositories focused on persistence and query concerns; repositories
  must not import services or API modules.
- Put shared metric catalogs and dependency-free helpers in `server/common`.
- Use the canonical ownership paths under `server/contracts`,
  `server/configurations`, `server/services`, and `server/repositories`.
- Do not recreate the removed `server.domain` or
  `server.repositories.serialization` namespaces as aliases.

## Async and Long-Running Work
- Use async endpoints only as orchestrators.
- Offload blocking or CPU-heavy logic via `asyncio.to_thread(...)` or background jobs.
- Do not perform heavy CPU work directly in async handlers.
- Use the job system (`JobManager`) for long-running operations.
- Long-running flows must support start, poll status, and cancel.

## Code Structure
- Keep functions focused and small.
- Make side effects explicit.
- Prefer composable logic over deeply nested control flow.
- Keep imports at file top.
- Avoid nested function definitions unless strictly necessary.
- Use classes to group related stateful logic.
- Keep modules near or below about 1000 LOC when practical.
- Add comments only where they materially improve clarity or safety.
- Avoid broad stylistic rewrites unrelated to the task.
