# CODING_RULES
Last updated: 2026-04-24

## Scope
These rules apply to all code in this repository. Keep changes small, verifiable, and aligned with existing module boundaries.

## Python Rules
### Runtime and dependencies
- Python target version: `>=3.14` (from `pyproject.toml`).
- Use `runtimes/.venv` when available.
- Do not create new virtual environments in this project.
- Keep dependency state aligned with:
  - `pyproject.toml`
  - `runtimes/uv.lock`
- Use `uv` as the dependency/runtime manager used by project scripts.

### Typing
- Type annotations are required for public APIs and non-trivial internal logic.
- Prefer built-in generics: `list[str]`, `dict[str, Any]`.
- Prefer `|` unions over `typing.Union`.
- Use `collections.abc` abstract types where appropriate.
- Treat typing as required quality, not optional documentation.

### Validation and API contracts
- Use Pydantic/domain models for request/response and settings validation.
- Avoid ad-hoc/manual validation when a model can encode the contract.
- Use explicit HTTP status codes for success and error paths.
- Keep response models stable and consistent.
- Handle errors safely and preserve traceability through job IDs/logs.

### Async and long-running work
- Use async endpoints only as orchestrators.
- Offload blocking or CPU-heavy logic via `asyncio.to_thread(...)` or background jobs.
- Do not perform heavy CPU work directly in async handlers.
- Use the job system (`JobManager`) for long-running operations.
- Long-running flows must support:
  - start
  - poll status
  - cancel

### Code structure
- Keep functions focused and small.
- Make side effects explicit.
- Prefer composable logic over deeply nested control flow.
- Keep imports at file top.
- Avoid nested function definitions unless strictly necessary.
- Use classes to group related stateful logic.
- Keep modules near or below ~1000 LOC when practical.
- Add comments only where they materially improve clarity/safety.
- Avoid broad stylistic rewrites unrelated to the task.

### Tooling and tests
- Lint/format with Ruff (or project-standard equivalent if changed in future).
- Type check expectations are Pylance-compatible typing.
- Tests use pytest.
- Minimum test impact coverage:
  - `tests/unit`
  - relevant `tests/e2e` when behavior crosses API/UI boundaries

## TypeScript Rules
Inferred from current React + Vite + TypeScript codebase.

### General
- Keep strict typing; avoid `any` unless unavoidable and documented.
- Model API payloads in `src/types/api.ts` and reuse those types.
- Centralize API paths via constants/services; do not hardcode endpoints repeatedly.
- Prefer functional React components and hooks.

### State and UI
- Keep page orchestration in pages/contexts/hooks, not deeply inside leaf components.
- Keep presentational components stateless where possible.
- Normalize and guard server payloads before rendering.
- Preserve accessibility attributes already used in components (labels, roles, `aria-*`).

### Styling
- Reuse existing CSS tokens and component class patterns from `App.css`.
- Do not introduce conflicting style systems for small incremental changes.
- Keep responsive behavior aligned with current breakpoints (`1100px`, `900px`, `700px` media queries).

## Cross-language quality gates
- Keep architecture layering intact: API -> service -> repository.
- Do not bypass domain validation models.
- Do not duplicate business logic across backend and frontend without necessity.
- Add/adjust tests when changing behavior, contracts, or data schemas.
