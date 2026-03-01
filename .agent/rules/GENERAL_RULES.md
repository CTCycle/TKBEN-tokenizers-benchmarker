# TKBEN Agent Rules

## Scope
- These rules apply to this repository only.
- The target application is **TKBEN Tokenizer Benchmarker**.
- Do not reference legacy or unrelated projects.

## Source of Truth
- Backend code: `TKBEN/server`
- Frontend code: `TKBEN/client`
- Runtime settings: `TKBEN/settings`
- Tests: `tests`
- Packaging: `docker-compose.yml`, `docker/`

## Mandatory Document Check
Before implementing significant changes, review:
- `GENERAL_RULES.md`
- `ARCHITECTURE.md`
- `PACKAGING_AND_RUNTIME_MODES.md`
- `GUIDELINES_PYTHON.md` (if Python/backend/scripts are touched)
- `GUIDELINES_TYPESCRIPT.md` (if frontend/TS is touched)
- `GUIDELINES_TESTS.md` (if tests are added/changed)
- `README_WRITING.md` (if README is edited)

## Change Hygiene
- Keep docs aligned with code when routes, flows, configuration, packaging, or test workflow changes.
- Prefer minimal, focused changes that preserve current behavior.
- Preserve current routing model:
  - Frontend pages: `/dataset`, `/tokenizers`, `/cross-benchmark`
  - Backend API prefixes: `/datasets`, `/tokenizers`, `/benchmarks`, `/jobs`, `/keys`

## Security Baseline
- Never commit secrets.
- `HF_KEYS_ENCRYPTION_KEY` must be provided via environment settings.
- Treat uploaded files and external data as untrusted input.

## Command Conventions
- On Windows automation scripts, use `cmd /c ...` where required by the runner.

