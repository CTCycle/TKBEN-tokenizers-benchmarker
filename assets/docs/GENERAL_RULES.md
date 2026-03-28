## WEB SEARCH
Use web search when external facts or version-sensitive information must be verified.
For repository-local behavior, prefer source code and scripts in this workspace.

## REQUIRED DOCUMENTATION REVIEW
Before any task, review the relevant files in `assets/docs`:

- `GENERAL_RULES.md` (mandatory for every task)
- `ARCHITECTURE.md` (system structure and API surface)
- `BACKGROUND_JOBS.md` (async job behavior and contracts)
- `GUIDELINES_PYTHON.md` (when changing Python code)
- `GUIDELINES_TYPESCRIPT.md` (when changing TypeScript code)
- `GUIDELINES_TESTS.md` (when adding/updating tests)
- `PACKAGING_AND_RUNTIME_MODES.md` (when changing runtime/bootstrap/packaging)
- `README_WRITING.md` (when changing README content)
- `WINDOWS_DESKTOP_RELEASE_2026-03-25.md` (when changing Windows release process or artifacts)

## SKILLS REFERENCE
When a task matches an available skill workflow, use the relevant skill instructions from the active skills repository.

## DOCUMENTATION UPDATES
If changes materially affect behavior, architecture, setup, runtime modes, or release outputs, update the relevant file(s) in `assets/docs` and report that update in your final summary.

## CROSS-LANGUAGE PRINCIPLES

### Code quality
- Prefer clear naming, focused modules, and low-coupling design.
- Optimize for readability and testability over cleverness.

### Testing and automation
- Keep checks actionable: format, lint, type-check, tests, and security validation where applicable.
- Prefer deterministic tests and reproducible local commands.

### Security
- Validate external input early.
- Never hardcode secrets.
- Keep attack surface minimal and follow least-privilege defaults.

## EXECUTION RULES
- Use PowerShell by default for terminal commands in this repository.
- Use `cmd /c` only for `.bat` execution or CMD-specific syntax.
- For Python commands, prefer `runtimes/.venv` when present:
  - `.\runtimes\.venv\Scripts\python.exe -m ...`
- For Node.js commands, prefer the bundled runtime:
  - `.\runtimes\nodejs\npm.cmd run <script>` from `TKBEN/client`

