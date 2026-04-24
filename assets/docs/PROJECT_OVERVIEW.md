# PROJECT_OVERVIEW
Last updated: 2026-04-24

## FILES INDEX
- PROJECT_OVERVIEW.md  
  Documentation entrypoint with docs index and rules for how documentation and context loading must be handled.

- ARCHITECTURE.md  
  End-to-end system architecture: repository structure, entry points, API map, layered design, persistence, and async/sync execution model.

- CODING_RULES.md  
  Coding and quality standards for Python and TypeScript in this repository, including typing, validation, async, tooling, and test expectations.

- RUNTIME_MODES.md  
  Supported runtime modes (local webapp, packaged desktop, test mode), startup commands, env/config differences, and deployment/build notes.

- UI_STANDARDS.md  
  Enforceable UI system derived from the current implementation (tokens, layout, components, states, responsiveness, accessibility).

## CONTEXT RULES
- Read documentation files only when required by the current task.
- Defer reading until the task proves the file is needed.
- Keep all affected documents updated whenever implementation changes alter behavior.
- Always include a `Last updated: YYYY-MM-DD` line when modifying a document.
- Do not read all `SKILL.md` files by default.
- Pre-select files to read based on folder structure and user intent before opening documentation.

## ENVIRONMENT RULES
- Windows is the default operating environment for this project.
- Document and support both PowerShell and CMD usage patterns where commands differ.
- Keep environment guidance aligned with launcher/runtime scripts in `TKBEN/*.bat`, `release/tauri/*.bat`, and `tests/run_tests.bat`.
- Update this section when new environment-specific constraints or solutions are introduced.
