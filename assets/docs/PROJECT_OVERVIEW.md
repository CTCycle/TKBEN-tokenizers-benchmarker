# Project Overview
Last updated: 2026-06-02

## Purpose
This file is the root index for `assets/docs`. Read it first to find the narrowest topic file for the task at hand.

## How To Navigate
1. Start with this file only.
2. Identify the topic area that matches the task.
3. Open the smallest leaf file that answers the question.
4. Open sibling files only when the task clearly crosses topic boundaries.
5. Do not read the entire tree unless the task explicitly requires broad context.

## Naming Rules
- All documentation files and folders under `assets/docs` use lower-case names.
- Topic folders group related leaf files by subject.
- Root-level files are reserved for entry points and top-level guidance.

## Documentation Ontology
### Root
- `project_overview.md`
  - Entry point and index for the full documentation tree.

### Architecture
- `architecture/system_overview.md`
  - Repository layout, entry points, and runtime interaction topology.
- `architecture/backend_api.md`
  - API surface and endpoint catalog.
- `architecture/execution_and_data_flow.md`
  - Layered backend flow, module responsibilities, and async/sync behavior.
- `architecture/persistence.md`
  - Data storage model and persisted artifact locations.
- `architecture/benchmark_contract.md`
  - Benchmark payload, runtime metadata, and report contract notes.

### Coding
- `coding/python.md`
  - Python runtime, typing, validation, async, and structure guidance.
- `coding/typescript.md`
  - Frontend TypeScript, state, UI, and styling guidance.
- `coding/testing_and_quality.md`
  - Testing, linting, and cross-language quality gates.

### Runtime
- `runtime/modes.md`
  - Supported runtime modes and operational differences.
- `runtime/startup.md`
  - Launcher commands and startup procedures.
- `runtime/configuration.md`
  - Environment variables and structured settings.
- `runtime/deployment.md`
  - Dependencies, packaging, and distribution notes.

### UI
- `ui/design_tokens.md`
  - Typography, layout, spacing, and color system.
- `ui/components_and_patterns.md`
  - Navigation, controls, forms, overlays, and feedback states.
- `ui/experience.md`
  - Page structure, UX rules, responsiveness, accessibility, and design principles.

## Reading Order
1. Read this root index.
2. Open the smallest leaf file that covers the current question.
3. Expand to adjacent files only when the task crosses topic boundaries.
4. Return here when you need to jump to a different topic branch.

## Context Rules
- Read documentation files only when required by the current task.
- Defer reading until the task proves the file is needed.
- Keep all affected documents updated whenever implementation changes alter behavior.
- Always include a `Last updated: YYYY-MM-DD` line when modifying a document.
- Do not read all `SKILL.md` files by default.
- Pre-select files to read based on folder structure and user intent before opening documentation.

## Environment Rules
- Windows is the default operating environment for this project.
- Document and support both PowerShell and CMD usage patterns where commands differ.
- Keep environment guidance aligned with launcher/runtime scripts in `start_on_windows.bat`, `release/tauri/*.bat`, and `app/tests/run_tests.bat`.
- Update this section when new environment-specific constraints or solutions are introduced.
