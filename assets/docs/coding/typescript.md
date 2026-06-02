# TypeScript Rules
Last updated: 2026-06-02

## General
- Keep strict typing; avoid `any` unless unavoidable and documented.
- Model API payloads in `src/types/api.ts` and reuse those types.
- Centralize API paths via constants and services; do not hardcode endpoints repeatedly.
- Prefer functional React components and hooks.

## State and UI
- Keep page orchestration in pages, contexts, and hooks, not deeply inside leaf components.
- Keep presentational components stateless where possible.
- Normalize and guard server payloads before rendering.
- Preserve accessibility attributes already used in components, including labels, roles, and `aria-*`.

## Styling
- Reuse existing CSS tokens and component class patterns from `App.css`.
- Do not introduce conflicting style systems for small incremental changes.
- Keep responsive behavior aligned with current breakpoints at `1100px`, `900px`, and `700px`.
