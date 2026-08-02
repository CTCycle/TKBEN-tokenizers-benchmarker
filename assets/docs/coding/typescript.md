# TypeScript Rules
Last updated: 2026-08-02

## General
- Keep strict typing; avoid `any` unless unavoidable and documented.
- Model API payloads in `src/types/api.ts` and reuse those types.
- Keep catalog filters typed with `DatasetCatalogFilters` and `TokenizerCatalogFilters`; serialize only populated, valid query values in the API service layer.
- Centralize API paths via constants and services; do not hardcode endpoints repeatedly.
- Prefer functional React components and hooks.

## State and UI
- Keep page orchestration in pages, contexts, and hooks, not deeply inside leaf components.
- Keep presentational components stateless where possible.
- Keep catalog filtering local to the page while exposing refresh operations through contexts/hooks; debounce server-backed filter changes and guard against stale responses.
- Normalize and guard server payloads before rendering.
- Keep shared normalization and derivation in pure feature helpers: dataset dashboard parsing belongs in `features/dataset/datasetDashboardData.ts`, while benchmark widget shape detection belongs in `features/benchmark-dashboard/benchmarkDashboardChartUtils.ts`.
- Keep reusable lifecycle behavior in hooks. Current shared hooks cover available-dataset loading state, modal body-scroll restoration, compact chart detection at `700px`, dashboard layout persistence, and benchmark workspace orchestration.
- Keep chart rendering and the payload-shaped benchmark data table separate from the widget shell so visualization changes do not change the underlying accessible data.
- Preserve accessibility attributes already used in components, including labels, roles, and `aria-*`.

## Styling
- Reuse existing CSS tokens and component class patterns from `App.css`.
- Do not introduce conflicting style systems for small incremental changes.
- Keep responsive behavior aligned with current breakpoints at `1100px`, `900px`, and `700px`.
- Prefer generic typed component boundaries for reusable controls such as `CatalogFilterToolbar`; preserve literal source and numeric-operator unions instead of accepting arbitrary event strings.
