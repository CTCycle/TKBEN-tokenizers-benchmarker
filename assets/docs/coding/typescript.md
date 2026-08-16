# TypeScript Rules
Last updated: 2026-08-02

## General
- Keep strict typing; avoid `any` unless unavoidable and documented.
- Model API payloads in `app/client/angular/app/core/api/api.models.ts` and reuse those types.
- Keep catalog filters typed with `DatasetCatalogFilters` and `TokenizerCatalogFilters`; serialize only populated, valid query values in the API service layer.
- Centralize API paths via constants and services; do not hardcode endpoints repeatedly.
- Prefer standalone Angular components, injectable services, Signals, and typed reactive forms.

## State and UI
- Keep page orchestration in feature pages and signal stores, not deeply inside leaf components.
- Keep presentational components stateless where possible.
- Keep catalog filtering local to the page while exposing refresh operations through signal stores; debounce server-backed filter changes and guard against stale responses with `switchMap`.
- Normalize and guard server payloads before rendering.
- Keep shared normalization and derivation in pure helpers under `app/client/angular/app/core/utils/`.
- Keep reusable lifecycle behavior in injectable services and Angular destroy-aware RxJS streams; dashboard layout persistence remains under the benchmark signal store.
- Keep chart rendering and the payload-shaped benchmark data table separate from the widget shell so visualization changes do not change the underlying accessible data.
- Preserve accessibility attributes already used in components, including labels, roles, and `aria-*`.

## Styling
- Reuse existing CSS tokens and component class patterns from `App.css`.
- Do not introduce conflicting style systems for small incremental changes.
- Keep responsive behavior aligned with current breakpoints at `1100px`, `900px`, and `700px`.
- Prefer generic typed component boundaries for reusable controls such as `CatalogFilterToolbar`; preserve literal source and numeric-operator unions instead of accepting arbitrary event strings.
