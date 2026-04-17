# TypeScript Guidelines (TKBEN Client)
Last updated: 2026-04-08


## 1. Stack Baseline
- React 19 + TypeScript + Vite
- Router: `react-router-dom` (v7)
- Charts: `recharts`
- Strict TS config enabled (`TKBEN/client/tsconfig.app.json`)

## 2. Code Organization
- Pages: `TKBEN/client/src/pages`
- Shared state: `TKBEN/client/src/contexts`
- API layer: `TKBEN/client/src/services`
- Shared API typing: `TKBEN/client/src/types/api.ts`
- Route shell/navigation: `TKBEN/client/src/components/AppShell.tsx`

## 3. Typing Rules
- Avoid `any` unless absolutely necessary.
- Prefer explicit domain types from `types/api.ts`.
- For uncertain payloads, use `unknown` and narrow safely.
- Keep parsing/normalization helpers deterministic and side-effect free.
- Do not introduce global variables or cross-module mutable singleton state.
- Keep state local to components/hooks/services and pass dependencies explicitly.

## 4. API Integration
- Use endpoint constants from `TKBEN/client/src/constants.ts`.
- Keep `/api`-based requests compatible with local and packaged runtime routing.
- For long-running backend work, use existing job polling helpers in `services/jobsApi.ts`.

## 5. UI and Routing Consistency
- Preserve active routes:
  - `/dataset`
  - `/tokenizers`
  - `/cross-benchmark`
- Keep behavior aligned with `AppShell` (header + tab navigation + key manager toggle).

## 6. Quality Checks
Run from `TKBEN/client` using bundled runtime:
```bat
..\..\runtimes\nodejs\npm.cmd run build
..\..\runtimes\nodejs\npm.cmd run lint
```

- Maintain accessibility labels and semantic controls for interactive UI elements.
