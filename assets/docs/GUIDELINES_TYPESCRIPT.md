# TypeScript Guidelines (TKBEN Client)

## 1. Stack Baseline
- React 19 + TypeScript + Vite
- Router: `react-router-dom`
- Charts: `recharts`
- Strict TS config enabled (`TKBEN/client/tsconfig.app.json`)

## 2. Code Organization
- Pages live in `TKBEN/client/src/pages`.
- Shared state lives in `TKBEN/client/src/contexts`.
- API calls live in `TKBEN/client/src/services`.
- Shared API types live in `TKBEN/client/src/types/api.ts`.

## 3. Typing Rules
- Do not introduce `any` unless unavoidable.
- Prefer explicit domain types from `types/api.ts`.
- For uncertain payloads, use `unknown` + safe narrowing.
- Keep helper parsing/normalization functions deterministic and side-effect free.

## 4. API Integration
- Use endpoints from `TKBEN/client/src/constants.ts`.
- Keep `/api`-based requests compatible with Vite proxy and Docker nginx proxy.
- For long-running backend operations, use the existing job polling utilities (`services/jobsApi.ts`).

## 5. UI and Routing Consistency
- Preserve active app routes:
  - `/dataset`
  - `/tokenizers`
  - `/cross-benchmark`
- Keep sidebar and page behavior aligned with current navigation model (`AppShell`, `Sidebar`).

## 6. Quality Checks
- Run build/lint before finalizing frontend-heavy changes:
  - `npm run build`
  - `npm run lint`
- Keep accessibility labels and button semantics for interactive controls.

