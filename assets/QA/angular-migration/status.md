# Angular migration status

Last updated: 2026-08-16

| Area | Status | Evidence / next gate |
| --- | --- | --- |
| React baseline capture | browser verified | React route references are preserved under `react-baseline/` at 1440x900 and 390x844. A disposable React re-export was attempted for the remaining viewports but Vite config loading is blocked by the managed temp-directory permission boundary; Angular was independently checked at all four required sizes. |
| Git migration branch | migrated | `migration/react-vite-to-angular` created from clean `develop`. |
| Angular foundation | migrated | Angular 22.1 standalone/zoneless scaffold, strict TypeScript, angular-eslint, Vitest, production build, preview server, dev proxy, and Node 22.23.1 launcher pin are in place. |
| Application shell and routing | browser verified | Standalone shell, active tabs, redirects, deep links, key-manager trigger, and route rendering verified in the in-app browser on the Angular production preview at port 8000. |
| Shared API/state services | migrated | Typed HttpClient services, FastAPI detail normalization, RxJS debounce/switchMap job polling, Signals stores, and storage keys are implemented. |
| Dataset route | browser verified | Catalog filters, selection, add/upload/download dialog, validation wizard, latest-report/delete actions, persisted report summary, character composition, histograms, Zipf curve, entropy/duplicate/concentration indicators, word cloud worker/fallback, and responsive layout verified. |
| Tokenizer/key-management routes | browser verified | Catalog filtering, report/vocabulary controls, upload dialog, key list/add/reveal/activation/delete UI and responsive rendering verified; real key mutation remains intentionally limited to disposable/non-secret coverage. |
| Cross-benchmark route | browser verified | Persisted report loading, payload-shaped tables, vertical/horizontal/interval/dot-whisker/box/histogram/grouped-bar/heatmap renderers, titles/tooltips, legends, run dialog, layout persistence, CDK drag/drop and Space/arrow/Space keyboard controls, customization/reset, mobile overflow, and export error handling verified. |
| Automated tests | functionally verified | Angular lint, production build, clean `npm ci`, and the post-parity Vitest unit suite (5 files, 11 tests) passed. Complete Windows runner passed: 267 Python tests passed, 5 skipped, frontend unit tests passed; focused Angular E2E suites passed (11 app-flow, 1 dashboard). |
| Final cleanup | migrated | React/Vite source, entrypoint, Vite config, and Node tsconfig removed; Angular preview/proxy, docs, launcher, and Node 22.23.1 pin updated. |

## Pause checkpoint

Work is intentionally paused after the latest feature-parity checkpoint. The branch remains in progress; next session should complete a post-change clean install and full E2E rerun, repeat the final cross-route visual parity pass, and perform the final legacy-marker/obsolete-dependency audit before declaring the migration complete.
