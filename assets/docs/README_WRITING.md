# README Writing Rules (TKBEN)

## 1. Goal
README content must describe the current TKBEN product accurately and avoid stale/legacy terminology.

## 2. Required Section Order
Use this structure:
1. Project Overview
2. Installation
3. How to Use
4. Setup and Maintenance
5. Resources
6. Configuration
7. License

## 3. Content Rules
- Describe user workflows first; keep implementation details minimal.
- Keep backend/frontend explanations high-level but concrete.
- Reference active UI routes only:
  - `/dataset`
  - `/tokenizers`
  - `/cross-benchmark`
- Reference active API families only:
  - `/datasets`, `/tokenizers`, `/benchmarks`, `/exports`, `/jobs`, `/keys`
- Do not mention removed features/endpoints.

## 4. Runtime Accuracy
Keep commands/ports aligned with:
- `TKBEN/start_on_windows.bat`
- `release/tauri/build_with_tauri.bat`
- `TKBEN/settings/.env.local.example`
- `TKBEN/settings/.env.local.tauri.example`

Default embedded DB path:
- `TKBEN/resources/database.db`

## 5. Media and Links
- Screenshots must reflect current UI pages and naming.
- Keep all file path references valid in this repository.

## 6. Quality Bar
- Prefer concise, factual wording.
- Avoid speculative claims and outdated release statements.
- Update README whenever runtime/setup/API contracts change.
