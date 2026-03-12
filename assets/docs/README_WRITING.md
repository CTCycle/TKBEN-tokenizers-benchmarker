# README Writing Rules (TKBEN)

## 1. Goal
README content must describe the current TKBEN product accurately and avoid legacy project terminology.

## 2. Required Sections
Use this structure:
1. Project Overview
2. Installation
3. How to Use
4. Setup and Maintenance
5. Resources
6. Configuration
7. License

## 3. Content Rules
- Describe user workflows, not internal implementation details.
- Keep backend/frontend explanations high-level but concrete.
- Reference real routes only:
  - `/dataset`
  - `/tokenizers`
  - `/cross-benchmark`
- Reference real API families only:
  - `/datasets`, `/tokenizers`, `/benchmarks`, `/jobs`, `/keys`
- Do not mention removed features (for example database browser pages/endpoints).

## 4. Runtime Accuracy
- Align commands and ports with:
  - `TKBEN/start_on_windows.bat`
  - `release/tauri/build_with_tauri.bat`
  - `TKBEN/settings/.env.local.example`
  - `TKBEN/settings/.env.local.tauri.example`
- Use `TKBEN/resources/database.db` as the default embedded DB path.

## 5. Media and Links
- If screenshots are included, they must match current UI pages.
- Keep file path references valid in this repository.

## 6. Quality Bar
- Prefer concise, factual wording.
- Avoid speculative claims.
- Update README when runtime/setup/API contracts change.
