# HOW TO TEST

This document describes the testing strategy and instructions for TKBEN.
Tests focus on end-to-end scenarios that exercise both the UI and backend API.
Coverage includes dataset workflows, tokenizer management, benchmark validation,
and database browsing.

## Overview
- Framework: Playwright with pytest
- Language: Python
- Scope: End-to-End (E2E) UI and REST API verification

## Test Suite Structure
tests/
|-- run_tests.bat            # Automated test runner (Windows)
|-- conftest.py              # Pytest configuration and fixtures
`-- e2e/
    |-- test_app_flow.py      # UI navigation and page rendering
    |-- test_datasets_api.py  # Dataset list, upload, analysis
    |-- test_tokenizers_api.py# Tokenizer settings/scan/upload/clear
    |-- test_benchmarks_api.py# Benchmark validation + optional run
    `-- test_browser_api.py   # Database browser endpoints

## Quick Start (Recommended)
Run the batch file to start servers, run tests, and clean up:
```
tests\run_tests.bat
```

This script will:
1. Check prerequisites
2. Install Playwright browsers if needed
3. Start the backend server
4. Start the frontend server
5. Run all tests
6. Stop the servers and report results

Note: Run `TKBEN\start_on_windows.bat` at least once before running tests to ensure all dependencies are installed.

## Manual Testing

### Prerequisites
- Python 3.14+ with `pytest` and `pytest-playwright`
- Playwright browsers installed

### Setup
1. Install test dependencies:
   ```
   pip install .[test]
   ```
2. Install Playwright browsers:
   ```
   python -m playwright install
   ```

### Running Tests Manually
1. Start the application:
   ```
   TKBEN\start_on_windows.bat
   ```
2. Run tests (in a separate terminal):
   ```
   pytest tests
   ```

### Useful Options
| Option | Description |
|--------|-------------|
| `--headed` | Run with browser visible |
| `--slowmo 500` | Slow down execution (ms) |
| `--video on` | Record video of tests |
| `-v` | Verbose output |
| `-x` | Stop on first failure |
| `-k "test_name"` | Run specific test by name |

Example:
```
pytest tests/e2e/test_app_flow.py --headed --slowmo 500
```

## Environment Variables
The test suite reads the following environment variables:

- `UI_BASE_URL` or `UI_URL`: Full base URL for the frontend (default: http://127.0.0.1:7861).
- `UI_HOST`, `UI_PORT`: Used if `UI_BASE_URL` is not set.
- `API_BASE_URL`: Full base URL for the backend (default: http://127.0.0.1:8000).
- `FASTAPI_HOST`, `FASTAPI_PORT`: Used if `API_BASE_URL` is not set.
- `E2E_SAMPLE_DATASET_FILE`: Filename used for the sample dataset upload (default: `e2e_sample.csv`).

## Optional Slow Tests
Some tests require network access and are opt-in by default:

- `E2E_RUN_HF_SCAN=1` enables the HuggingFace tokenizer scan test.
- `E2E_RUN_BENCHMARKS=1` enables an end-to-end benchmark run using a small model.

These are disabled by default to keep the suite fast and predictable.

## Writing New Tests
- Place tests in `tests/e2e/` and name files `test_*.py`.
- Prefer API tests that use small payloads and avoid large dataset downloads.
- For UI tests, target stable labels/IDs (e.g., sidebar link aria-labels).

## API Test Notes
- When using `api_context.post()` to send JSON data, use the `data=` parameter with a Python dict (e.g., `data={"key": "value"}`). Playwright automatically serializes this to JSON.
- Do **not** use a `json=` parameter as it does not exist in Playwright's `APIRequestContext`.

## Troubleshooting
- Connection refused: ensure the app is running before tests.
- Playwright not found: use `python -m playwright install`.
