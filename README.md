# TKBEN Tokenizer Benchmarker

<p align="center"><img src='app/client/public/tkben-logo.png' alt='TKBEN logo' width="112"></p>

**A local workbench for inspecting text datasets, exploring tokenizers, and comparing tokenizer behavior.**


[![License](https://img.shields.io/github/license/CTCycle/TKBEN-tokenizers-benchmarker?style=flat-square)](LICENSE) [![Release](https://img.shields.io/github/v/release/CTCycle/TKBEN-tokenizers-benchmarker?display_name=tag&style=flat-square)](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/releases) [![CI](https://img.shields.io/github/actions/workflow/status/CTCycle/TKBEN-tokenizers-benchmarker/ci.yml?branch=develop&label=CI&style=flat-square)](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/workflows/ci.yml?query=branch%3Adevelop) [![Python](https://img.shields.io/badge/Python-3.14.7-3776AB?logo=python&logoColor=white&style=flat-square)](https://www.python.org/) [![Angular](https://img.shields.io/badge/Angular-22-DD0031?logo=angular&logoColor=white&style=flat-square)](https://angular.dev/) [![Platform](https://img.shields.io/badge/Platform-Windows-0078D4?logo=windows&logoColor=white&style=flat-square)](https://www.microsoft.com/windows)

## Overview

TKBEN helps you examine text data and compare how tokenizers process the same material. The local web application stores datasets, tokenizer assets, and reports on the computer running the app. SQLite is the default database; PostgreSQL is available for advanced setups. Hugging Face sources are optional.

No single metric identifies the best tokenizer for every task. Results depend on the chosen dataset, measures, run settings, and machine.

## Key Features

- Add predefined or Hugging Face datasets, or upload local CSV and XLSX files, then review dataset and lexical statistics.
- Search for tokenizer assets, download them, or upload a custom `tokenizer.json`; inspect vocabulary, token lengths, and special tokens.
- Compare up to five prepared tokenizers on the same saved dataset using selected performance, vocabulary, and fidelity measures.
- Save and reopen reports, customize dashboard views, inspect run diagnostics, and export completed reports as PDFs.
- Configure, test, and switch SQLite or PostgreSQL connections from Settings → Database. Runtime settings and dashboard preferences save automatically to, and remain scoped to, the active database.

## Technology Stack

- Python 3.14.7, uv, and FastAPI backend.
- Angular 22 client, built with Node.js 22.23.1.
- SQLite by default, with optional PostgreSQL.
- Optional Hugging Face access for dataset and tokenizer sources.

## Installation and Usage

The supported Windows workflow uses the repository launcher. From the repository root in PowerShell, run:

```powershell
.\start_on_windows.ps1
```

Choose **Launch application**. The first launch may download portable runtimes and dependencies, prepare the database, and build the web client. Open the local address printed by the launcher.

## Configuration

The launcher creates `settings/.env` from `settings/.env.example` when the file is missing. Set `TKBEN_DATA_DIR` to choose a persistent data root other than the default `data` directory. Use **Settings → Database** to configure and test a SQLite or PostgreSQL connection, **Settings** for application options, and **Settings → Keys** for Hugging Face access keys.

## Documentation

- [Runtime startup](assets/docs/runtime/startup.md)
- [Runtime configuration](assets/docs/runtime/configuration.md)
- [Release notes and procedure](assets/docs/runtime/release.md)

## License

Licensed under the MIT License. See [LICENSE](LICENSE).
