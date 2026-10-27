# TKBEN Tokenizer Benchmarker

TKBEN is a local web application for inspecting text datasets and comparing tokenizers on the same data. It combines dataset analysis, tokenizer reports, and benchmark dashboards in one place. The supported launcher runs on Windows; the interface opens in your browser.

![Dataset dashboard showing corpus statistics, character composition, and length distributions](assets/figures/dataset-dashboard.jpg)

*Dataset dashboard for a saved WikiText analysis.*

![Tokenizer report with vocabulary statistics and preview](assets/figures/tokenizer-dashboard.jpg)

*Tokenizer report and vocabulary preview.*

![Cross-benchmark dashboard comparing tokenizer metrics](assets/figures/cross-benchmark.png)

*Cross-benchmark dashboard comparing three tokenizers.*

## What you can do

- Import CSV or XLSX data, or add a predefined or Hugging Face dataset. Run validation to inspect document lengths, vocabulary, character composition, and other corpus statistics.
- Download tokenizer assets or upload a `tokenizer.json`. Inspect vocabulary and tokenizer reports, or compare how up to three local tokenizers split text in the playground.
- Run a cross-benchmark on a saved dataset with up to five prepared tokenizers. Compare performance, vocabulary, and fidelity metrics; reopen reports, select a baseline, and export dashboards to PDF.
- Track concurrent dataset validation, tokenizer report, and benchmark jobs in the Jobs panel. Filter the dataset and tokenizer catalogs and manage application settings in the interface.

Benchmark results depend on the dataset, selected metrics, run settings, and machine. They should be interpreted in that context rather than as a universal tokenizer ranking.

## Get started

On Windows, clone or download the repository and open PowerShell in its root directory:

```powershell
.\start_on_windows.ps1
```

Choose **Launch application**. On first launch, the launcher may download its portable Python, Node.js, and uv runtimes, install dependencies, initialize the database, and build the frontend. It opens the application once the API and frontend are ready. Keep the launcher window open and press **Ctrl+C** there to stop the app.

To start directly without the menu, run `.\start_on_windows.ps1 -Launch`. The launcher serves the Angular frontend and FastAPI API from the same local address. See [startup](assets/docs/runtime/startup.md) for maintenance options and troubleshooting.

## Run a comparison

1. In **Datasets**, add a dataset and run its validation pipeline to inspect its contents.
2. In **Tokenizers**, add the tokenizer assets you want to compare and open their reports.
3. In **Cross Benchmark**, choose **Run**, select a saved dataset and tokenizers, and configure the metrics and run settings.
4. Review the resulting dashboard. Saved reports can be reopened, tagged, compared with a baseline, and exported.

Use the Help button in the application for page-specific guidance. Hugging Face downloads need network access; gated assets also need suitable credentials.

## Data and configuration

SQLite is the default database; PostgreSQL can be configured under **Settings → Database**. Dataset and tokenizer artifacts are stored in the configured data directory, which defaults to `data`. Settings → Storage controls data and log paths; path changes take effect after a backend restart and do not move existing data. Application settings and dashboard preferences are stored in the active database. Hugging Face keys are managed under **Settings → Keys**.

The launcher creates `settings/.env` from `settings/.env.example` if needed. See [configuration](assets/docs/runtime/configuration.md) for listener settings, storage, and database details.

## Development

The frontend uses Angular and the backend uses Python and FastAPI. For a two-terminal development setup, run the following from the repository root (with Python/uv and Node.js available):

```powershell
# Terminal 1: API
Set-Location app/server
uv sync
uv run python -m uvicorn server.app:app --app-dir .. --reload --host 127.0.0.1 --port 5000
```

```powershell
# Terminal 2: frontend
Set-Location app/client
npm ci
npm run dev -- --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The development server proxies `/api` to `127.0.0.1:5000` by default; set `FASTAPI_HOST` and `FASTAPI_PORT` in the frontend terminal to use another backend address. The supported test runner is `.\app\tests\run_tests.bat`. See the [documentation index](assets/docs/project_index.md) for architecture, configuration, and testing details.

## Contributing

Issues and focused pull requests are welcome. Describe the problem or change and include relevant validation.

## License

[MIT](LICENSE)
