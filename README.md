# TKBEN Tokenizer Benchmarker
Last updated: 2026-08-31

[![Release](https://img.shields.io/github/v/release/CTCycle/TKBEN-tokenizers-benchmarker?display_name=tag)](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/releases)
![Python](https://img.shields.io/badge/python-%3E%3D3.14-3776AB?logo=python&logoColor=white)
![Node.js](https://img.shields.io/badge/node.js-%3E%3D22-339933?logo=node.js&logoColor=white)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CTCycle Portfolio](https://img.shields.io/badge/CTCycle-Portfolio-58a6ff?style=flat-square)](https://ctcycle.github.io/CTCycle/)
[![CI](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/actions/workflows/ci.yml?query=branch%3Adevelop)

## 1. Project Overview

TKBEN is a local web application for understanding how tokenizers behave on real text. It helps you inspect datasets, examine tokenizer vocabularies, measure tokenizer performance, and compare several tokenizers under the same conditions.

The application is useful when you need to answer questions such as:

- How varied, repetitive, or uneven is a text dataset?
- How does a tokenizer divide that text into token pieces?
- Which tokenizer is faster or more economical for this dataset?
- Does a tokenizer preserve text reliably when it is encoded and decoded?
- Can the results be saved, revisited, and shared as a report?

TKBEN runs locally. The browser is the user interface, while a local Python service performs the analysis and stores datasets, tokenizer assets, and reports in the application workspace. SQLite is the default local store; an externally managed PostgreSQL store is also supported for advanced setups. Hugging Face provides optional dataset and tokenizer sources, and completed dashboards can be exported as PDF files.

### Main workflows

- **Dataset validation**: choose a predefined or Hugging Face dataset, or upload a local CSV/Excel file. Select the checks you want, choose a sample or document limit, and review saved statistics and visualizations.
- **Tokenizer examination**: discover or add tokenizer identifiers, download supported assets, or upload a custom `tokenizer.json`. Inspect vocabulary size, token-length behavior, special tokens, and a paginated vocabulary preview.
- **Cross-benchmark comparison**: choose a saved dataset and up to five tokenizers, select benchmark measures, run a repeatable comparison, customize the dashboard, and export the result as a PDF.

The application opens on the **Dataset** page and provides three primary pages: **Dataset**, **Tokenizers**, and **Cross Benchmark**. A normal session moves from dataset preparation to tokenizer preparation and then to comparison, but saved reports can be reopened at any time without repeating the work.

### How the analysis works

A tokenizer converts text into smaller pieces called **tokens**. Those pieces are the units that a language model reads. Different tokenizers can represent the same sentence with different numbers and kinds of pieces, which affects speed, memory use, context-window usage, and how well unusual words or characters are handled.

TKBEN keeps the comparison grounded in the text you choose:

1. The selected dataset is downloaded or imported and saved locally.
2. Dataset validation describes its size, structure, vocabulary, frequency patterns, and possible quality issues.
3. The same saved text can be used to inspect and benchmark multiple tokenizers.
4. Benchmark reports record the selected dataset, tokenizers, measures, sample size, and runtime context so that results can be interpreted later.

No single metric identifies the best tokenizer for every purpose. TKBEN presents several complementary signals so you can balance speed, token usage, fidelity, vocabulary behavior, and resource consumption for your own data.

### Main ideas behind the metrics

- **Dataset diversity**: type-token ratio compares distinct words with total words. Shannon entropy describes how evenly word usage is distributed; a higher value generally means frequency is less concentrated, but it is not a quality score by itself.
- **Frequency and redundancy**: Zipf views show how word frequency falls as rank increases. Exact and near-duplicate rates indicate repeated content, while concentration and rare-tail measures show whether a small group of words dominates the corpus.
- **Tokenizer vocabulary**: vocabulary size, token lengths, special-token behavior, normalization, and fallback behavior help explain what the tokenizer can represent and how it divides text.
- **Benchmark efficiency**: throughput measures how much text can be processed per unit of time, while latency describes how long individual observations take. Higher throughput and lower latency are usually preferable, but both depend on the machine and the chosen run settings.
- **Fragmentation**: this describes how many token pieces are used for words or text spans. Fewer pieces can reduce token usage, but a lower value is not automatically better if it comes with poor coverage or other trade-offs.
- **Round-trip fidelity**: these checks examine whether text remains stable after encoding and decoding. They describe token or normalized-text stability, not semantic correctness or whether the text is well-written.

## 2. Installation

### 2.1 Windows (recommended)

Windows users should use the project launcher. From the repository root, run:

```powershell
.\start_on_windows.ps1
```

Choose **Launch application** from the menu. On first use, the launcher prepares the required local runtimes and dependencies, creates `settings/.env`, starts the application services, waits for them to become ready, and reports the address to open in your browser. The first launch may take a few minutes and requires an internet connection so that missing runtimes and packages can be obtained.

On later launches, the prepared environment is reused when it is still valid. You normally do not need to start the frontend or backend separately on Windows.

If Windows blocks the automatic browser opening, this does not necessarily mean that TKBEN failed to start. Copy the local URL printed by the launcher and open it manually.

### 2.2 macOS / Linux

The automatic portable-runtime setup is Windows-only. On macOS or Linux, install these prerequisites using your normal system tools:

- Python 3.14 or newer
- Node.js 22 or newer
- `uv`

From the repository root, use two terminals. In the first terminal, prepare and start the local service:

```bash
cd app/server
uv sync
uv run python -m uvicorn server.app:app --app-dir .. --host 127.0.0.1 --port 5000
```

In the second terminal, prepare and start the web interface:

```bash
cd app/client
npm ci
npm run preview -- --host 127.0.0.1 --port 8000 --strictPort
```

Then open `http://127.0.0.1:8000`. Keep both terminals running while you use the application. Stop each process with `Ctrl+C` when finished.

## 3. How to Use

### 3.1 Start the local web application

On Windows, run the launcher from the repository root:

```powershell
.\start_on_windows.ps1
```

The launcher prints the active local address. Use that address rather than guessing a port, especially if you have changed local settings or another application is already using the default port. TKBEN is local by default: the address is intended for use on the same computer.

### 3.2 A typical first run

1. Launch TKBEN and open the local address shown by the launcher.
2. On **Dataset**, add a predefined or Hugging Face dataset, or upload a CSV/Excel file.
3. Select the dataset, choose validation measures and sampling options, and run validation. Wait for the progress indicator to finish, then open the saved report.
4. On **Tokenizers**, search for or enter tokenizer identifiers, download the assets you need, or upload a custom tokenizer, and inspect the resulting reports.
5. On **Cross Benchmark**, choose the validated dataset, select up to five tokenizers and the measures you want, review the summary, and start the benchmark.
6. Open the completed report, adjust the dashboard if useful, and export a PDF when you are ready to share the comparison.

Long downloads, validation runs, and benchmarks are handled as background jobs. Progress is shown in the interface; wait for the job to finish before opening its report or starting a dependent workflow.

### 3.3 Dataset validation

Use the **Dataset** page to build a local, reusable text collection.

- Filter the catalog by name, source, or document count.
- Use **Add dataset** to choose a predefined source, enter a Hugging Face dataset name and optional configuration, or upload a local `.csv`, `.xls`, or `.xlsx` file.
- For local files, keep a clear column containing the document text and remove empty or non-text rows where possible. The application looks for common text fields when importing data.
- Select a dataset row to see its document count and available actions.
- In the validation wizard, choose the metric groups you need, then choose either a fraction of the dataset or a document count. Optional length limits and empty-document exclusion help focus the analysis.
- Give the session a useful name if you expect to compare several validation passes.

The predefined C4 option intentionally uses a manageable sample of up to 10,000 documents for local work. If you need a different portion or the full source, add the Hugging Face dataset by its name instead.

The saved dashboard includes aggregate and word-level statistics, character composition, document- and word-length histograms, frequency views, entropy, duplicate indicators, concentration signals, and a word cloud when the selected data supports them. A missing chart or unavailable measure means that the relevant data was not available for that run; it is not silently replaced with a zero.

### 3.4 Tokenizer examination

Use the **Tokenizers** page to prepare and understand tokenizer assets.

- Search or filter discovered repositories by name, author, task, access type, or vocabulary size.
- Add one or more tokenizer identifiers manually when you already know which assets you want.
- Download selected Hugging Face tokenizers, or upload a custom `tokenizer.json` for a tokenizer that is not hosted there.
- Open a tokenizer report to inspect basic metadata, vocabulary statistics, token-length distributions, special-token information, and a paginated vocabulary preview.

Only tokenizers that download and load successfully become available for benchmarking. If a download fails, check the identifier, network connection, and access rights before trying again.

### 3.5 Cross-benchmark comparison

Use **Cross Benchmark** to compare tokenizer behavior on the same saved dataset.

1. Open **Run benchmark** and select the metric categories or individual measures you want.
2. Select a dataset and up to five prepared tokenizers.
3. Choose how many documents to process and give the run a descriptive name.
4. Review the summary and start the run. Advanced controls are available for trial counts, batching, token-processing behavior, and per-document statistics; the defaults are suitable for an initial comparison.
5. Wait for the progress indicator to finish, then select the saved report.

The report dashboard presents comparable metrics as charts. Depending on the data, you can switch between compatible chart styles, reorder widgets, hide measures that are not useful for the current question, and open a data table beneath a chart. Dashboard layout choices are saved in the browser for later visits; changing the layout does not rerun the benchmark.

The report also shows tokenizer-specific failures and unavailable measures explicitly. A failed tokenizer is not displayed as a misleading zero-value result. You can cancel an active benchmark from the run wizard; a cancelled run does not create a completed benchmark report.

### 3.6 Reading results responsibly

For a fair comparison, keep the following consistent:

- the dataset and document sample
- the selected metrics
- token-processing options and trial settings
- the computer and general runtime conditions, when comparing speed or memory

Performance values are affected by CPU, memory, operating system, background activity, batch size, and the number of trials. Treat them as measurements for the selected environment, not universal properties of a tokenizer.

When reviewing a report:

- A larger vocabulary is not automatically better; it may improve some coverage patterns while increasing memory or model-size costs.
- Higher throughput generally means faster processing, while lower latency generally means quicker individual responses.
- Lower fragmentation often means fewer token pieces, but vocabulary coverage and fidelity should be considered alongside it.
- Entropy, Zipf, duplicate rates, and concentration describe the dataset distribution. They help you understand the evaluation context rather than rank datasets on a single quality scale.
- `N/A`, unavailable, or missing values should be read as “not measured or not applicable,” not as zero.

### 3.7 Product Snapshots

The following full-page captures show the main screens with populated sample data. Your counts, charts, and report values will depend on the datasets, tokenizers, metrics, and sampling choices you use.

The settings view centralizes optional local runtime choices. Most users can keep the generated defaults and work entirely from the launcher.

![Settings](assets/figures/settings.png)
*Settings page showing the local runtime, port, logging, and integration controls used by the launcher.*

Dataset dashboard with a loaded validation session, aggregate statistics, histograms, and word-cloud analytics.

![Dataset workspace](assets/figures/dataset.png)
*Full-page dataset dashboard for a 200-document C4 validation. Review dataset health, lexical metrics, distributions, entropy, concentration, and word-cloud signals before benchmarking.*

Tokenizers dashboard with an opened tokenizer report, vocabulary statistics, and populated token preview table.

![Tokenizers workspace](assets/figures/tokenizers-overview.png)
*Full-page GPT-2 tokenizer report showing model metadata, a 50,257-token vocabulary, token-length distribution, and a paginated token preview.*

Cross-benchmark dashboard with a loaded run summary and comparative metric panels.

![Cross-benchmark dashboard](assets/figures/cross-benchmark.png)
*Full-page C4 cross-benchmark report comparing GPT-2 and RoBERTa across throughput, vocabulary, latency, round-trip fidelity, memory, and run diagnostics.*

## 4. Setup and Maintenance

Most users only need **Launch application** in the Windows launcher menu. Use the other options when the menu or a release note specifically calls for them.

The maintenance menu can help you:

- install or update the local dependencies used by the application
- rebuild the web interface after an application update
- initialize or recheck the local database
- run the project’s validation checks
- inspect or remove logs and disposable caches
- check for or apply an application source update
- remove all locally saved application data after an explicit confirmation

The launcher checks the local database when the application starts and applies required updates automatically. Do not delete the database or saved resource folders manually while the application is running.

The **Remove All Data** action is permanent for the local workspace. It removes saved datasets, tokenizer files, reports, logs, and stored Hugging Face access-key material while preserving the application files themselves. Back up anything you may need before confirming this action.

## 5. Troubleshooting

### The browser did not open

Copy the local address printed by the launcher and open it manually. On managed Windows machines, browser auto-open can be blocked even when the services started correctly.

### The launcher appears to be taking a long time

The first launch may be downloading runtimes, installing packages, preparing the database, or building the web interface. Allow the progress indicators to finish. If there is no progress, check your internet connection and available disk space, close duplicate TKBEN windows, and run the launcher again.

### The application says that an address or port is already in use

Close another TKBEN instance or the application using that local address, then restart TKBEN. If you intentionally need different ports, change the local settings and restart both parts of the application. Always use the URL printed by the launcher after the change.

### A Hugging Face dataset or tokenizer cannot be downloaded

Check the repository name and optional dataset configuration, then retry with a stable network connection. For gated or private resources, accept the source’s terms and add a Hugging Face read-access key using the key button in the application header. If a key is already configured but access is denied, verify that it belongs to an account allowed to use that resource.

### A CSV or Excel upload is rejected or produces little data

Make sure the file is one of the supported formats and contains a column with document text. Remove empty rows, confirm that the file is not damaged, and try a smaller sample if the file is very large. A spreadsheet with only numeric fields or unrelated metadata cannot provide a useful text analysis.

### A report or chart is empty

Confirm that the relevant dataset validation or benchmark finished successfully and that you opened the saved report rather than only selecting an input. Reset catalog filters if no items are visible. Some measures require enough documents or observations and may correctly appear as unavailable.

### A tokenizer failed inside a benchmark

Open the run diagnostics to see which tokenizer failed. Check that the tokenizer was downloaded or uploaded completely, that its source is accessible, and that the dataset contains usable text. Remove and prepare the tokenizer again if necessary, then start a new run.

### A benchmark is slow or the numbers vary between runs

Large datasets, many selected metrics, optional language-model measures, and detailed per-document statistics require more time and memory. Start with a smaller document sample and the default settings. For speed comparisons, repeat runs under similar computer conditions and compare the same sample.

### PDF export does not complete

Open a completed report before exporting and choose a folder where you can create files. If you cancel the native save dialog, no PDF is created and no error is expected. If export still fails, reduce the dashboard to the measures you need and try again.

### macOS or Linux reports that a command is missing or permission is denied

Confirm that Python, Node.js, and `uv` are installed and available in the terminal you are using. Run the commands from the repository directories shown above, use a shell with permission to read the project, and keep the backend and frontend terminals separate. The Windows launcher is not available on these platforms, so the two manual processes must both be started.

## 6. Saved Data and Resources

TKBEN keeps its working data locally so that completed analyses can be reopened after a restart.

- `app/resources`: saved datasets, tokenizer assets, reports, the local database, and logs. Back up the relevant contents of this folder if you need to preserve your work.
- `settings`: local settings and templates used by the launcher. Most users never need to edit this folder.
- `assets/figures`: screenshots used in this guide.
- `assets/docs`: deeper project and runtime reference material for advanced users and maintainers.

The application does not provide cloud synchronization by default. Moving TKBEN to another computer therefore requires you to preserve any local data you want to keep and then prepare the local runtime on the new machine.

## 7. Optional Configuration

The Windows launcher creates `settings/.env` automatically and supplies sensible defaults. Most users should leave those defaults unchanged.

You may need to edit the local settings only when you want to:

- use a different local address or port because of a conflict
- store the local workspace somewhere other than the repository’s default data folder
- change whether backend logs appear in a separate window
- connect to an externally managed PostgreSQL database instead of the default embedded store

Restart TKBEN after changing `settings/.env`. Keep this file private: it can contain machine-specific paths, database connection details, or other sensitive values. Hugging Face access keys should be added and managed through the application’s key manager rather than placed in screenshots or shared documentation.

## 8. Releases and Data Safety

Versioned source releases are available from the [GitHub releases page](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker/releases). A source archive contains the application files, not your local datasets, downloaded tokenizer assets, credentials, logs, or generated reports.

Before updating to a new release:

1. Export or back up reports and any local data you need to retain.
2. Use the launcher’s update or installation options on Windows, or follow the platform-specific setup above on macOS/Linux.
3. Allow the first startup to finish its local database and dependency preparation.
4. If a report from a much older application version is no longer listed, rerun the dataset validation or benchmark with the current version. This avoids mixing incompatible report formats.

## 9. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
