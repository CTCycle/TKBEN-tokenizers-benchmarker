# Benchmark validation campaign — 2026-09-23

## Scope and outcome

This controlled local campaign covers T3-01 (benchmark measurement), T3-02
(supported run options), and T5-04 (large-run progress, cancellation, resource
sampling, and recovery). It used the official Windows launcher with isolated
application data and logs, a deterministic local CSV dataset, a local
WordLevel tokenizer, and Chrome. The campaign was run from the working tree
based on `d412b61`.

| Gate | Result | Evidence boundary |
| --- | --- | --- |
| T3-01 | PASS | Three same-host, same-configuration reports on 1,000 documents. The samples are descriptive and do not support a general performance claim. |
| T3-02 | PARTIAL | Supported tokenization options and per-document statistics were exercised and persisted. `parallelism=2` is accepted and saved but does not affect the serial tokenizer loop. |
| T5-04 | PASS | A 10,000-document run showed progress, cancelled without saving a report, and was followed by a completed, rendered two-document run. Backend RSS samples were available; browser heap was not instrumented. |

## Runtime and workload

- Windows 11 build `10.0.26200`, Python `3.14.7`, AMD64 Family 25 Model 80
  Stepping 0 (AuthenticAMD), 12 logical and 6 physical cores, 32,108.0586 MB
  system memory.
- Package versions: tokenizers `0.22.2`, transformers `4.57.3`, NumPy `2.4.1`,
  pandas `3.0.0`, FastAPI `0.128.0`; sentencepiece `0.2.1`; tiktoken was not
  installed.
- T3-01 dataset: 1,000 deterministic documents, 29,465 characters and UTF-8
  bytes. Each baseline used warmup 2, timed trials 8, batch size 100, seed 42,
  parallelism 1, token processing options disabled, and per-document
  statistics disabled. The tokenizer was local; no network provider was used.
- The full request and response data, including raw observations, report
  configuration, runtime metadata, and configuration checks, is in
  [`t3-benchmark-measurements-and-options.json`](t3-benchmark-measurements-and-options.json).

## T3-01 raw measurements

Each report processed the same 1,000-document workload and produced 80 timed
batch latency observations. Phase timings are encode-only, dataset stream,
post-processing, and end-to-end, in that order. Throughput confidence intervals
are the report's 95% intervals.

| Run | Tokens/s (95% CI) | Latency p50 / p95 / p99 (ms), n | Encode / stream / post / end-to-end (s) | Peak RSS / memory delta (MB) |
| --- | --- | --- | --- | --- |
| 1 | 715018.5298006544 (650390.667162634–779646.3924386748) | 0.004776 / 0.00760725 / 0.00900463, n=80 | 0.0397206 / 0.0105718 / 0.0222682 / 1.0599154 | 259.9375 / 0 |
| 2 | 674748.3027620474 (577288.3171513587–772208.2883727361) | 0.0050015 / 0.00784605 / 0.01097205, n=80 | 0.0429875 / 0.0225988 / 0.0194596 / 1.1656743 | 259.94140625 / 0.00390625 |
| 3 | 512432.16451787355 (446160.33362820046–578703.9954075466) | 0.006437 / 0.0109828 / 0.01378509, n=80 | 0.0563735 / 0.0103489 / 0.0344019 / 1.3047628 | 259.94921875 / 0.0078125 |

The samples vary materially on this machine. Treat each as a local observation;
do not infer a general improvement, regression, or cross-machine comparison.
RSS is process RSS; memory delta is the reported increase from the first sample
to the observed peak.

## T3-02 supported option checks

The controlled run set observed 210 tokens with default token processing,
330 with special tokens, 360 with padding, and 150 with truncation at length 3.
The combined non-default run used 60 documents, warmup 1, timed trials 2,
batch size 7, seed 99, `parallelism=2`, special tokens/padding/truncation on,
maximum length 4, and per-document statistics with sample size 23. It produced
480 tokens (`60 × 2 × 4`) and 23 per-document samples. The retrieved saved
report configuration matched the requested fields and values.

The wizard view was visually inspected: the removed control is absent.
Persisted report `config` responses also contain only the currently declared
configuration fields. The parallelism value appears in the saved configuration
but has no observable execution effect, so T3-02 remains PARTIAL pending a
decision to implement its semantics or remove the setting.

Screenshot: [`t3-02-run-options.png`](t3-02-run-options.png).

## T5-04 load, cancellation, and recovery

- The 10,000-document benchmark displayed running progress at 20% in the UI.
- Cancel returned in 216.8898 ms; the job reached terminal `cancelled` and the
  cancelled run had no saved report.
- Eight backend RSS samples were collected at 262.09375 MB. No browser heap
  instrumentation was available in this campaign.
- A two-document run then completed successfully with token processing and
  per-document statistics enabled. The persisted run report and populated
  dashboard rendered. The browser and browser-driven HTTP error collections
  were empty.

Raw cancellation/run data is in
[`t5-04-cancellation.json`](t5-04-cancellation.json). Visual evidence:
[`t5-04-running-progress.png`](t5-04-running-progress.png) and
[`t5-04-completed-rerun.png`](t5-04-completed-rerun.png).

## Focused quality checks

- Backend service tests: 24 passed.
- Frontend unit tests: 62 passed across 14 files.
- Ruff and frontend lint passed.
- BasedPyright reported 0 errors and 2,004 warnings.
- Production frontend build passed.
- Opt-in Chrome E2E for the measurement/options campaign and cancellation
  workflow: 2 passed. Pytest emitted one existing unknown `cache_dir` option
  warning.
- `git diff --check` is recorded after the campaign evidence and ledger edits.

## Remaining limits and follow-up

- T3-02 remains PARTIAL until `parallelism` either has implemented execution
  semantics or is removed from supported controls and persisted configuration.
- T5-04 did not measure browser heap. Its required live progress, cancellation,
  available backend RSS sampling, and successful rerun checks passed.
- T3-05 populated PDF parity, T4-01/03 provider checks, T4-04 PostgreSQL,
  T4-02 public dataset download, T5-02/03 responsive coverage, and T5-05/06
  platform/release evidence remain separately tracked in the validation ledger.
- T2-01 CSV/XLSX and complete UI coverage, and T2-04 restart, collision, and
  complete UI coverage, remain documented follow-ups outside this campaign.
