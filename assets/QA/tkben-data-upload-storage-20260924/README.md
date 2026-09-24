# Dataset size and storage recovery validation

Date: 2026-09-24
Base revision: `6889b1bf743fc2f9bdd3f3691c24c368517038ad` (`develop`)
Scope: T2-01 extension and `data.large-file-and-disk-exhaustion`

## Result

The dataset upload limit and controlled database-full recovery pass. The wider
large-file and disk-exhaustion debt remains **PARTIAL** because this validation
did not exhaust the host filesystem.

## Evidence

- The upload API accepted a CSV at the configured 25 MiB limit and returned
  HTTP 202. It rejected a 25 MiB + 1 byte upload with HTTP 413 and did not
  dispatch a second job.
- A 24 MiB CSV was parsed by `DatasetService.upload_and_persist`; its single
  large text row persisted and the dataset reached `ready` with one document.
- An in-memory SQLite database with foreign keys enabled was capped with
  `PRAGMA max_page_count`. After one document batch committed, the next batch
  raised SQLite's actual `SQLITE_FULL` error. Both the custom CSV upload path and
  the dataset-backed `persist_dataset` path removed the incomplete dataset and
  its documents. Once the page limit was restored, a retry with the same name
  completed with two documents and status `ready`.
- Focused dataset route and storage tests passed **9/9**. The complete backend
  unit suite passed **515/515** in 33.37 seconds.
- Ruff passed on all changed Python files. BasedPyright reported **0 errors and
  2,018 warnings** for the backend. The unit suite reported 23 deprecation
  warnings.

The unit suite used the repository's canonical `runtimes/cache` root with an
isolated per-run pytest temp directory beneath it. The standard temp directory
was denied by a Windows ACL; the optional `stepwise` pytest plugin was disabled
because its installed module was also denied. The cache-layout tests passed on
the adjusted runner, and the canonical cache root remained unchanged.

## Remaining limits and other open gates

- A full host drive was not simulated. If a real filesystem-full condition also
  prevents the cleanup transaction from committing, the application logs that
  cleanup failure and an incomplete `loading` row may remain until storage is
  available. Do not treat the SQLite page-limit simulation as proof of Windows
  filesystem recovery.
- No near-limit XLSX import was attempted; ordinary CSV/XLSX coverage is in the
  [Dataset and Settings QA record](../tkben-t2-01-t4-02-t5-02-dataset-settings-validation-20260924/README.md).
- Hugging Face tokenizer discovery/report flow remains **BLOCKED** pending
  provider/network validation; gated/private access remains **BLOCKED** without
  an approved credential. The synthetic `persist_dataset` check did not call
  Hugging Face.
- PostgreSQL runtime validation remains **BLOCKED**. A local server answered
  on `127.0.0.1:5433`, but repository settings select SQLite, no PostgreSQL
  connection credentials are configured, and the Docker engine is unavailable.
  The local service was not treated as disposable.
- Manual PDF inspection of other visualization overrides and vocabulary exports
  beyond 23 entries remains open. Hosted CI evidence is available for the prior
  validation commit, while release publication evidence remains **PARTIAL** and
  was not part of this storage slice.
