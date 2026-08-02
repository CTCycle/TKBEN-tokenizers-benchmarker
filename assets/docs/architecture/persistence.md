# Persistence
Last updated: 2026-08-02

## Storage selection

Embedded SQLite is the default local store at `app/resources/database.db`. The
PostgreSQL backend is selected with `DATABASE_EMBEDDED=false` and
`postgresql+psycopg` settings. Database access is injected through the
repository backend.

The schema is canonical and intentionally has no migration or compatibility
layer. Existing development databases must be deleted and recreated after a
schema change. Startup validates the complete table set and fails on a
mismatch.

## Canonical tables

The schema contains `dataset`, `dataset_document`, `analysis_session`,
`metric_type`, `metric_value`, `histogram_artifact`, `tokenizer`,
`tokenizer_vocabulary`, `tokenizer_report`, `benchmark_report`, and
`hf_access_keys`. The obsolete `dataset_validation_report` table is removed.

Datasets are imported as `loading` records and become visible only after all
documents are inserted and the row is finalized as `ready`. Documents have a
dataset-local ordinal and the ready document count is stored on `dataset`.
Failed or cancelled loading imports are deleted and database cascades remove
their documents.

Metric values carry the owning dataset, enforce composite session/document
ownership, require exactly one value representation, and use partial unique
indexes for aggregate and per-document values. Tokenizer reports are
current-only: replacing a report replaces its vocabulary and report as one
logical operation. Benchmark reports keep immutable schema-3/report-5 JSON snapshots plus
projected summary columns; list queries do not select the full payload. Reports from older contracts are filtered out and never migrated; dashboard histogram bins remain inside the immutable payload snapshot.

## Backend and transaction guarantees

SQLite enables foreign keys and WAL at connection time. SQLite and PostgreSQL
share the same repository transaction behavior: bounded inserts and upserts
commit once per logical operation and roll back the whole operation on error.
Hugging Face key activation is atomic and the database permits at most one
active key. SQLite serializes writes; PostgreSQL is required for concurrent
writer validation.

Timestamps are normalized to aware UTC values at the persistence boundary.
JSON object and array fields are validated by typed SQLAlchemy decorators;
domain validation remains responsible for nested JSON contracts and histogram
array compatibility.

Serializer ownership is split by persisted domain: dataset and analysis
materialization remains in `repositories/serialization/datasets.py`, tokenizer
reports and vocabulary belong to `repositories/serialization/tokenizer_reports.py`,
and benchmark reports remain in `benchmark_reports.py`. This is an ownership
refactor only; it does not change the schema, response fields, ordering, or
transaction behavior.

## Validation

The SQLite persistence contract covers schema creation, foreign keys, lifecycle
visibility, composite ownership, partial uniqueness, value-shape constraints,
cascades, rollback, active-key uniqueness, and keyset document streaming.
PostgreSQL validation is optional when `TKBEN_TEST_POSTGRES_URL` is absent and
must be run against a disposable database before claiming PostgreSQL runtime
equivalence.
