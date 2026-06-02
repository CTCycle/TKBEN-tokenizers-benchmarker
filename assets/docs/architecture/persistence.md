# Persistence
Last updated: 2026-06-02

## Data Persistence
- Default embedded persistence:
  - SQLite file: `app/resources/database.db`
- Optional external persistence:
  - PostgreSQL via `postgresql+psycopg` when `DATABASE_EMBEDDED=false` in `settings/.env`
- Non-DB persisted artifacts:
  - `app/resources/sources/datasets` for download caches and uploads
  - `app/resources/sources/tokenizers` for tokenizer caches and custom uploads
  - `app/resources/logs` for runtime logs

## Persistence Model Notes
- Embedded SQLite is the default local storage path for the repository.
- PostgreSQL is selected through environment configuration when embedded storage is disabled.
- Repository code should treat database access as an injected dependency rather than as global state.
