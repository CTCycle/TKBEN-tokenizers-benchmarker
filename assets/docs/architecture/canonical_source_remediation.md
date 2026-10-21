# Canonical Source Remediation
Last updated: 2026-10-21

## Objective
TKBEN should expose one authoritative implementation and one authoritative source of truth for each responsibility. Runtime compatibility paths are retained only when a current external dependency requires them.

## Canonical ownership
- `settings/.env` and the process environment own bootstrap-only values,
  including network, paths, launcher behavior, and security.
- `<TKBEN_DATA_DIR>/database-config.json` is the database-independent source of
  truth for database connection settings. Database passwords are encrypted.
- Typed Pydantic runtime models own application defaults. The selected database
  stores sparse user overrides for the exact Settings API allowlist; it never
  stores bootstrap values or database credentials.
- `server.configurations.get_server_settings()` owns the resolved immutable backend settings object.
- `server.configurations` loads the environment before importing settings or
  database surfaces that resolve environment-derived values.
- Alembic owns schema evolution and one-time data migrations.
- SQLAlchemy repositories own durable report and catalog persistence.
- Backend Pydantic contracts own API response semantics.
- Angular domain stores own page-level client state.

## Removed compatibility paths
- The raw `ConfigurationManager` payload API and its `get_block()` / `get_value()` accessors were removed. Backend code now loads the validated configuration directly.
- The permissive boolean coercion helper was removed. Runtime booleans accept the documented `true` or `false` values and reject aliases.
- Launcher cleanup no longer carries or recreates historical cache locations. Only the current managed cache roots remain.
- Legacy `DATABASE_*` environment import and `.env` rewriting were removed. Database connection settings come only from `database-config.json`.
- The legacy `<TKBEN_DATA_DIR>/runtime-settings.json` import was removed. Runtime settings and dashboard preferences come only from the active database.
- The deprecated `GET /api/tokenizers/settings` compatibility endpoint was removed; `/api/settings` is the only settings contract.
- The pre-baseline Alembic revisions were replaced by the single `0001_current_schema` baseline. Existing databases from older revisions are rejected and must be recreated.

## Bootstrap rule
The environment profile must be loaded before configuration modules that can reach environment-derived paths or database settings are imported. `server.configurations` performs that bootstrap before importing its settings/startup surface.

The bootstrap and validation path is now the merged `develop` behavior. Runtime
booleans accept only `true` or `false`; typed environment and runtime models
reject unknown fields. The application consumes a single immutable effective
settings snapshot after merging typed defaults and validated sparse overrides,
rather than reconstructing configuration from competing sources. Runtime
updates replace only runtime-editable nested models after an atomic persistence
commit; existing operations retain their captured snapshot.

## Remaining canonicalization work
The following audit findings require broader contract changes and are intentionally tracked separately from the initial configuration cleanup:
- consolidate environment-derived resource and log paths into the resolved settings object;
- remove duplicated dataset metric response representations and frontend reconstruction fallbacks;
- replace generic dashboard export dictionaries with typed persisted-report references plus UI-only presentation overrides;
- generate frontend API contract types from backend OpenAPI rather than maintaining handwritten copies;
- expose dataset preset/capability metadata from the backend instead of duplicating operational catalog data in Angular;
- make launcher and CI toolchain versions consume one repository-owned version declaration.

These items should be completed without introducing transitional runtime adapters. Where persisted data requires conversion, prefer one-time Alembic migration and then accept only the current format.
