# Canonical Source Remediation
Last updated: 2026-09-10

## Objective
TKBEN should expose one authoritative implementation and one authoritative source of truth for each responsibility. Runtime compatibility paths are retained only when a current external dependency requires them.

## Canonical ownership
- `settings/.env` owns environment-specific runtime values.
- `settings/configurations.json` owns structured application tuning values. Every structured setting is explicit and required; Pydantic validates values but does not provide competing defaults.
- `server.configurations.get_server_settings()` owns the resolved immutable backend settings object.
- Alembic owns schema evolution and one-time data migrations.
- SQLAlchemy repositories own durable report and catalog persistence.
- Backend Pydantic contracts own API response semantics.
- Angular domain stores own page-level client state.

## Removed compatibility paths
- The raw `ConfigurationManager` payload API and its `get_block()` / `get_value()` accessors were removed. Backend code now loads the validated configuration directly.
- The permissive boolean coercion helper was removed. Runtime booleans accept the documented `true` or `false` values and reject aliases.
- Launcher cleanup no longer carries or recreates historical cache locations. Only the current managed cache roots remain.

## Bootstrap rule
The environment profile must be loaded before configuration modules that can reach environment-derived paths or database settings are imported. `server.configurations` performs that bootstrap before importing its settings/startup surface.

## Remaining canonicalization work
The following audit findings require broader contract changes and are intentionally tracked separately from the initial configuration cleanup:
- consolidate environment-derived resource and log paths into the resolved settings object;
- remove duplicated dataset metric response representations and frontend reconstruction fallbacks;
- replace generic dashboard export dictionaries with typed persisted-report references plus UI-only presentation overrides;
- generate frontend API contract types from backend OpenAPI rather than maintaining handwritten copies;
- expose dataset preset/capability metadata from the backend instead of duplicating operational catalog data in Angular;
- make launcher and CI toolchain versions consume one repository-owned version declaration.

These items should be completed without introducing transitional runtime adapters. Where persisted data requires conversion, prefer one-time Alembic migration and then accept only the current format.
