# Canonical Configuration
Last updated: 2026-09-10

TKBEN has two configuration inputs with non-overlapping responsibilities:

- `settings/.env` contains machine/runtime values such as database connection details, network bindings, security switches, and storage paths.
- `settings/configurations.json` contains application tuning values for datasets, tokenizers, benchmarks, and jobs.

`server.configurations.get_server_settings()` is the canonical resolved backend settings interface. Structured JSON settings are required explicitly. Missing or unknown settings fail during validation rather than falling back to hidden model defaults.

Boolean environment settings use the documented `true` and `false` literals. Compatibility aliases such as `yes`, `no`, `1`, and `0` are not accepted.

The environment profile is loaded before configuration modules that may reach environment-derived settings. This prevents manual startup from observing different values than the Windows launcher.
