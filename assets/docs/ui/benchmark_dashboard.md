# Benchmark Dashboard
Last updated: 2026-07-21

Cross-benchmark reports use schema version 2 and report version 3. The report exposes a normalized `dashboard` payload whose widgets are emitted only for calculated, finite metric series; zero remains a valid calculated value.

Widget definitions are canonical backend definitions. Each provides its category, display metadata, visualization, width, and default visibility. The browser persists the committed widget order and hidden IDs under `tkben:cross-benchmark-dashboard-layout:v1`; malformed storage resets to defaults and unavailable report widgets are omitted without deleting the global preference.

The customizer lists only widgets available in the active report. It keeps a draft selection until Apply, supports category select/clear, prevents an empty applied selection, closes on Escape, and returns focus to its trigger. The dashboard grid uses two columns above 900px and one column at or below 900px; wide widgets span both columns.

Reports from earlier versions are neither listed nor loaded. They must be rerun.
