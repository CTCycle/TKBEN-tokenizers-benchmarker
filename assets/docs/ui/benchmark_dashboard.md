# Benchmark Dashboard
Last updated: 2026-07-22

Cross-benchmark reports use schema version 2 and report version 3. The report exposes a normalized `dashboard` payload whose widgets are emitted only for calculated, finite metric series; zero remains a valid calculated value.

Widget definitions are canonical backend definitions. Each provides its category, display metadata, visualization, width, and default visibility. The browser persists the committed widget order and hidden IDs under `tkben:cross-benchmark-dashboard-layout:v1`; malformed storage resets to defaults and unavailable report widgets are omitted without deleting the global preference.

The dashboard header can restore the complete default layout. This removes the persisted layout, returning both visibility and widget ordering to the current report defaults; the control is disabled while the active layout already matches those defaults.

The customizer lists only widgets available in the active report. It keeps a draft selection until Apply, supports category select/clear, prevents an empty applied selection, closes on Escape, and returns focus to its trigger. The dashboard grid uses two columns above 900px and one column at or below 900px; wide widgets span both columns. Widgets can be reordered with pointer drag or the DnD-kit keyboard flow (Space, arrow keys, Space); both use the same pure order rule and persist the committed order.

Each chart widget includes a collapsed `View data table` disclosure. It exposes the same values as the chart, including intervals, distribution quartiles, buckets, units, and sample counts, so chart interpretation is not dependent on color or pointer interaction.

Benchmark PDF export receives the active report plus `visible_widget_ids` and `ordered_widget_ids`. The server selects only IDs visible in the payload, orders those IDs by the submitted layout, and renders the normalized dashboard widgets with wide widgets on their own page.

PDF export feedback is shown next to the export control as an accessible success or error message, rather than a browser alert. Cancelling the native save picker remains silent.

Reports from earlier versions are neither listed nor loaded. They must be rerun.
