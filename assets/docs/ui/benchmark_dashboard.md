# Benchmark Dashboard
Last updated: 2026-07-28

Cross-benchmark reports use schema version 2 and report version 4. The report exposes a normalized `dashboard` payload whose widgets are emitted only for calculated, finite metric series; zero remains a valid calculated value.

Widget definitions are canonical backend definitions. Each provides its category, display metadata, strict `default_visualization`, ordered compatible visualization choices, width, and default visibility. Scalar points expose `bar` and `lollipop`; interval points expose `interval_bar` and `dot_whisker`; five-number distributions expose `box_plot` and `range_plot`; tokenizer-by-bucket matrices expose `grouped_bar` and `heatmap`. The browser persists the committed widget order, hidden IDs, known IDs, and per-widget choices under `tkben:cross-benchmark-dashboard-layout:v2`; malformed storage resets to defaults and unavailable report widgets are omitted without deleting the global preference. The obsolete v1 key is removed without migration.

The dashboard header can restore the complete default layout. This removes the persisted layout, returning visibility, widget ordering, and visualization choices to the current report defaults; the control is disabled while the active layout already matches those defaults.

The customizer lists only widgets available in the active report. It keeps a draft selection until Apply, supports category select/clear, prevents an empty applied selection, closes on Escape, and returns focus to its trigger. The dashboard grid uses two columns above 900px and one column at or below 900px; wide widgets span both columns. Widgets can be reordered with pointer drag or the DnD-kit keyboard flow (Space, arrow keys, Space); both use the same pure order rule and persist the committed order.

Each chart widget includes a visible compact icon group with one accessible, title-bearing button per compatible visualization and `aria-pressed` on the active choice. It also includes a collapsed `View data table` disclosure. The table remains payload-shaped, exposing intervals, distribution quartiles, buckets, units, and sample counts independent of active chart type.

Benchmark PDF export receives the active report plus `visible_widget_ids`, `ordered_widget_ids`, and `visualization_by_widget_id`. The server validates every override against the widget’s compatible list, selects only IDs visible in the payload, orders those IDs by the submitted layout, and renders all eight normalized chart forms with wide widgets on their own page.

PDF export feedback is shown next to the export control as an accessible success or error message, rather than a browser alert. Cancelling the native save picker remains silent.

Reports from earlier versions are neither listed nor loaded. They must be rerun.
