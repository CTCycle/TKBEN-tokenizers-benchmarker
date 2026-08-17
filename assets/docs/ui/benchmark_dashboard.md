# Benchmark Dashboard
Last updated: 2026-08-17

Cross-benchmark reports use schema version 3 and report version 5. The report exposes a normalized `dashboard` payload whose widgets are emitted only for calculated, finite metric series; zero remains a valid calculated value.

Widget definitions are canonical backend definitions. Each provides its category, display metadata, strict `default_visualization`, ordered compatible visualization choices, data-driven width, and default visibility. Scalar points expose `bar` and `horizontal_bar`; interval points expose `interval_bar` and `dot_whisker`; five-number distributions expose `box_plot` and `histogram`; tokenizer-by-bucket comparisons expose `grouped_bar` only so the dashboard does not offer a table-like alternative. Histogram widgets also expose shared-edge `histogram_bins`. The browser persists the committed widget order, hidden IDs, known IDs, and per-widget choices under `tkben:cross-benchmark-dashboard-layout:v3`; malformed or older storage is ignored and the current defaults are used without migration.

The dashboard header can restore the complete default layout. This removes the persisted layout, returning visibility, widget ordering, and visualization choices to the current report defaults; the control is disabled while the active layout already matches those defaults.

The customizer lists only widgets available in the active report. It keeps a draft selection until Apply, supports category select/clear, prevents an empty applied selection, closes on Escape, and returns focus to its trigger. The dashboard grid auto-fits panels at a 340px minimum; dense data (more than four tokenizers or five buckets) spans two columns, independent of the selected visualization. Chart stages are 260px on desktop and 240px on mobile. Each widget uses an unboxed compact visualization selector: the role group remains in the DOM for accessibility, while its controls are 32px square on desktop and 34px square below 700px. Point and distribution plots use centered, data-shape-aware content frames bounded at 760px and 1280px respectively; bucket comparisons remain full width. Sparse scalar, interval, and grouped bar series cap bar thickness so one or two tokenizers do not stretch disproportionately. Custom box plots, forest plots, and histogram small multiples preserve their existing viewBoxes and aspect ratio; histogram cells use a label row and a contained, centered SVG row capped near 600px wide. Chart stages and small-multiple cells contain their SVG content, while narrow histogram grids collapse without horizontal card overflow. Widgets can be reordered with pointer drag or the Angular CDK keyboard flow (Space, arrow keys, Space); both use the same pure order rule and persist the committed order.

Each chart widget includes a visible compact icon group with one accessible, title-bearing button per compatible visualization and `aria-pressed` on the active choice. It also includes a collapsed `View data table` disclosure. The table remains payload-shaped, exposing intervals, distribution quartiles, buckets, units, and sample counts independent of active chart type.

The widget shell classifies each payload as a point, bucket, or distribution
shape before choosing the chart renderer. Compact chart sizing is derived from
the `700px` viewport breakpoint, while the data table remains available as the
stable accessible representation when a chart is visually dense or unavailable.

Benchmark PDF export receives the active report plus `visible_widget_ids`, `ordered_widget_ids`, and `visualization_by_widget_id`. The server validates every override against the widget’s compatible list, selects only IDs visible in the payload, orders those IDs by the submitted layout, and renders all eight normalized chart forms with strict parity: vertical/horizontal bars, interval bars/forest plots, box plots/histograms, and grouped bars/heatmaps.

PDF export feedback is shown next to the export control as an accessible success or error message, rather than a browser alert. Cancelling the native save picker remains silent.

Reports from earlier versions are neither listed nor loaded. They must be rerun.
