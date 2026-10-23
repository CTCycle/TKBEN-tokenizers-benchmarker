# Experience
Last updated: 2026-10-23

## Page Structure
- `DatasetPage`
  - Dataset selection and upload, validation workflow, persisted analytics dashboard.
  - The validation wizard's Sampling & Filters step runs a debounced
    server-side preview of the selected dataset: document, empty, and length
    statistics, a length histogram, and a live "N of M documents match; K will
    be analyzed" line. The count slider is capped at the matching document
    count, length inputs show the real range as placeholders, and a filter
    combination that matches nothing blocks Next and Run Validation.
- `TokenizerExaminationPage`
  - Tokenizer selection and reporting plus vocabulary panel. The selected
    tokenizer preview reflects vocabulary size from a report as soon as the
    first report finishes loading or generating.
  - Collapsible Tokenizer Playground below the dashboard: paste up to 2000
    characters, optionally filter and select up to 3 tokenizers from the local
    catalog, and compare token chips (token text with ID), token counts, and
    unknown counts per tokenizer. Selected tokenizer chips stay visible when
    the filter changes and choices carry across previews on the current page.
    The selector uses a bounded virtual list for large catalogs. Results are
    transient, labelled as a single-pass preview rather than a benchmark
    measurement, and per-tokenizer failures render as inline error cards.
- `CrossBenchmarkPage`
  - Run and open benchmark reports, compare tokenizer metrics against a
    report-scoped baseline, clone eligible configurations, and edit report tags.
    The run and clone wizard keeps the header, step navigation, and actions
    visible while step content scrolls, resetting the scroll position between
    steps. Step labels are visually separated and wrap on narrow screens.
- `SettingsPage`
  - Configure supported runtime defaults through a left-side section
    navigation and background-level form sections.
- `AppShell`
- Branded header with logo, integrated primary tabs, and a stroked Settings sliders icon. Provider credentials are reached through Settings → Keys.

Navigation hierarchy:
- Root redirects to `/dataset`.
- Unknown routes redirect to `/dataset`.

Catalog interactions:
- Dataset and tokenizer preview catalogs support debounced search, source filtering, and numeric threshold filtering.
- Filtering refreshes catalog results without changing the current benchmark selection.
- Empty states distinguish an unfiltered empty catalog from a catalog with no matches.
- The dataset-add dialog separates predefined datasets, Hugging Face name entry, and custom CSV/Excel upload into keyboard-navigable tabs.
- Dataset dashboard data is normalized before chart rendering; invalid optional histogram, frequency, Zipf, and word-cloud payloads produce empty or unavailable states instead of breaking the page.

## User Experience Standards
- Maintain workflow continuity:
  - start long operation -> show progress -> poll job -> render result
  - display progress percentages with at most two decimal places so floating
    point precision does not leak into status text
- Startup readiness:
  - show a single-lane pipeline where source words enter the left port of a
    schematic tokenizer, are split into subword pieces, and leave the right
    port as `##`-prefixed tokens while the backend initializes;
  - keep transient connection failures quiet;
  - surface a slow-start message and a retryable failure state when the
    existing launcher readiness window is exceeded;
  - keep the startup artwork animated while connecting, then pause it in the
    terminal failure state so the retry action is visually stable;
  - reveal the normal shell only after backend health is confirmed.
- Keep interaction language consistent with existing labels and panel semantics.
- Use dismissible errors for recoverable issues.
- Keep clone eligibility and tag validation errors inline; do not use browser
  alert dialogs for these workflows.
- Preserve dashboard-first workflows by loading persisted reports before deep visual analysis.
- Settings uses a full-width two-column layout: visible left-side navigation for
  Data, Tokenizers, Runtime, and Keys; the content column expands to the
  available viewport and falls back to a horizontally scrollable tab row on
  narrow screens. Keyboard tab navigation remains available.
- Benchmark report diagnostics must explicitly surface:
  - tokenizer failure rows with `status/error_type/error_message`
  - metric availability state
  - timing boundary definitions

## Responsiveness
Current breakpoints in CSS:
- `max-width: 1100px`
- `max-width: 900px`
- `max-width: 700px`

Required behavior:
- Multi-column dashboards collapse to fewer columns or a single column.
- Header and nav paddings plus tab sizing adapt for narrow screens.
- Modal dimensions and table overflow adjust for mobile and narrow viewports.
- Baseline controls, comparison strips, report-tag chips, and the clone wizard
  remain usable without horizontal page overflow.
- Open modals lock document scrolling and restore the previous scroll position when closed. Benchmark chart sizing also switches to compact behavior at `700px`.

## Accessibility
- Preserve keyboard navigability for all controls.
- Keep `aria-label`, `aria-expanded`, `aria-controls`, and dialog semantics in interactive components.
- Maintain a visible `:focus-visible` style on tabs, buttons, and inputs.
- Ensure charts and tables include textual context so key metrics are not color-only.
- Comparison deltas must remain text-labeled and must not rely on green/red
  color semantics.
- Respect reduced motion via existing `prefers-reduced-motion` overrides.

## Design Principles
- Consistency first: reuse existing tokens and component classes.
- Clarity over decoration: prioritize readable metrics and workflows.
- Predictability: navigation, action placement, and panel semantics should remain stable.
- Avoid introducing parallel style systems for incremental features.

## Visual validation matrix
- Review the three primary routes at 1920x1080, 1440x900, 1024x768, and 390x844.
- Check shell alignment, page gutters, panel edges, chart title/plot alignment, contained table overflow, and dialog viewport fit.
- Treat populated, empty, loading, error, and long-identifier states as required visual states; visual polish must not hide existing diagnostics or controls.
- Include filtered catalog loading and no-match states, plus the integrated header navigation and logo, in route-level visual review.
- Include startup loading, slow-start, failure, retry, reduced-motion,
  normal desktop, and narrow viewport states in launch validation.
- Include benchmark point, bucket, and distribution widgets, their data-table disclosures, modal scroll locking, and malformed/empty dataset chart payloads in focused component review.
- Include baseline selection and persistence, clone eligibility/copying, report
  tag editing, and narrow-screen report-manager/wizard states in benchmark review.
