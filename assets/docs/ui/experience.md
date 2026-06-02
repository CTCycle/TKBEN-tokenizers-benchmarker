# Experience
Last updated: 2026-06-02

## Page Structure
- `DatasetPage`
  - Dataset selection and upload, validation workflow, persisted analytics dashboard.
- `TokenizerExaminationPage`
  - Tokenizer selection and reporting plus vocabulary panel.
- `CrossBenchmarkPage`
  - Run and open benchmark reports and compare tokenizer metrics.
- `AppShell`
  - Global header, tabs, and Hugging Face key manager access.

Navigation hierarchy:
- Root redirects to `/dataset`.
- Unknown routes redirect to `/dataset`.

## User Experience Standards
- Maintain workflow continuity:
  - start long operation -> show progress -> poll job -> render result
- Keep interaction language consistent with existing labels and panel semantics.
- Use dismissible errors for recoverable issues.
- Preserve dashboard-first workflows by loading persisted reports before deep visual analysis.
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

## Accessibility
- Preserve keyboard navigability for all controls.
- Keep `aria-label`, `aria-expanded`, `aria-controls`, and dialog semantics in interactive components.
- Maintain a visible `:focus-visible` style on tabs, buttons, and inputs.
- Ensure charts and tables include textual context so key metrics are not color-only.
- Respect reduced motion via existing `prefers-reduced-motion` overrides.

## Design Principles
- Consistency first: reuse existing tokens and component classes.
- Clarity over decoration: prioritize readable metrics and workflows.
- Predictability: navigation, action placement, and panel semantics should remain stable.
- Avoid introducing parallel style systems for incremental features.
