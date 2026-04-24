# UI_STANDARDS
Last updated: 2026-04-24

## Scope
This standard reflects the implemented UI in `TKBEN/client/src` and `App.css`. New UI must conform to these tokens/patterns unless a planned design-system migration is approved.

## Typography
- Base family:
  - `'Segoe UI', Roboto, system-ui, -apple-system, BlinkMacSystemFont, sans-serif`
- Global body text:
  - default foreground `#f5f5f5` on dark background
- Typical scale in current UI:
  - 11-12px: dense labels, legends, compact metadata
  - 13-15px: body text, form labels/inputs
  - 16-20px: section titles and KPI values
- Labels frequently use uppercase + letter-spacing for hierarchy.

## Layout and Spacing
- App shell is fixed-header + fixed-tab-nav + scrollable content region.
- Primary spacing tokens:
  - `--space-1: 4px`
  - `--space-2: 8px`
  - `--space-3: 12px`
  - `--space-4: 16px`
  - `--space-5: 24px`
- Standard radii:
  - `--radius-sm: 8px`
  - `--radius-md: 10px`
  - `--radius-lg: 12px`
- Common panel layout:
  - panel header
  - panel body
  - optional panel footer
- Page grids are used for dataset/tokenizer/benchmark dashboards and collapse under responsive breakpoints.

## Color System
Core tokens:
- Backgrounds:
  - `--color-bg: #161a1d`
  - `--color-panel: #21252a`
  - `--color-panel-alt: #1b1f23`
- Borders:
  - `--color-border: #353b45`
  - `--color-border-strong: #ffeb3b`
- Text:
  - `--color-text: #f5f5f5`
  - `--color-muted: #9ea7b3`
- Accent:
  - `--color-accent: #ffe81f`
  - `--color-accent-dark: #f0c000`

Semantic usage:
- Success messages: green-tinted surfaces/borders
- Error banners: red surfaces/borders (`#3b1414`, `#7f1d1d`, `#fecaca`)
- Warning areas: amber/yellow palettes

Accessibility baseline:
- Maintain high contrast on dark surfaces.
- Focus state uses visible ring (`--focus-ring` and highlight border).

## Components and Interaction Patterns
### Navigation
- Top tab navigation for:
  - `/dataset`
  - `/tokenizers`
  - `/cross-benchmark`
- Active state must be visually explicit (`app-tab--active`, `aria-current="page"`).

### Buttons
- Primary action:
  - outline accent style, filled accent on hover
- Secondary action:
  - neutral border, subtle hover fill
- Icon buttons:
  - subtle/accent/danger variants
  - keyboard focus-visible ring required

### Forms
- Inputs/select/textarea:
  - dark panel backgrounds
  - bordered fields
  - strong visible focus border + shadow
- File upload controls are icon-triggered with hidden native input where required.

### Overlays
- Modal and popover patterns:
  - centered card
  - backdrop with dark overlay/blur
  - explicit close action
  - keyboard-focus-compatible controls

### Feedback states
- Loading:
  - spinner + concise status copy
- Error:
  - dismissible banner with action
- Empty:
  - placeholder component with optional detail text
- Disabled:
  - reduced opacity and non-interactive cursor

## Page Structure
- `DatasetPage`: dataset selection/upload, validation workflow, persisted analytics dashboard.
- `TokenizerExaminationPage`: tokenizer selection/reporting + vocabulary panel.
- `CrossBenchmarkPage`: run/open benchmark reports and compare tokenizer metrics.
- `AppShell`: global header, tabs, and HF key manager access.

Navigation hierarchy:
- Root redirects to `/dataset`.
- Unknown routes redirect to `/dataset`.

## User Experience Standards
- Maintain workflow continuity:
  - start long operation -> show progress -> poll job -> render result
- Keep interaction language consistent with existing labels/panel semantics.
- Use dismissible errors for recoverable issues.
- Preserve dashboard-first workflows (load persisted report before deep visual analysis).

## Responsiveness
Current breakpoints in CSS:
- `max-width: 1100px`
- `max-width: 900px`
- `max-width: 700px`

Required behavior:
- Multi-column dashboards collapse to fewer columns/single column.
- Header/nav paddings and tab sizing adapt for narrow screens.
- Modal dimensions and table overflow adjust for mobile/narrow viewports.

## Accessibility
- Preserve keyboard navigability for all controls.
- Keep `aria-label`, `aria-expanded`, `aria-controls`, and dialog semantics in interactive components.
- Maintain visible `:focus-visible` style on tabs/buttons/inputs.
- Ensure charts/tables include textual context so key metrics are not color-only.
- Respect reduced motion via existing `prefers-reduced-motion` overrides.

## Design Principles
- Consistency first: reuse existing tokens and component classes.
- Clarity over decoration: prioritize readable metrics and workflows.
- Predictability: navigation, action placement, and panel semantics should remain stable.
- Avoid introducing parallel style systems for incremental features.
