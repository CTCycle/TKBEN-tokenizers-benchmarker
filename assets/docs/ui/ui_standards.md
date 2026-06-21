# UI Standards
Last updated: 2026-06-17

## Spacing and Layout
- Use the existing spacing tokens:
  - `--space-1: 4px`
  - `--space-2: 8px`
  - `--space-3: 12px`
  - `--space-4: 16px`
  - `--space-5: 24px`
- Prefer 8px rhythm for new layout spacing. Use 4px only for tight label/detail spacing.
- Keep page-level gaps at 24px, card groups at 12-16px, and compact control groups at 8px.
- Preserve the documented breakpoints: `1100px`, `900px`, and `700px`.

## Typography
- Use the global system font stack from `App.css`.
- Keep dense labels and metadata at 12px.
- Use 13-15px for body text, inputs, tables, and form labels.
- Use 16-20px for section values and KPI emphasis.
- Labels may use uppercase and letter spacing, but avoid adding new sizes unless a real hierarchy gap exists.

## Color System
- Keep the dark theme:
  - background: `--color-bg`
  - panels: `--color-panel`, `--color-panel-alt`
  - borders: `--color-border`, `--color-border-subtle`
  - text: `--color-text`, `--color-muted`
  - accent: `--color-accent`
- Use semantic tokens for danger, warning, and success states instead of raw one-off colors.
- Use chart tokens for Recharts fills, axes, grids, and tooltips. Do not introduce unrelated chart palettes.
- Do not rely on color alone for meaning; pair status colors with visible text.

## Components
- Buttons must use `primary-button`, `secondary-button`, or `icon-button`.
- Icon buttons should stay 34-36px square, include an `aria-label`, and show `:focus-visible`.
- Inputs, selects, and textareas must use visible labels with `htmlFor`/`id` or a clear `aria-label`.
- Cards should use shared panel/card surfaces: tokenized border, radius, padding, and background.
- Tables must allow intentional horizontal scrolling on narrow screens instead of clipping text.
- Loading states should use spinner plus concise status copy. Progress bars should expose live status text.
- Error and warning banners should be dismissible only when the underlying issue is recoverable.
- Modals must have a labelled dialog container, a close action, and scroll safely within the viewport.

## Do and Don't
- Do reuse existing tokens before adding new values.
- Do keep incremental polish aligned with the current product structure.
- Do verify responsive behavior visually when changing layout.
- Do not introduce a second styling system.
- Do not add decorative visual noise or redesign dashboard workflows.
- Do not hide overflow unless a child element has its own accessible scroll region.
