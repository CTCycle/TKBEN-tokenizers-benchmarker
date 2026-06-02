# Design Tokens
Last updated: 2026-06-02

## Typography
- Base family:
  - `'Segoe UI', Roboto, system-ui, -apple-system, BlinkMacSystemFont, sans-serif`
- Global body text:
  - default foreground `#f5f5f5` on dark background
- Typical scale in the current UI:
  - 11-12px: dense labels, legends, and compact metadata
  - 13-15px: body text, form labels, and inputs
  - 16-20px: section titles and KPI values
- Labels frequently use uppercase plus letter-spacing for hierarchy.

## Layout and Spacing
- App shell is fixed header plus fixed tab navigation plus a scrollable content region.
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
- Page grids are used for dataset, tokenizer, and benchmark dashboards and collapse under responsive breakpoints.

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
- Success messages use green-tinted surfaces and borders.
- Error banners use red surfaces and borders.
- Warning areas use amber and yellow palettes.

Accessibility baseline:
- Maintain high contrast on dark surfaces.
- Focus state uses a visible ring with highlight border.
