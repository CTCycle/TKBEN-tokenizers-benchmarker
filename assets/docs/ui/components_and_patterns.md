# Components and Patterns
Last updated: 2026-07-19

## Navigation
- Top tab navigation for:
  - `/dataset`
  - `/tokenizers`
  - `/cross-benchmark`
- Active state must be visually explicit with `app-tab--active` and `aria-current="page"`.

## Buttons
- Primary action:
  - outline accent style, filled accent on hover
- Secondary action:
  - neutral border, subtle hover fill
- Icon buttons:
  - subtle, accent, and danger variants
  - keyboard focus-visible ring required

## Forms
- Inputs, selects, and textareas:
  - dark panel backgrounds
  - bordered fields
  - strong visible focus border and shadow
- File upload controls are icon-triggered with a hidden native input where required.

## Overlays
- Modal and popover patterns:
  - centered card
  - backdrop with dark overlay or blur
  - explicit close action
  - keyboard-focus-compatible controls

## Feedback States
- Loading:
  - spinner plus concise status copy
- Error:
  - dismissible banner with action
- Empty:
  - placeholder component with optional detail text
- Disabled:
  - reduced opacity and non-interactive cursor

## Shared visual contracts
- The application shell uses a compact brand lockup and a visible accent underline for the active route.
- Dataset and tokenizer preview sections use aligned two-column cards with shared panel padding and surface treatment.
- Tokenizer report and vocabulary panels use CSS Grid stretch alignment; JavaScript height measurement is not required.
- Chart cards use centralized heights and legend spacing so adjacent plots keep consistent geometry.
- Dataset and tokenizer catalogs use the shared stateless `CatalogFilterToolbar`; search, source, and numeric filters are server-driven and must not alter benchmark selection.
