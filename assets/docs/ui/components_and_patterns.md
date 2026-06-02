# Components and Patterns
Last updated: 2026-06-02

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
