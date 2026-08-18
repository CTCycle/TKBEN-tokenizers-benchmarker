# Components and Patterns
Last updated: 2026-08-18

## Navigation
- Top tab navigation for:
  - `/dataset`
  - `/tokenizers`
  - `/cross-benchmark`
- Active state must be visually explicit with `app-tab--active` and `aria-current="page"`.
- The route tabs are integrated into the branded header beside the `tkben-logo.png` mark; the header also exposes the Hugging Face key manager as an icon button.

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
  - shared `ModalCloseButton` and `useBodyScrollLock` behavior where the modal owns viewport interaction

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
- The Tokenizer Manager keeps manual input/upload beside a compact Hugging Face discovery form. Primary search, result limit, text-task category, and sort are visible; author, access, required/excluded tags, vocabulary comparison, and vocabulary ordering stay inside a collapsed advanced section. Upload JSON is a flat workflow area with a compact file chooser rather than a nested card. Discovery results remain structured and selections store only repository identifiers.
- The Dataset Manager uses focused tabs for Predefined, Add by name, and Custom dataset workflows. The predefined catalogue remains the default tab; Hugging Face name/configuration entry and CSV/Excel upload are isolated panels so the grouped catalogue stays readable. Switching tabs preserves the entered form values and uses the existing dataset download/upload store actions.
- Cross Benchmark presents the current report summary and an integrated Reports action inside one compact outer header surface. KPI summaries and Benchmark Actions are flattened into the header rhythm without inner cards. The report manager uses server-side search, newest/oldest sorting, 25-row pagination, row-level selection, and inline danger confirmation before physical deletion; its modal remains unchanged. Filtering the catalogue does not unload the currently displayed dashboard report.
- Catalog filter changes are debounced, and filtered catalogs need distinct loading, no-match, and no-data messaging.
- Tokenizer preview rows expose source and vocabulary metadata with independent report and benchmark-selection actions; custom tokenizers cannot open generated reports.
- `CatalogFilterToolbar` is a generic typed control shared by dataset and tokenizer pages; it accepts only configured source and numeric-operator values from its option lists.
- Dataset dashboard charts consume normalized helper output rather than parsing unknown API payloads inside each chart component. Benchmark widgets keep chart rendering, data-shape classification, and the accessible data table as separate responsibilities.
