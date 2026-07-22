import type { BenchmarkDashboardWidgetData } from '../../types/api';

export const BENCHMARK_DASHBOARD_STORAGE_KEY = 'tkben:cross-benchmark-dashboard-layout:v1';
export type BenchmarkDashboardLayoutState = { version: 1; ordered_widget_ids: string[]; hidden_widget_ids: string[]; known_widget_ids: string[]; };
export type ResolvedBenchmarkDashboardLayout = { orderedWidgetIds: string[]; visibleWidgetIds: string[]; };

const unique = (values: string[]) => [...new Set(values.filter(Boolean))];
export const validateStoredDashboardLayout = (value: unknown): BenchmarkDashboardLayoutState | null => {
  if (!value || typeof value !== 'object') return null;
  const candidate = value as Partial<BenchmarkDashboardLayoutState>;
  if (candidate.version !== 1 || !Array.isArray(candidate.ordered_widget_ids) || !Array.isArray(candidate.hidden_widget_ids) || !Array.isArray(candidate.known_widget_ids) || ![candidate.ordered_widget_ids, candidate.hidden_widget_ids, candidate.known_widget_ids].every((list) => list.every((id) => typeof id === 'string'))) return null;
  return { version: 1, ordered_widget_ids: unique(candidate.ordered_widget_ids), hidden_widget_ids: unique(candidate.hidden_widget_ids), known_widget_ids: unique(candidate.known_widget_ids) };
};
export const resetDashboardLayout = (widgets: BenchmarkDashboardWidgetData[]): BenchmarkDashboardLayoutState => ({ version: 1, ordered_widget_ids: widgets.map((widget) => widget.widget_id), hidden_widget_ids: widgets.filter((widget) => !widget.default_visible).map((widget) => widget.widget_id), known_widget_ids: widgets.map((widget) => widget.widget_id) });
export const isDefaultDashboardLayout = (layout: BenchmarkDashboardLayoutState | null, widgets: BenchmarkDashboardWidgetData[]): boolean => {
  const defaults = resetDashboardLayout(widgets);
  const current = layout ?? defaults;
  return current.ordered_widget_ids.join('|') === defaults.ordered_widget_ids.join('|')
    && current.hidden_widget_ids.join('|') === defaults.hidden_widget_ids.join('|');
};
export const resolveAvailableDashboardLayout = (stored: BenchmarkDashboardLayoutState | null, widgets: BenchmarkDashboardWidgetData[]): ResolvedBenchmarkDashboardLayout => {
  const base = stored ?? resetDashboardLayout(widgets); const available = new Set(widgets.map((widget) => widget.widget_id));
  const fresh = widgets.filter((widget) => !base.known_widget_ids.includes(widget.widget_id));
  const orderedWidgetIds = unique([...base.ordered_widget_ids.filter((id) => available.has(id)), ...fresh.map((widget) => widget.widget_id)]);
  const hidden = new Set(base.hidden_widget_ids); fresh.forEach((widget) => { if (!widget.default_visible) hidden.add(widget.widget_id); });
  return { orderedWidgetIds, visibleWidgetIds: orderedWidgetIds.filter((id) => !hidden.has(id)) };
};
export const serializeDashboardLayout = (layout: BenchmarkDashboardLayoutState): string => JSON.stringify(layout);
export const moveDashboardWidget = (order: string[], activeId: string, overId: string | null): string[] => {
  if (!overId || activeId === overId || !order.includes(activeId) || !order.includes(overId)) return order;
  const next = order.filter((id) => id !== activeId); next.splice(next.indexOf(overId), 0, activeId); return next;
};
export const swapDashboardWidgets = (order: string[], first: string, second: string): string[] => { const next = [...order]; const a = next.indexOf(first); const b = next.indexOf(second); if (a < 0 || b < 0) return order; [next[a], next[b]] = [next[b], next[a]]; return next; };
export const insertDashboardWidget = moveDashboardWidget;
export const packDashboardGrid = (widgets: BenchmarkDashboardWidgetData[]): BenchmarkDashboardWidgetData[][] => { const rows: BenchmarkDashboardWidgetData[][] = []; let row: BenchmarkDashboardWidgetData[] = []; widgets.forEach((widget) => { if (widget.width === 'wide') { if (row.length) rows.push(row); rows.push([widget]); row = []; } else { row.push(widget); if (row.length === 2) { rows.push(row); row = []; } } }); if (row.length) rows.push(row); return rows; };
