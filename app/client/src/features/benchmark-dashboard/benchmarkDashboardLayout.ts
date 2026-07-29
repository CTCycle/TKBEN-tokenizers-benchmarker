import type { BenchmarkDashboardWidgetData, BenchmarkVisualizationKind } from '../../types/api';

export const BENCHMARK_DASHBOARD_STORAGE_KEY = 'tkben:cross-benchmark-dashboard-layout:v3';
export const BENCHMARK_DASHBOARD_STORAGE_KEY_V2 = 'tkben:cross-benchmark-dashboard-layout:v2';
export type BenchmarkDashboardLayoutState = { version: 3; ordered_widget_ids: string[]; hidden_widget_ids: string[]; known_widget_ids: string[]; visualization_by_widget_id: Record<string, BenchmarkVisualizationKind>; };
export type ResolvedBenchmarkDashboardLayout = { orderedWidgetIds: string[]; visibleWidgetIds: string[]; visualizationByWidgetId: Record<string, BenchmarkVisualizationKind>; };

const VISUALIZATIONS: BenchmarkVisualizationKind[] = ['bar', 'horizontal_bar', 'interval_bar', 'dot_whisker', 'box_plot', 'histogram', 'grouped_bar', 'heatmap'];

const unique = (values: string[]) => [...new Set(values.filter(Boolean))];
export const validateStoredDashboardLayout = (value: unknown): BenchmarkDashboardLayoutState | null => {
  if (!value || typeof value !== 'object') return null;
  const candidate = value as Partial<BenchmarkDashboardLayoutState>;
  if (candidate.version !== 3 || !Array.isArray(candidate.ordered_widget_ids) || !Array.isArray(candidate.hidden_widget_ids) || !Array.isArray(candidate.known_widget_ids) || !candidate.visualization_by_widget_id || typeof candidate.visualization_by_widget_id !== 'object' || ![candidate.ordered_widget_ids, candidate.hidden_widget_ids, candidate.known_widget_ids].every((list) => list.every((id) => typeof id === 'string'))) return null;
  const visualizations = Object.fromEntries(Object.entries(candidate.visualization_by_widget_id).filter(([, value]) => typeof value === 'string' && VISUALIZATIONS.includes(value as BenchmarkVisualizationKind))) as Record<string, BenchmarkVisualizationKind>;
  return { version: 3, ordered_widget_ids: unique(candidate.ordered_widget_ids), hidden_widget_ids: unique(candidate.hidden_widget_ids), known_widget_ids: unique(candidate.known_widget_ids), visualization_by_widget_id: visualizations };
};
export const resetDashboardLayout = (widgets: BenchmarkDashboardWidgetData[]): BenchmarkDashboardLayoutState => ({ version: 3, ordered_widget_ids: widgets.map((widget) => widget.widget_id), hidden_widget_ids: widgets.filter((widget) => !widget.default_visible).map((widget) => widget.widget_id), known_widget_ids: widgets.map((widget) => widget.widget_id), visualization_by_widget_id: Object.fromEntries(widgets.map((widget) => [widget.widget_id, widget.default_visualization])) as Record<string, BenchmarkVisualizationKind> });
export const isDefaultDashboardLayout = (layout: BenchmarkDashboardLayoutState | null, widgets: BenchmarkDashboardWidgetData[]): boolean => {
  const defaults = resetDashboardLayout(widgets);
  const current = layout ?? defaults;
  return current.ordered_widget_ids.join('|') === defaults.ordered_widget_ids.join('|')
    && current.hidden_widget_ids.join('|') === defaults.hidden_widget_ids.join('|')
    && widgets.every((widget) => current.visualization_by_widget_id[widget.widget_id] === widget.default_visualization);
};
export const resolveAvailableDashboardLayout = (stored: BenchmarkDashboardLayoutState | null, widgets: BenchmarkDashboardWidgetData[]): ResolvedBenchmarkDashboardLayout => {
  const base = stored ?? resetDashboardLayout(widgets); const available = new Set(widgets.map((widget) => widget.widget_id));
  const fresh = widgets.filter((widget) => !base.known_widget_ids.includes(widget.widget_id));
  const orderedWidgetIds = unique([...base.ordered_widget_ids.filter((id) => available.has(id)), ...fresh.map((widget) => widget.widget_id)]);
  const hidden = new Set(base.hidden_widget_ids); fresh.forEach((widget) => { if (!widget.default_visible) hidden.add(widget.widget_id); });
  const visualizationByWidgetId = Object.fromEntries(widgets.map((widget) => {
    const candidate = base.visualization_by_widget_id[widget.widget_id];
    return [widget.widget_id, widget.compatible_visualizations.includes(candidate) ? candidate : widget.default_visualization];
  })) as Record<string, BenchmarkVisualizationKind>;
  return { orderedWidgetIds, visibleWidgetIds: orderedWidgetIds.filter((id) => !hidden.has(id)), visualizationByWidgetId };
};
export const serializeDashboardLayout = (layout: BenchmarkDashboardLayoutState): string => JSON.stringify(layout);
export const moveDashboardWidget = (order: string[], activeId: string, overId: string | null): string[] => {
  if (!overId || activeId === overId || !order.includes(activeId) || !order.includes(overId)) return order;
  const next = order.filter((id) => id !== activeId); next.splice(next.indexOf(overId), 0, activeId); return next;
};
export const swapDashboardWidgets = (order: string[], first: string, second: string): string[] => { const next = [...order]; const a = next.indexOf(first); const b = next.indexOf(second); if (a < 0 || b < 0) return order; [next[a], next[b]] = [next[b], next[a]]; return next; };
export const insertDashboardWidget = moveDashboardWidget;
export const packDashboardGrid = (widgets: BenchmarkDashboardWidgetData[]): BenchmarkDashboardWidgetData[][] => { const rows: BenchmarkDashboardWidgetData[][] = []; let row: BenchmarkDashboardWidgetData[] = []; widgets.forEach((widget) => { if (widget.width === 'wide') { if (row.length) rows.push(row); rows.push([widget]); row = []; } else { row.push(widget); if (row.length === 2) { rows.push(row); row = []; } } }); if (row.length) rows.push(row); return rows; };
