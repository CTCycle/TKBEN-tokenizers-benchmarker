import { useEffect, useMemo, useState } from 'react';
import type { BenchmarkDashboardData, BenchmarkDashboardWidgetData, BenchmarkVisualizationKind } from '../types/api';
import { BENCHMARK_DASHBOARD_STORAGE_KEY, BENCHMARK_DASHBOARD_STORAGE_KEY_V2, isDefaultDashboardLayout, resetDashboardLayout, resolveAvailableDashboardLayout, validateStoredDashboardLayout, type BenchmarkDashboardLayoutState, type ResolvedBenchmarkDashboardLayout } from '../features/benchmark-dashboard/benchmarkDashboardLayout';

export interface UseBenchmarkDashboardLayoutResult {
  widgets: BenchmarkDashboardWidgetData[];
  visibleWidgets: BenchmarkDashboardWidgetData[];
  resolved: ResolvedBenchmarkDashboardLayout;
  apply: (visibleIds: string[]) => void;
  reset: () => void;
  isDefault: boolean;
  setOrder: (orderedWidgetIds: string[]) => void;
  setVisualization: (widgetId: string, visualization: BenchmarkVisualizationKind) => void;
}

const readStored = (): BenchmarkDashboardLayoutState | null => { try { window.localStorage.removeItem(BENCHMARK_DASHBOARD_STORAGE_KEY_V2); return validateStoredDashboardLayout(JSON.parse(window.localStorage.getItem(BENCHMARK_DASHBOARD_STORAGE_KEY) ?? 'null')); } catch { return null; } };
export const useBenchmarkDashboardLayout = (dashboard: BenchmarkDashboardData | undefined): UseBenchmarkDashboardLayoutResult => {
  const widgets = useMemo(() => dashboard?.widgets ?? [], [dashboard]); const [layout, setLayout] = useState<BenchmarkDashboardLayoutState | null>(() => readStored());
  const resolved = useMemo(() => resolveAvailableDashboardLayout(layout, widgets), [layout, widgets]);
  useEffect(() => { if (layout) window.localStorage.setItem(BENCHMARK_DASHBOARD_STORAGE_KEY, JSON.stringify(layout)); }, [layout]);
  const visibleWidgets = resolved.visibleWidgetIds.map((id) => widgets.find((widget) => widget.widget_id === id)).filter((widget): widget is BenchmarkDashboardWidgetData => Boolean(widget));
  const apply = (visibleIds: string[]) => setLayout((current) => { const base = current ?? resetDashboardLayout(widgets); return { ...base, ordered_widget_ids: resolved.orderedWidgetIds, hidden_widget_ids: widgets.map((widget) => widget.widget_id).filter((id) => !visibleIds.includes(id)), known_widget_ids: [...new Set([...base.known_widget_ids, ...widgets.map((widget) => widget.widget_id)])], visualization_by_widget_id: resolved.visualizationByWidgetId }; });
  const reset = () => { window.localStorage.removeItem(BENCHMARK_DASHBOARD_STORAGE_KEY); setLayout(null); };
  const setVisualization = (widgetId: string, visualization: BenchmarkVisualizationKind) => setLayout((current) => ({ ...(current ?? resetDashboardLayout(widgets)), visualization_by_widget_id: { ...(current?.visualization_by_widget_id ?? resetDashboardLayout(widgets).visualization_by_widget_id), [widgetId]: visualization } }));
  return { widgets, visibleWidgets, resolved, apply, reset, isDefault: isDefaultDashboardLayout(layout, widgets), setOrder: (orderedWidgetIds: string[]) => setLayout((current) => ({ ...(current ?? resetDashboardLayout(widgets)), ordered_widget_ids: orderedWidgetIds })), setVisualization };
};
