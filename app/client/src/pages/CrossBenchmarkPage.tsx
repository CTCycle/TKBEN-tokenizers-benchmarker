import { DndContext, PointerSensor, closestCenter, useSensor, useSensors, type DragEndEvent } from '@dnd-kit/core';
import { SortableContext, rectSortingStrategy, useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { useRef, useState, type KeyboardEvent, type RefObject } from 'react';
import BenchmarkRunWizard from '../components/BenchmarkRunWizard';
import { BenchmarkMetricWidget } from '../components/benchmark-dashboard/BenchmarkMetricWidget';
import DashboardExportButton from '../components/DashboardExportButton';
import DismissibleBanner from '../components/DismissibleBanner';
import { moveDashboardWidget } from '../features/benchmark-dashboard/benchmarkDashboardLayout';
import { useBenchmarkDashboardLayout } from '../hooks/useBenchmarkDashboardLayout';
import { useBodyScrollLock } from '../hooks/useBodyScrollLock';
import { useBenchmarkWorkspace } from '../hooks/useBenchmarkWorkspace';
import type { BenchmarkDashboardWidgetData, BenchmarkRunWizardPayload } from '../types/api';

type SortableWidgetProps = {
  widget: BenchmarkDashboardWidgetData;
  visualization: BenchmarkDashboardWidgetData['default_visualization'];
  onVisualizationChange: (visualization: BenchmarkDashboardWidgetData['default_visualization']) => void;
  onKeyboardReorder: (event: KeyboardEvent<HTMLButtonElement>, widgetId: string) => void;
};

const SortableWidget = ({ widget, visualization, onVisualizationChange, onKeyboardReorder }: SortableWidgetProps) => {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id: widget.widget_id });
  return <div ref={setNodeRef} className="benchmark-dashboard-sortable" style={{ transform: CSS.Transform.toString(transform), transition }} data-dragging={isDragging || undefined}>
    <button type="button" className="benchmark-dashboard-drag-handle" {...attributes} {...listeners} aria-label={`Reorder ${widget.label} widget`} title="Drag to reorder" onKeyDown={(event) => onKeyboardReorder(event, widget.widget_id)}><span aria-hidden="true">⋮⋮</span></button>
    <BenchmarkMetricWidget widget={widget} visualization={visualization} onVisualizationChange={onVisualizationChange} />
  </div>;
};

type CustomizerProps = {
  widgets: BenchmarkDashboardWidgetData[];
  visibleIds: string[];
  onApply: (ids: string[]) => void;
  onClose: () => void;
  trigger: RefObject<HTMLButtonElement | null>;
};

const Customizer = ({ widgets, visibleIds, onApply, onClose, trigger }: CustomizerProps) => {
  const [draft, setDraft] = useState(visibleIds);
  const groups = widgets.reduce<Record<string, BenchmarkDashboardWidgetData[]>>((result, widget) => {
    (result[widget.category_label] ??= []).push(widget);
    return result;
  }, {});
  const close = () => {
    onClose();
    trigger.current?.focus();
  };

  useBodyScrollLock(true);

  return (
    <div className="benchmark-dashboard-modal-backdrop" role="presentation" onMouseDown={close}>
      <section
        className="benchmark-dashboard-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="dashboard-customizer-title"
        onMouseDown={(event) => event.stopPropagation()}
        onKeyDown={(event) => {
          if (event.key === 'Escape') close();
        }}
      >
        <h2 id="dashboard-customizer-title">Customize benchmark dashboard</h2>
        <p>{draft.length} of {widgets.length} available widgets selected.</p>
        {Object.entries(groups).map(([category, entries]) => (
          <fieldset key={category}>
            <legend>{category}</legend>
            <div className="benchmark-dashboard-section-actions">
              <button
                type="button"
                className="icon-button subtle benchmark-dashboard-section-action"
                aria-label={`Select all ${category} metrics`}
                title="Select all"
                onClick={() => setDraft((current) => [...new Set([...current, ...entries.map((item) => item.widget_id)])])}
              >
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 4 4L19 6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </button>
              <button
                type="button"
                className="icon-button subtle benchmark-dashboard-section-action"
                aria-label={`Clear ${category} metrics`}
                title="Clear"
                onClick={() => setDraft((current) => current.filter((id) => !entries.some((item) => item.widget_id === id)))}
              >
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
              </button>
            </div>
            {entries.map((widget) => (
              <label key={widget.widget_id}>
                <input
                  type="checkbox"
                  checked={draft.includes(widget.widget_id)}
                  onChange={() => setDraft((current) => current.includes(widget.widget_id) ? current.filter((id) => id !== widget.widget_id) : [...current, widget.widget_id])}
                />
                {widget.label} — {widget.description} ({widget.unit}, {widget.default_visualization})
              </label>
            ))}
          </fieldset>
        ))}
        <footer className="modal-wizard-footer">
          <button type="button" className="secondary-button" onClick={() => setDraft(widgets.filter((item) => item.default_visible).map((item) => item.widget_id))}>Reset to defaults</button>
          <button type="button" className="secondary-button" onClick={close}>Cancel</button>
          <button type="button" className="primary-button" disabled={!draft.length} onClick={() => { onApply(draft); close(); }}>Apply</button>
        </footer>
      </section>
    </div>
  );
};

const CrossBenchmarkPage = () => {
  const workspace = useBenchmarkWorkspace();
  const [wizardOpen, setWizardOpen] = useState(false);
  const [customizing, setCustomizing] = useState(false);
  const [keyboardDragId, setKeyboardDragId] = useState<string | null>(null);
  const customizeButton = useRef<HTMLButtonElement>(null);
  const layout = useBenchmarkDashboardLayout(workspace.activeReport?.dashboard);
  const failed = workspace.activeReport?.tokenizer_results.filter((item) => item.status === 'failed') ?? [];
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 8 } }));

  const handleDragEnd = ({ active, over }: DragEndEvent) => {
    const next = moveDashboardWidget(layout.resolved.orderedWidgetIds, String(active.id), over ? String(over.id) : null);
    if (next !== layout.resolved.orderedWidgetIds) layout.setOrder(next);
  };

  const handleKeyboardReorder = (event: KeyboardEvent<HTMLButtonElement>, widgetId: string) => {
    if (event.key === ' ' || event.code === 'Space') {
      event.preventDefault();
      setKeyboardDragId((current) => current === widgetId ? null : widgetId);
      return;
    }
    if (keyboardDragId !== widgetId || !['ArrowLeft', 'ArrowUp', 'ArrowRight', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const currentIndex = layout.resolved.orderedWidgetIds.indexOf(widgetId);
    const direction = event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1;
    const targetIndex = currentIndex + direction;
    if (targetIndex >= 0 && targetIndex < layout.resolved.orderedWidgetIds.length) {
      const next = [...layout.resolved.orderedWidgetIds];
      const [moved] = next.splice(currentIndex, 1);
      next.splice(targetIndex, 0, moved);
      layout.setOrder(next);
    }
  };

  const payload = workspace.activeReport ? { report: workspace.activeReport, visible_widget_ids: layout.resolved.visibleWidgetIds, ordered_widget_ids: layout.resolved.orderedWidgetIds, visualization_by_widget_id: layout.resolved.visualizationByWidgetId } : null;

  return (
    <section className="page-content cross-benchmark-page">
      {workspace.error && <DismissibleBanner message={workspace.error} onDismiss={workspace.clearError} />}
      {workspace.loadingPage ? <p>Loading benchmark workspace…</p> : (
        <div className="cross-benchmark-workspace-shell">
          <header className="cross-benchmark-control-surface" aria-label="Cross benchmark controls and report overview">
            <div className="cross-benchmark-header-main">
              {workspace.activeReport &&
                <div className="cross-benchmark-top-row">
                  <div className="cross-benchmark-report-picker">
                    <label className="field-label panel-label" htmlFor="cross-benchmark-report-select">Select report</label>
                    <select id="cross-benchmark-report-select" value={workspace.selectedReportId ?? ''} onChange={(event) => { const id = Number(event.target.value); if (id) void workspace.loadReportById(id); }}>
                      <option value="">Select a report</option>
                      {workspace.reports.map((report) => <option key={report.report_id} value={report.report_id}>{report.run_name || report.dataset_name}</option>)}
                    </select>
                  </div>
                  <div className="cross-benchmark-overview-grid" aria-label="Benchmark summary">
                    <article className="cross-benchmark-kpi-card">
                      <span className="cross-benchmark-kpi-label">Dataset</span>
                      <strong className="cross-benchmark-kpi-value">{workspace.activeReport.dataset_name}</strong>
                      <small className="cross-benchmark-kpi-detail">Selected benchmark corpus</small>
                    </article>
                    <article className="cross-benchmark-kpi-card">
                      <span className="cross-benchmark-kpi-label">Documents</span>
                      <strong className="cross-benchmark-kpi-value">{workspace.activeReport.documents_processed.toLocaleString()}</strong>
                      <small className="cross-benchmark-kpi-detail">Processed documents</small>
                    </article>
                    <article className="cross-benchmark-kpi-card">
                      <span className="cross-benchmark-kpi-label">Tokenizers</span>
                      <strong className="cross-benchmark-kpi-value">{workspace.activeReport.tokenizers_count}</strong>
                      <small className="cross-benchmark-kpi-detail">Compared tokenizers</small>
                    </article>
                  </div>
                </div>
              }
            </div>
            <nav id="cross-benchmark-command-navbar" className="cross-benchmark-command-navbar" aria-label="Benchmark controls">
              <div className="cross-benchmark-command-navbar__header">
                <span className="cross-benchmark-command-navbar__title">Benchmark actions</span>
              </div>
              <div id="cross-benchmark-command-navbar-content" className="cross-benchmark-command-navbar__content" role="group" aria-label="Benchmark actions">
                <button ref={customizeButton} type="button" className="cross-benchmark-action-button" disabled={!workspace.activeReport || workspace.loadingReport} title="Customize benchmark dashboard" aria-label="Customize benchmark dashboard" onClick={() => setCustomizing(true)}>
                  <svg viewBox="0 0 24 24" aria-hidden="true" width="16" height="16"><path d="M12 15a3 3 0 100-6 3 3 0 000 6z" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
                  <span>Customize</span>
                </button>
                <button type="button" className="cross-benchmark-action-button dashboard-layout-reset" aria-label="Restore default layout" title="Restore default layout" disabled={!workspace.activeReport || workspace.loadingReport || layout.isDefault} onClick={layout.reset}>
                  <svg viewBox="0 0 24 24" aria-hidden="true" width="16" height="16"><path d="M1 4v6h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M3.51 15a9 9 0 102.13-9.36L1 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                  <span>Reset</span>
                </button>
                <DashboardExportButton dashboardType="benchmark" reportName={workspace.activeReport?.run_name ?? 'benchmark-dashboard'} dashboardPayload={payload} label="Export" />
                <button type="button" className="cross-benchmark-run-button" aria-label="Run benchmark" title="Run benchmark" disabled={workspace.loadingPage || workspace.loadingReport} onClick={() => setWizardOpen(true)}>
                  <svg viewBox="0 0 24 24" aria-hidden="true" width="16" height="16"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor"/></svg>
                  <span>Run</span>
                </button>
              </div>
            </nav>
          </header>
          <main className="cross-benchmark-workspace-main">
            {workspace.activeReport && <>
              <div className="cross-benchmark-kpi-grid">
                {layout.visibleWidgets.length ? <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}><SortableContext items={layout.visibleWidgets.map((widget) => widget.widget_id)} strategy={rectSortingStrategy}><div className="benchmark-dashboard-grid" aria-label="Benchmark metric widgets. Use the reorder buttons to drag a widget or use Space then arrow keys to reorder it.">{layout.visibleWidgets.map((widget) => <SortableWidget key={widget.widget_id} widget={widget} visualization={layout.resolved.visualizationByWidgetId[widget.widget_id]} onVisualizationChange={(visualization) => layout.setVisualization(widget.widget_id, visualization)} onKeyboardReorder={handleKeyboardReorder} />)}</div></SortableContext></DndContext> : <p className="empty-state">No metric widgets are selected. Customize the dashboard to show one or more calculated metrics.</p>}
              </div>
              <p className="sr-only" aria-live="polite">{keyboardDragId ? 'Keyboard widget reordering active. Use arrow keys to move the widget, then Space to finish.' : ''}</p>
              <article className="cross-benchmark-drilldown-card"><h2>Run diagnostics</h2>{failed.length ? failed.map((item) => <p key={item.tokenizer}>{item.tokenizer}: {item.error_type ?? 'Failed'} — {item.error_message ?? 'No message'}</p>) : <p>No tokenizer failures recorded for this run.</p>}</article>
            </>}
          </main>
        </div>
      )}
      {wizardOpen && <BenchmarkRunWizard isOpen={wizardOpen} categories={workspace.metricCategories} availableTokenizers={workspace.tokenizers} availableDatasets={workspace.datasets} defaultDatasetName={workspace.datasets[0] ?? null} defaultMaxDocuments={1000} running={workspace.runningBenchmark} onCancel={workspace.cancelBenchmark} onClose={() => setWizardOpen(false)} onRun={async (runPayload: BenchmarkRunWizardPayload) => { if (await workspace.runFromWizard(runPayload)) setWizardOpen(false); }} />}
      {customizing && <Customizer widgets={layout.widgets} visibleIds={layout.resolved.visibleWidgetIds} onApply={layout.apply} onClose={() => setCustomizing(false)} trigger={customizeButton} />}
    </section>
  );
};

export default CrossBenchmarkPage;
