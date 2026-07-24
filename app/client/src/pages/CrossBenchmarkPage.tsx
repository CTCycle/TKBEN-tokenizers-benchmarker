import { DndContext, KeyboardSensor, PointerSensor, closestCenter, useSensor, useSensors, type DragEndEvent } from '@dnd-kit/core';
import { SortableContext, rectSortingStrategy, sortableKeyboardCoordinates, useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { useRef, useState, type KeyboardEvent, type RefObject } from 'react';
import BenchmarkRunWizard from '../components/BenchmarkRunWizard';
import { BenchmarkMetricWidget } from '../components/benchmark-dashboard/BenchmarkMetricWidget';
import DashboardExportButton from '../components/DashboardExportButton';
import DismissibleBanner from '../components/DismissibleBanner';
import { moveDashboardWidget } from '../features/benchmark-dashboard/benchmarkDashboardLayout';
import { useBenchmarkDashboardLayout } from '../hooks/useBenchmarkDashboardLayout';
import { useBenchmarkWorkspace, type BenchmarkRunPayload } from '../hooks/useBenchmarkWorkspace';
import type { BenchmarkDashboardWidgetData } from '../types/api';

const SortableWidget = ({ widget, onKeyboardReorder }: { widget: BenchmarkDashboardWidgetData; onKeyboardReorder: (event: KeyboardEvent<HTMLDivElement>, widgetId: string) => void }) => {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id: widget.widget_id });
  return <div ref={setNodeRef} className="benchmark-dashboard-sortable" style={{ transform: CSS.Transform.toString(transform), transition }} {...attributes} {...listeners} data-dragging={isDragging || undefined} onKeyDown={(event) => onKeyboardReorder(event, widget.widget_id)}><BenchmarkMetricWidget widget={widget} /></div>;
};

const Customizer = ({ widgets, visibleIds, onApply, onClose, trigger }: { widgets: BenchmarkDashboardWidgetData[]; visibleIds: string[]; onApply: (ids: string[]) => void; onClose: () => void; trigger: RefObject<HTMLButtonElement | null> }) => { const [draft, setDraft] = useState(visibleIds); const groups = widgets.reduce<Record<string, BenchmarkDashboardWidgetData[]>>((result, widget) => { (result[widget.category_label] ??= []).push(widget); return result; }, {}); const close = () => { onClose(); trigger.current?.focus(); }; return <div className="benchmark-dashboard-modal-backdrop" role="presentation" onMouseDown={close}><section className="benchmark-dashboard-modal" role="dialog" aria-modal="true" aria-labelledby="dashboard-customizer-title" onMouseDown={(event) => event.stopPropagation()} onKeyDown={(event) => { if (event.key === 'Escape') close(); }}><h2 id="dashboard-customizer-title">Customize benchmark dashboard</h2><p>{draft.length} of {widgets.length} available widgets selected.</p>{Object.entries(groups).map(([category, entries]) => <fieldset key={category}><legend>{category}</legend><button type="button" onClick={() => setDraft((current) => [...new Set([...current, ...entries.map((item) => item.widget_id)])])}>Select all</button><button type="button" onClick={() => setDraft((current) => current.filter((id) => !entries.some((item) => item.widget_id === id)))}>Clear</button>{entries.map((widget) => <label key={widget.widget_id}><input type="checkbox" checked={draft.includes(widget.widget_id)} onChange={() => setDraft((current) => current.includes(widget.widget_id) ? current.filter((id) => id !== widget.widget_id) : [...current, widget.widget_id])} />{widget.label} — {widget.description} ({widget.unit}, {widget.visualization})</label>)}</fieldset>)}<footer><button type="button" onClick={() => setDraft(widgets.filter((item) => item.default_visible).map((item) => item.widget_id))}>Reset to defaults</button><button type="button" onClick={close}>Cancel</button><button type="button" className="primary-button" disabled={!draft.length} onClick={() => { onApply(draft); close(); }}>Apply</button></footer></section></div> };

const CrossBenchmarkPage = () => {
  const workspace = useBenchmarkWorkspace();
  const [wizardOpen, setWizardOpen] = useState(false);
  const [customizing, setCustomizing] = useState(false);
  const [controlsOpen, setControlsOpen] = useState(true);
  const [keyboardDragId, setKeyboardDragId] = useState<string | null>(null);
  const customizeButton = useRef<HTMLButtonElement>(null);
  const layout = useBenchmarkDashboardLayout(workspace.activeReport?.dashboard);
  const failed = workspace.activeReport?.tokenizer_results.filter((item) => item.status === 'failed') ?? [];
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 8 } }), useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }));

  const handleDragEnd = ({ active, over }: DragEndEvent) => {
    const next = moveDashboardWidget(layout.resolved.orderedWidgetIds, String(active.id), over ? String(over.id) : null);
    if (next !== layout.resolved.orderedWidgetIds) layout.setOrder(next);
  };

  const handleKeyboardReorder = (event: KeyboardEvent<HTMLDivElement>, widgetId: string) => {
    if (event.key === ' ' || event.code === 'Space') { event.preventDefault(); event.stopPropagation(); setKeyboardDragId((current) => current === widgetId ? null : widgetId); return; }
    if (keyboardDragId !== widgetId || !['ArrowLeft', 'ArrowUp', 'ArrowRight', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault(); event.stopPropagation();
    const currentIndex = layout.resolved.orderedWidgetIds.indexOf(widgetId); const direction = event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1; const targetIndex = currentIndex + direction;
    if (targetIndex >= 0 && targetIndex < layout.resolved.orderedWidgetIds.length) { const next = [...layout.resolved.orderedWidgetIds]; const [moved] = next.splice(currentIndex, 1); next.splice(targetIndex, 0, moved); layout.setOrder(next); }
  };

  const payload = workspace.activeReport ? { report: workspace.activeReport, visible_widget_ids: layout.resolved.visibleWidgetIds, ordered_widget_ids: layout.resolved.orderedWidgetIds } : null;

  return (
    <section className="page-content cross-benchmark-page">
      {workspace.error && <DismissibleBanner message={workspace.error} onDismiss={workspace.clearError} />}
      {workspace.loadingPage ? <p>Loading benchmark workspace…</p> : (
        <div className="cross-benchmark-workspace-shell">
          <aside id="cross-benchmark-command-navbar" className={`cross-benchmark-command-navbar${controlsOpen ? '' : ' is-collapsed'}`} aria-label="Benchmark controls">
            <button type="button" className="icon-button subtle cross-benchmark-controls-toggle" aria-expanded={controlsOpen} aria-controls="cross-benchmark-command-navbar-content" aria-label={controlsOpen ? 'Collapse benchmark controls' : 'Expand benchmark controls'} onClick={() => setControlsOpen((current) => !current)}>☰</button>
            {controlsOpen && <div id="cross-benchmark-command-navbar-content" className="cross-benchmark-command-navbar__content">
              <div className="cross-benchmark-command-navbar__title">Benchmark controls</div>
              <div className="cross-benchmark-command-control"><button ref={customizeButton} type="button" className="icon-button subtle" disabled={!workspace.activeReport || workspace.loadingReport} title="Customize benchmark dashboard" aria-label="Customize benchmark dashboard" onClick={() => setCustomizing(true)}>⚙</button><span>Customize</span></div>
              <div className="cross-benchmark-command-control"><button type="button" className="secondary-button dashboard-layout-reset" aria-label="Restore default layout" title="Restore default layout" disabled={!workspace.activeReport || workspace.loadingReport || layout.isDefault} onClick={layout.reset}>Restore default layout</button><span>Reset</span></div>
              <div className="cross-benchmark-command-control"><DashboardExportButton dashboardType="benchmark" reportName={workspace.activeReport?.run_name ?? 'benchmark-dashboard'} dashboardPayload={payload} /><span>Export</span></div>
              <div className="cross-benchmark-command-control"><button type="button" className="primary-button" aria-label="Run benchmark" title="Run benchmark" onClick={() => setWizardOpen(true)}>Run benchmark</button><span>Run</span></div>
            </div>}
          </aside>
          <div className="cross-benchmark-header-main">
            {workspace.activeReport &&
              <div className="cross-benchmark-top-row">
                <div className="cross-benchmark-report-picker">
                  <label className="field-label" htmlFor="cross-benchmark-report-select">Select report</label>
                  <select id="cross-benchmark-report-select" value={workspace.selectedReportId ?? ''} onChange={(event) => { const id = Number(event.target.value); if (id) void workspace.loadReportById(id); }}>
                    <option value="">Select a report</option>
                    {workspace.reports.map((report) => <option key={report.report_id} value={report.report_id}>{report.run_name || report.dataset_name}</option>)}
                  </select>
                </div>
                <div className="cross-benchmark-overview-grid" aria-label="Benchmark summary">
                  <article className="cross-benchmark-kpi-card"><span>Dataset</span><strong>{workspace.activeReport.dataset_name}</strong><small>Selected benchmark corpus</small></article>
                  <article className="cross-benchmark-kpi-card"><span>Documents</span><strong>{workspace.activeReport.documents_processed.toLocaleString()}</strong><small>Processed documents</small></article>
                  <article className="cross-benchmark-kpi-card"><span>Tokenizers</span><strong>{workspace.activeReport.tokenizers_count}</strong><small>Compared tokenizers</small></article>
                </div>
              </div>
            }
          </div>
          <main className="cross-benchmark-workspace-main">
            {workspace.activeReport && <>
              <div className="cross-benchmark-kpi-grid">
                {layout.visibleWidgets.length ? <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}><SortableContext items={layout.visibleWidgets.map((widget) => widget.widget_id)} strategy={rectSortingStrategy}><div className="benchmark-dashboard-grid" aria-label="Benchmark metric widgets. Drag a widget or use Space then arrow keys to reorder it.">{layout.visibleWidgets.map((widget) => <SortableWidget key={widget.widget_id} widget={widget} onKeyboardReorder={handleKeyboardReorder} />)}</div></SortableContext></DndContext> : <p className="empty-state">No metric widgets are selected. Customize the dashboard to show one or more calculated metrics.</p>}
              </div>
              <p className="sr-only" aria-live="polite">{keyboardDragId ? 'Keyboard widget reordering active. Use arrow keys to move the widget, then Space to finish.' : ''}</p>
              <article className="cross-benchmark-drilldown-card"><h2>Run diagnostics</h2>{failed.length ? failed.map((item) => <p key={item.tokenizer}>{item.tokenizer}: {item.error_type ?? 'Failed'} — {item.error_message ?? 'No message'}</p>) : <p>No tokenizer failures recorded for this run.</p>}</article>
            </>}
          </main>
        </div>
      )}
      {wizardOpen && <BenchmarkRunWizard isOpen={wizardOpen} categories={workspace.metricCategories} availableTokenizers={workspace.tokenizers} availableDatasets={workspace.datasets} defaultDatasetName={workspace.datasets[0] ?? null} defaultMaxDocuments={1000} running={workspace.runningBenchmark} onCancel={workspace.cancelBenchmark} onClose={() => setWizardOpen(false)} onRun={async (runPayload: BenchmarkRunPayload) => { await workspace.runFromWizard(runPayload); }} />}
      {customizing && <Customizer widgets={layout.widgets} visibleIds={layout.resolved.visibleWidgetIds} onApply={layout.apply} onClose={() => setCustomizing(false)} trigger={customizeButton} />}
    </section>
  );
};

export default CrossBenchmarkPage;
