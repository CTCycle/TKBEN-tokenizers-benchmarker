import { CdkDrag, CdkDragDrop, CdkDropList, moveItemInArray } from '@angular/cdk/drag-drop';
import { Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { BenchmarkStore } from '../core/state/benchmark.store';
import { BenchmarkMetricChartComponent } from '../components/benchmark-metric-chart.component';
import { ExportApiService } from '../core/api/export-api.service';
import type { BenchmarkDashboardWidgetData, BenchmarkVisualizationKind } from '../core/api/api.models';
import { errorMessageAsync } from '../core/api/error-utils';

@Component({
  selector: 'app-cross-benchmark-page',
  imports: [ReactiveFormsModule, CdkDropList, CdkDrag, BenchmarkMetricChartComponent],
  templateUrl: './cross-benchmark-page.component.html',
})
export class CrossBenchmarkPageComponent {
  protected readonly store = inject(BenchmarkStore);
  private readonly exportApi = inject(ExportApiService);
  private readonly destroyRef = inject(DestroyRef);
  protected readonly selectedReport = new FormControl<number | null>(null);
  protected readonly runOpen = signal(false);
  protected readonly runStep = signal<1 | 2 | 3>(1);
  protected readonly customizeOpen = signal(false);
  protected readonly tableOpen = signal<ReadonlySet<string>>(new Set());
  protected readonly visualizations = signal<Record<string, string>>({});
  protected readonly restoreDisabled = signal(false);
  protected readonly exportError = signal<string | null>(null);
  protected readonly keyboardGrabbed = signal<string | null>(null);
  protected readonly runForm = new FormGroup({
    dataset: new FormControl('', { nonNullable: true, validators: [Validators.required] }),
    tokenizers: new FormControl('', { nonNullable: true, validators: [Validators.required] }),
    runName: new FormControl('', { nonNullable: true }),
  });
  protected readonly orderedWidgets = computed(() => {
    const report = this.store.report();
    const layout = this.store.layout();
    if (!report) return [];
    const widgets = report.dashboard.widgets;
    return [...widgets].sort((a, b) => (layout.indexOf(a.widget_id) < 0 ? Number.MAX_SAFE_INTEGER : layout.indexOf(a.widget_id)) - (layout.indexOf(b.widget_id) < 0 ? Number.MAX_SAFE_INTEGER : layout.indexOf(b.widget_id)));
  });
  protected readonly failedResults = computed(() => (this.store.report()?.tokenizer_results ?? []).filter((result) => result.status === 'failed'));

  constructor() {
    try {
      const raw = localStorage.getItem('tkben:cross-benchmark-dashboard-layout:v3');
      const parsed: unknown = raw ? JSON.parse(raw) : null;
      if (parsed && typeof parsed === 'object' && 'visualization_by_widget_id' in parsed) {
        const map = (parsed as { visualization_by_widget_id?: unknown }).visualization_by_widget_id;
        if (map && typeof map === 'object') this.visualizations.set(map as Record<string, string>);
      }
    } catch { /* corrupted storage is ignored */ }
  }

  protected selectReport(value: string): void {
    const reportId = Number(value);
    if (Number.isFinite(reportId)) this.store.selectReport(reportId);
  }

  protected runBenchmark(): void {
    const value = this.runForm.getRawValue();
    const tokenizers = value.tokenizers.split(',').map((tokenizer) => tokenizer.trim()).filter(Boolean);
    if (!value.dataset.trim() || tokenizers.length === 0) return;
    this.store.run({ tokenizers, dataset_name: value.dataset.trim(), run_name: value.runName.trim() || null, selected_metric_keys: null, config: { warmup_trials: 1, timed_trials: 3, batch_size: 8, seed: 42, parallelism: 1, include_lm_metrics: false } });
    this.runOpen.set(false);
  }

  protected openRun(): void { this.runForm.patchValue({ dataset: this.store.report()?.dataset_name ?? '' }); this.runStep.set(1); this.runOpen.set(true); }
  protected nextStep(): void { if (this.runStep() < 3) this.runStep.update((step) => (step + 1) as 1 | 2 | 3); }
  protected previousStep(): void { if (this.runStep() > 1) this.runStep.update((step) => (step - 1) as 1 | 2 | 3); }

  protected exportDashboard(): void {
    const report = this.store.report();
    if (!report) return;
    this.exportError.set(null);
    this.exportApi.dashboardPdf({ dashboardType: 'benchmark', reportName: report.run_name || `benchmark-report-${report.report_id ?? 'latest'}`, fileName: `benchmark-report-${report.report_id ?? 'latest'}.pdf`, dashboardPayload: report as unknown as Record<string, unknown> }).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({ next: (result) => { const url = URL.createObjectURL(result.blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = result.fileName; anchor.click(); URL.revokeObjectURL(url); }, error: (error: unknown) => { void errorMessageAsync(error, 'Failed to export dashboard.').then((message) => this.exportError.set(message)); } });
  }

  protected dropWidget(event: CdkDragDrop<unknown[]>): void {
    const ids = this.orderedWidgets().map((widget) => widget.widget_id);
    moveItemInArray(ids, event.previousIndex, event.currentIndex);
    const from = this.store.layout().indexOf(ids[event.currentIndex]);
    if (from >= 0) this.store.reorder(from, event.currentIndex);
  }

  protected moveWidget(index: number, direction: -1 | 1): void {
    const target = index + direction;
    if (target < 0 || target >= this.orderedWidgets().length) return;
    const ids = this.orderedWidgets().map((widget) => widget.widget_id);
    const fromId = ids[index];
    const toId = ids[target];
    const from = this.store.layout().indexOf(fromId);
    const to = this.store.layout().indexOf(toId);
    if (from >= 0 && to >= 0) this.store.reorder(from, to);
  }

  protected onWidgetKeydown(event: KeyboardEvent, index: number): void {
    const widget = this.orderedWidgets()[index];
    if (!widget) return;
    if (event.key === ' ' || event.key === 'Enter') {
      event.preventDefault();
      this.keyboardGrabbed.update((current) => current === widget.widget_id ? null : widget.widget_id);
      return;
    }
    if (this.keyboardGrabbed() !== widget.widget_id) return;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      event.preventDefault();
      this.moveWidget(index, -1);
    } else if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      event.preventDefault();
      this.moveWidget(index, 1);
    }
  }

  protected toggleTable(widgetId: string): void {
    const next = new Set(this.tableOpen());
    if (next.has(widgetId)) next.delete(widgetId); else next.add(widgetId);
    this.tableOpen.set(next);
  }

  protected setVisualization(widgetId: string, value: string): void {
    const next = { ...this.visualizations(), [widgetId]: value };
    this.visualizations.set(next);
    try { localStorage.setItem('tkben:cross-benchmark-dashboard-layout:v3', JSON.stringify({ version: 3, ordered_widget_ids: this.store.layout(), hidden_widget_ids: [], known_widget_ids: this.store.layout(), visualization_by_widget_id: next })); } catch { /* storage is optional */ }
  }

  protected isVisualization(widget: BenchmarkDashboardWidgetData, value: string): boolean {
    const stored = this.visualizations()[widget.widget_id];
    const selected = stored && widget.compatible_visualizations.includes(stored as BenchmarkVisualizationKind) ? stored as BenchmarkVisualizationKind : widget.default_visualization;
    return selected === value;
  }

  protected restoreDefaults(): void {
    this.customizeOpen.set(false);
    this.restoreDisabled.set(true);
    this.store.resetLayout();
    this.visualizations.set({});
    try { localStorage.removeItem('tkben:cross-benchmark-dashboard-layout:v3'); } catch { /* storage is optional */ }
  }
}
