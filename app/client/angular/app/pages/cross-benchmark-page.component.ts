import { CdkDrag, CdkDragDrop, CdkDropList } from '@angular/cdk/drag-drop';
import { DecimalPipe } from '@angular/common';
import { Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { BenchmarkStore } from '../core/state/benchmark.store';
import { SettingsStore } from '../core/state/settings.store';
import { BenchmarkMetricChartComponent } from '../components/benchmark-metric-chart.component';
import { ExportApiService } from '../core/api/export-api.service';
import type {
  BenchmarkDashboardWidgetData,
  BenchmarkMetricCatalogCategory,
  BenchmarkReportSort,
  BenchmarkReportSummary,
  BenchmarkVisualizationKind,
} from '../core/api/api.models';
import { errorMessageAsync } from '../core/api/error-utils';
import {
  benchmarkComparisonRows,
  benchmarkComparisonTokenizers,
  classifyBenchmarkDataShape,
  formatBenchmarkDelta,
  formatBenchmarkValue,
} from '../core/utils/benchmark-dashboard-data';
import { ModalA11yDirective } from '../core/ui/modal-a11y.directive';

@Component({
  selector: 'app-cross-benchmark-page',
  imports: [ReactiveFormsModule, DecimalPipe, CdkDropList, CdkDrag, BenchmarkMetricChartComponent, ModalA11yDirective],
  templateUrl: './cross-benchmark-page.component.html',
})
export class CrossBenchmarkPageComponent {
  protected readonly store = inject(BenchmarkStore);
  private readonly settingsStore = inject(SettingsStore);
  private readonly exportApi = inject(ExportApiService);
  private readonly destroyRef = inject(DestroyRef);
  protected readonly runOpen = signal(false);
  protected readonly runStep = signal<1 | 2 | 3>(1);
  protected readonly runMode = signal<'new' | 'clone'>('new');
  protected readonly customizeOpen = signal(false);
  protected readonly reportManagerOpen = signal(false);
  protected readonly reportDeleteConfirmId = signal<number | null>(null);
  protected readonly editingTagsReportId = signal<number | null>(null);
  protected readonly tagDraft = signal('');
  protected readonly customizeDraft = signal<readonly string[]>([]);
  protected readonly runSelectedMetricKeys = signal<readonly string[]>([]);
  protected readonly runSelectedTokenizers = signal<readonly string[]>([]);
  protected readonly tokenizerQuery = signal('');
  protected readonly restoreDisabled = signal(false);
  protected readonly exportError = signal<string | null>(null);
  protected readonly cloneError = signal<string | null>(null);
  protected readonly keyboardGrabbed = signal<string | null>(null);
  protected readonly runForm = new FormGroup({
    dataset: new FormControl('', { nonNullable: true, validators: [Validators.required] }),
    tokenizers: new FormControl('', { nonNullable: true, validators: [Validators.required] }),
    runName: new FormControl('', { nonNullable: true, validators: [Validators.maxLength(120)] }),
    maxDocuments: new FormControl(1000, { nonNullable: true, validators: [Validators.min(0), Validators.max(100000)] }),
    warmupTrials: new FormControl(2, { nonNullable: true, validators: [Validators.min(0), Validators.max(100)] }),
    timedTrials: new FormControl(8, { nonNullable: true, validators: [Validators.min(1), Validators.max(200)] }),
    batchSize: new FormControl(16, { nonNullable: true, validators: [Validators.min(1), Validators.max(4096)] }),
    seed: new FormControl(42, { nonNullable: true }),
    parallelism: new FormControl(1, { nonNullable: true, validators: [Validators.min(1), Validators.max(128)] }),
    addSpecialTokens: new FormControl(false, { nonNullable: true }),
    padding: new FormControl(false, { nonNullable: true }),
    truncation: new FormControl(false, { nonNullable: true }),
    maxLength: new FormControl<number | null>(null, { validators: [Validators.min(1)] }),
    storePerDocumentStats: new FormControl(true, { nonNullable: true }),
    perDocumentSampleSize: new FormControl(500, { nonNullable: true, validators: [Validators.min(1), Validators.max(10000)] }),
  });
  protected readonly allOrderedWidgets = computed(() => {
    const report = this.store.report();
    const layout = this.store.layout();
    if (!report) return [];
    const widgets = report.dashboard.widgets;
    return [...widgets].sort((a, b) => (layout.indexOf(a.widget_id) < 0 ? Number.MAX_SAFE_INTEGER : layout.indexOf(a.widget_id)) - (layout.indexOf(b.widget_id) < 0 ? Number.MAX_SAFE_INTEGER : layout.indexOf(b.widget_id)));
  });
  protected readonly orderedWidgets = computed(() => this.allOrderedWidgets().filter((widget) => !this.store.hiddenWidgetIds().includes(widget.widget_id)));
  protected readonly failedResults = computed(() => (this.store.report()?.tokenizer_results ?? []).filter((result) => result.status === 'failed'));
  protected readonly filteredRunTokenizers = computed(() => {
    const query = this.tokenizerQuery().trim().toLowerCase();
    return this.store.availableTokenizers().filter((tokenizer) => !query || tokenizer.toLowerCase().includes(query));
  });
  protected readonly comparisonTokenizers = computed(() => {
    const report = this.store.report();
    return report ? benchmarkComparisonTokenizers(report.dashboard) : [];
  });

  protected openReportManager(): void { this.reportManagerOpen.set(true); }
  protected closeReportManager(): void { this.reportManagerOpen.set(false); this.reportDeleteConfirmId.set(null); }
  protected selectReportSummary(report: BenchmarkReportSummary): void {
    this.store.selectReport(report.report_id);
    this.closeReportManager();
  }
  protected reportTitle(report: BenchmarkReportSummary): string {
    return report.run_name?.trim() || `Benchmark run #${report.report_id}`;
  }
  protected reportDate(report: BenchmarkReportSummary): string {
    return this.formatReportDate(report.created_at);
  }
  protected formatReportDate(value: string | null | undefined): string {
    if (!value) return 'Unknown date';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? 'Unknown date' : date.toLocaleString();
  }
  protected reportRangeStart(): number {
    return this.store.reportTotal() > 0 ? this.store.reportOffset() + 1 : 0;
  }
  protected reportRangeEnd(): number {
    return Math.min(this.store.reportOffset() + this.store.reports().length, this.store.reportTotal());
  }
  protected updateReportSearch(event: Event): void {
    this.store.setReportSearch((event.target as HTMLInputElement).value);
  }
  protected updateReportSort(event: Event): void {
    this.store.setReportSort((event.target as HTMLSelectElement).value as BenchmarkReportSort);
  }
  protected askToDeleteReport(reportId: number): void { this.reportDeleteConfirmId.set(reportId); }
  protected cancelDeleteReport(): void { this.reportDeleteConfirmId.set(null); }
  protected deleteReport(reportId: number): void { this.store.deleteReport(reportId); }

  protected startTagEdit(report: BenchmarkReportSummary): void {
    this.editingTagsReportId.set(report.report_id);
    this.tagDraft.set((report.tags ?? []).join(', '));
  }

  protected cancelTagEdit(): void {
    this.editingTagsReportId.set(null);
    this.tagDraft.set('');
  }

  protected updateTagDraft(event: Event): void {
    this.tagDraft.set((event.target as HTMLInputElement).value);
  }

  protected saveReportTags(reportId: number): void {
    if (this.store.updatingReportTagsId() === reportId) return;
    const tags = this.tagDraft().split(',').map((tag) => tag.trim()).filter(Boolean);
    this.store.updateReportTags(reportId, tags);
    this.cancelTagEdit();
  }

  protected reportTags(report: BenchmarkReportSummary | { tags?: readonly string[] }): readonly string[] {
    return report.tags ?? [];
  }

  protected selectBaseline(event: Event): void {
    const tokenizer = (event.target as HTMLSelectElement).value;
    this.store.setBaseline(tokenizer || null);
  }

  protected openCloneRun(): void {
    const report = this.store.report();
    if (!report) return;
    const eligibilityError = this.cloneEligibilityError(report);
    if (eligibilityError) {
      this.cloneError.set(eligibilityError);
      return;
    }
    const config = report.config;
    this.runMode.set('clone');
    this.cloneError.set(null);
    this.runForm.patchValue({
      dataset: report.dataset_name,
      tokenizers: report.tokenizers_processed.join(','),
      runName: this.cloneRunName(report),
      maxDocuments: config.max_documents ?? report.documents_processed,
      warmupTrials: config.warmup_trials,
      timedTrials: config.timed_trials,
      batchSize: config.batch_size,
      seed: config.seed,
      parallelism: config.parallelism,
      addSpecialTokens: config.add_special_tokens,
      padding: config.padding,
      truncation: config.truncation,
      maxLength: config.max_length ?? null,
      storePerDocumentStats: config.store_per_document_stats,
      perDocumentSampleSize: config.per_document_sample_size,
    });
    this.runSelectedMetricKeys.set([...report.selected_metric_keys]);
    this.runSelectedTokenizers.set([...report.tokenizers_processed]);
    this.tokenizerQuery.set('');
    this.runStep.set(3);
    this.runOpen.set(true);
  }

  private cloneEligibilityError(report: { dataset_name: string; tokenizers_processed: string[]; selected_metric_keys: string[] }): string | null {
    if (!this.store.availableDatasets().includes(report.dataset_name)) {
      return `Cannot clone this benchmark because dataset “${report.dataset_name}” is no longer available.`;
    }
    if (report.tokenizers_processed.length > 5) {
      return 'Cannot clone this benchmark because it contains more than 5 tokenizers.';
    }
    const missingTokenizer = report.tokenizers_processed.find((tokenizer) => !this.store.availableTokenizers().includes(tokenizer));
    if (missingTokenizer) return `Cannot clone this benchmark because tokenizer “${missingTokenizer}” is no longer available.`;
    const metricKeys = new Set(this.store.metricCategories().flatMap((category) => category.metrics.map((metric) => metric.key)));
    const missingMetric = report.selected_metric_keys.find((metric) => !metricKeys.has(metric));
    if (missingMetric) return `Cannot clone this benchmark because metric “${missingMetric}” is no longer available.`;
    return null;
  }

  private cloneRunName(report: { report_id: number | null; run_name: string | null }): string {
    const source = report.run_name?.trim() || `benchmark ${report.report_id ?? 'unknown'}`;
    return Array.from(`Clone of ${source}`).slice(0, 120).join('');
  }

  protected runBenchmark(): void {
    const value = this.runForm.getRawValue();
    const tokenizers = [...this.runSelectedTokenizers()];
    if (this.runForm.controls.runName.invalid || this.runForm.controls.maxLength.invalid || !value.dataset.trim() || tokenizers.length === 0 || !value.runName.trim() || this.runSelectedMetricKeys().length === 0) return;
    this.store.run({ tokenizers, dataset_name: value.dataset.trim(), run_name: value.runName.trim(), selected_metric_keys: [...this.runSelectedMetricKeys()], config: { max_documents: value.maxDocuments, warmup_trials: value.warmupTrials, timed_trials: value.timedTrials, batch_size: value.batchSize, seed: value.seed, parallelism: value.parallelism, add_special_tokens: value.addSpecialTokens, padding: value.padding, truncation: value.truncation, max_length: value.maxLength, store_per_document_stats: value.storePerDocumentStats, per_document_sample_size: value.perDocumentSampleSize } });
    this.runOpen.set(false);
  }

  protected openRun(): void {
    this.initializeNewRun();
  }

  private initializeNewRun(): void {
    const dataset = this.store.report()?.dataset_name ?? this.store.availableDatasets()[0] ?? '';
    const benchmarkDefaults = this.settingsStore.settings()?.benchmarks;
    this.runMode.set('new');
    this.cloneError.set(null);
    this.runForm.patchValue({
      dataset,
      tokenizers: '',
      runName: '',
      maxDocuments: benchmarkDefaults?.default_max_documents ?? 1000,
      warmupTrials: 2,
      timedTrials: 8,
      batchSize: benchmarkDefaults?.default_batch_size ?? 16,
      seed: 42,
      parallelism: benchmarkDefaults?.default_parallelism ?? 1,
      addSpecialTokens: false,
      padding: false,
      truncation: false,
      maxLength: null,
      storePerDocumentStats: true,
      perDocumentSampleSize: 500,
    });
    this.runSelectedMetricKeys.set(this.store.metricCategories().flatMap((category) => category.metrics.map((metric) => metric.key)));
    // The dataset may be preselected, but tokenizers are intentionally left
    // empty so the user explicitly chooses the benchmark inputs.
    this.runSelectedTokenizers.set([]);
    this.runForm.controls.tokenizers.setValue('');
    this.tokenizerQuery.set('');
    this.runStep.set(1);
    this.runOpen.set(true);
  }
  protected nextStep(): void {
    if (this.runStep() === 1 && this.runSelectedMetricKeys().length === 0) return;
    if (this.runStep() === 2 && (this.runSelectedTokenizers().length === 0 || !this.runForm.controls.dataset.value.trim())) return;
    if (this.runStep() < 3) this.runStep.update((step) => (step + 1) as 1 | 2 | 3);
  }
  protected previousStep(): void { if (this.runStep() > 1) this.runStep.update((step) => (step - 1) as 1 | 2 | 3); }

  protected cancelBenchmark(): void {
    if (this.store.busy()) this.store.cancel(); else this.runOpen.set(false);
  }

  protected openCustomize(): void {
    this.customizeDraft.set(this.orderedWidgets().map((widget) => widget.widget_id));
    this.restoreDisabled.set(false);
    this.customizeOpen.set(true);
  }

  protected toggleCustomizeWidget(widgetId: string, enabled: boolean): void {
    const next = new Set(this.customizeDraft());
    if (enabled) next.add(widgetId); else next.delete(widgetId);
    this.customizeDraft.set([...next]);
  }

  protected toggleCustomizeCategory(category: string, enabled: boolean): void {
    const ids = this.allOrderedWidgets().filter((widget) => widget.category_label === category).map((widget) => widget.widget_id);
    const next = new Set(this.customizeDraft());
    ids.forEach((id) => enabled ? next.add(id) : next.delete(id));
    this.customizeDraft.set([...next]);
  }

  protected customizeCategorySelected(category: string): boolean {
    const ids = this.allOrderedWidgets().filter((widget) => widget.category_label === category).map((widget) => widget.widget_id);
    return ids.length > 0 && ids.every((id) => this.customizeDraft().includes(id));
  }

  protected applyCustomize(): void {
    const visible = new Set(this.customizeDraft());
    this.store.setHiddenWidgetIds(this.allOrderedWidgets().filter((widget) => !visible.has(widget.widget_id)).map((widget) => widget.widget_id));
    this.customizeOpen.set(false);
  }

  protected runMetricCategorySelected(category: BenchmarkMetricCatalogCategory): boolean {
    return category.metrics.length > 0 && category.metrics.every((metric) => this.runSelectedMetricKeys().includes(metric.key));
  }

  protected toggleRunMetric(metricKey: string, enabled: boolean): void {
    const next = new Set(this.runSelectedMetricKeys());
    if (enabled) next.add(metricKey); else next.delete(metricKey);
    this.runSelectedMetricKeys.set([...next]);
  }

  protected toggleRunMetricCategory(category: BenchmarkMetricCatalogCategory, enabled: boolean): void {
    const next = new Set(this.runSelectedMetricKeys());
    category.metrics.forEach((metric) => enabled ? next.add(metric.key) : next.delete(metric.key));
    this.runSelectedMetricKeys.set([...next]);
  }

  protected toggleRunTokenizer(tokenizer: string, enabled: boolean): void {
    const next = new Set(this.runSelectedTokenizers());
    if (enabled && next.size < 5) next.add(tokenizer); else if (!enabled) next.delete(tokenizer);
    this.runSelectedTokenizers.set([...next]);
    this.runForm.controls.tokenizers.setValue([...next].join(','));
  }

  protected customizeCategories(): string[] { return [...new Set(this.allOrderedWidgets().map((widget) => widget.category_label))]; }

  protected exportDashboard(): void {
    const report = this.store.report();
    if (!report) return;
    this.exportError.set(null);
    const orderedWidgetIds = this.allOrderedWidgets().map((widget) => widget.widget_id);
    const visibleWidgets = this.orderedWidgets();
    const visualizationByWidgetId = Object.fromEntries(
      visibleWidgets.map((widget) => [widget.widget_id, this.visualizationFor(widget)]),
    );
    const dashboardPayload = {
      ...report,
      visible_widget_ids: visibleWidgets.map((widget) => widget.widget_id),
      ordered_widget_ids: orderedWidgetIds,
      visualization_by_widget_id: visualizationByWidgetId,
    } as unknown as Record<string, unknown>;
    this.exportApi.dashboardPdf({ dashboardType: 'benchmark', reportName: report.run_name || `benchmark-report-${report.report_id ?? 'latest'}`, fileName: `benchmark-report-${report.report_id ?? 'latest'}.pdf`, dashboardPayload }).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({ next: (result) => { const url = URL.createObjectURL(result.blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = result.fileName; anchor.click(); URL.revokeObjectURL(url); }, error: (error: unknown) => { void errorMessageAsync(error, 'Failed to export dashboard.').then((message) => this.exportError.set(message)); } });
  }

  protected dropWidget(event: CdkDragDrop<unknown[]>): void {
    this.store.reorderVisible(event.previousIndex, event.currentIndex);
  }

  protected moveWidget(index: number, direction: -1 | 1): void {
    const target = index + direction;
    if (target < 0 || target >= this.orderedWidgets().length) return;
    this.store.reorderVisible(index, target);
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

  protected setVisualization(widgetId: string, value: string): void {
    this.store.setVisualization(widgetId, value);
  }

  protected isVisualization(widget: BenchmarkDashboardWidgetData, value: string): boolean {
    const stored = this.store.visualizations()[widget.widget_id];
    const selected = stored && widget.compatible_visualizations.includes(stored as BenchmarkVisualizationKind) ? stored as BenchmarkVisualizationKind : widget.default_visualization;
    return selected === value;
  }

  protected visualizationFor(widget: BenchmarkDashboardWidgetData): BenchmarkVisualizationKind {
    const stored = this.store.visualizations()[widget.widget_id];
    return stored && widget.compatible_visualizations.includes(stored as BenchmarkVisualizationKind)
      ? stored as BenchmarkVisualizationKind
      : widget.default_visualization;
  }

  protected dataShape(widget: BenchmarkDashboardWidgetData): string {
    return classifyBenchmarkDataShape(widget);
  }

  protected comparisonRows(widget: BenchmarkDashboardWidgetData): ReturnType<typeof benchmarkComparisonRows> {
    return benchmarkComparisonRows(widget, this.store.baselineTokenizer());
  }

  protected comparisonDelta(widget: BenchmarkDashboardWidgetData, tokenizer: string): string {
    const row = this.comparisonRows(widget).find((item) => item.tokenizer === tokenizer);
    return row?.formattedDelta ?? 'N/A';
  }

  protected hasComparisonData(widget: BenchmarkDashboardWidgetData): boolean {
    return Boolean(widget.points?.length || widget.distributions?.length);
  }

  protected comparisonStripRows(widget: BenchmarkDashboardWidgetData): ReturnType<typeof benchmarkComparisonRows> {
    const rows = this.comparisonRows(widget);
    return rows.some((row) => !row.isBaseline && row.deltaPercent !== null) ? rows.filter((row) => !row.isBaseline) : [];
  }

  protected readonly formatBenchmarkDelta = formatBenchmarkDelta;

  protected readonly formatBenchmarkValue = formatBenchmarkValue;

  protected restoreDefaults(): void {
    this.customizeOpen.set(false);
    this.restoreDisabled.set(true);
    this.store.resetLayout();
  }
}
