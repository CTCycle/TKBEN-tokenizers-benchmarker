import { DestroyRef, Injectable, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { catchError, forkJoin, of } from 'rxjs';
import { BenchmarksApiService } from '../api/benchmarks-api.service';
import { DatasetsApiService } from '../api/datasets-api.service';
import { JobsApiService } from '../api/jobs-api.service';
import { TokenizersApiService } from '../api/tokenizers-api.service';
import { errorMessage } from '../api/error-utils';
import type { BenchmarkMetricCatalogCategory, BenchmarkReportSummary, BenchmarkRunRequest, BenchmarkRunResponse } from '../api/api.types';

@Injectable({ providedIn: 'root' })
export class BenchmarkStore {
  private readonly api = inject(BenchmarksApiService);
  private readonly datasetsApi = inject(DatasetsApiService);
  private readonly tokenizersApi = inject(TokenizersApiService);
  private readonly jobsApi = inject(JobsApiService);
  private readonly destroyRef = inject(DestroyRef);

  readonly reports = signal<readonly BenchmarkReportSummary[]>([]);
  readonly selectedReportId = signal<number | null>(null);
  readonly loading = signal(true);
  readonly error = signal<string | null>(null);
  readonly report = signal<BenchmarkRunResponse | null>(null);
  readonly metricCategories = signal<readonly BenchmarkMetricCatalogCategory[]>([]);
  readonly availableTokenizers = signal<readonly string[]>([]);
  readonly availableDatasets = signal<readonly string[]>([]);
  readonly loadingWorkspace = signal(true);
  readonly busy = signal(false);
  readonly progress = signal<number | null>(null);
  readonly activeJobId = signal<string | null>(null);
  readonly layout = signal<readonly string[]>([]);
  readonly hiddenWidgetIds = signal<readonly string[]>([]);
  private readonly layoutStorageKey = 'tkben:cross-benchmark-dashboard-layout:v3';

  constructor() {
    this.restoreLayout();
    this.loadWorkspaceMeta();
    this.refresh();
  }

  loadWorkspaceMeta(): void {
    this.loadingWorkspace.set(true);
    forkJoin({
      metrics: this.api.metricsCatalog().pipe(catchError(() => of({ categories: [] as BenchmarkMetricCatalogCategory[] }))),
      tokenizers: this.tokenizersApi.list().pipe(catchError(() => of({ tokenizers: [] }))),
      datasets: this.datasetsApi.list().pipe(catchError(() => of({ datasets: [] }))),
    }).pipe(takeUntilDestroyed(this.destroyRef)).subscribe(({ metrics, tokenizers, datasets }) => {
      this.metricCategories.set(metrics.categories ?? []);
      this.availableTokenizers.set((tokenizers.tokenizers ?? []).map((item) => item.tokenizer_name));
      this.availableDatasets.set((datasets.datasets ?? []).map((item) => item.dataset_name));
      this.loadingWorkspace.set(false);
    });
  }

  refresh(): void {
    this.loading.set(true);
    this.api.reports().pipe(
      catchError((error: unknown) => {
        this.error.set(errorMessage(error, 'Failed to fetch benchmark reports.'));
        return of({ reports: [] as readonly BenchmarkReportSummary[] });
      }),
    ).pipe(takeUntilDestroyed(this.destroyRef)).subscribe((response) => {
      const reports = response.reports ?? [];
      this.reports.set(reports);
      if (this.selectedReportId() === null) this.selectedReportId.set(reports[0]?.report_id ?? null);
      if (this.selectedReportId() !== null) this.loadReport(this.selectedReportId()!);
      this.loading.set(false);
    });
  }

  selectReport(reportId: number): void {
    this.selectedReportId.set(reportId);
    this.loadReport(reportId);
  }

  loadReport(reportId: number): void {
    this.api.report(reportId).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (report) => {
        this.report.set(report);
        if (!this.layout().length) {
          this.layout.set(report.dashboard.widgets.map((widget) => widget.widget_id));
          this.hiddenWidgetIds.set(report.dashboard.widgets.filter((widget) => !widget.default_visible).map((widget) => widget.widget_id));
        }
      },
      error: (error: unknown) => this.error.set(errorMessage(error, 'Failed to fetch benchmark report.')),
    });
  }

  run(request: BenchmarkRunRequest): void {
    this.busy.set(true);
    this.progress.set(0);
    this.api.run(request, (status) => this.progress.set(status.progress), (job) => this.activeJobId.set(job.job_id)).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (report) => { this.report.set(report); this.busy.set(false); this.activeJobId.set(null); this.progress.set(100); this.refresh(); },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to run benchmarks.')); this.busy.set(false); this.activeJobId.set(null); this.progress.set(null); },
    });
  }

  cancel(): void {
    const jobId = this.activeJobId();
    if (!jobId) return;
    this.jobsApi.cancel(jobId).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: () => { this.error.set('Benchmark cancellation requested.'); },
      error: (error: unknown) => this.error.set(errorMessage(error, 'Failed to cancel benchmark.')),
    });
  }

  reorder(from: number, to: number): void {
    const next = [...this.layout()];
    const [item] = next.splice(from, 1);
    if (item) next.splice(to, 0, item);
    this.layout.set(next);
    this.persistLayout(next);
  }

  resetLayout(): void {
    const widgets = this.report()?.dashboard.widgets ?? [];
    const defaults = widgets.map((widget) => widget.widget_id);
    this.layout.set(defaults);
    this.hiddenWidgetIds.set(widgets.filter((widget) => !widget.default_visible).map((widget) => widget.widget_id));
    this.persistLayout(defaults);
  }

  private restoreLayout(): void {
    try {
      const raw = localStorage.getItem(this.layoutStorageKey);
      const parsed: unknown = raw ? JSON.parse(raw) : null;
      if (Array.isArray(parsed) && parsed.every((item) => typeof item === 'string')) this.layout.set(parsed);
      else if (parsed && typeof parsed === 'object' && 'ordered_widget_ids' in parsed) {
        const order = (parsed as { ordered_widget_ids?: unknown }).ordered_widget_ids;
        if (Array.isArray(order) && order.every((item) => typeof item === 'string')) this.layout.set(order);
        const hidden = (parsed as { hidden_widget_ids?: unknown }).hidden_widget_ids;
        if (Array.isArray(hidden) && hidden.every((item) => typeof item === 'string')) this.hiddenWidgetIds.set(hidden);
      }
    } catch { /* corrupted storage is ignored */ }
  }

  private persistLayout(order: readonly string[]): void {
    try {
      const previous = JSON.parse(localStorage.getItem(this.layoutStorageKey) ?? 'null') as { visualization_by_widget_id?: unknown } | null;
      localStorage.setItem(this.layoutStorageKey, JSON.stringify({
        version: 3,
        ordered_widget_ids: order,
        hidden_widget_ids: this.hiddenWidgetIds(),
        known_widget_ids: order,
        visualization_by_widget_id: previous && previous.visualization_by_widget_id && typeof previous.visualization_by_widget_id === 'object' ? previous.visualization_by_widget_id : {},
      }));
    } catch { /* storage is optional */ }
  }

  setHiddenWidgetIds(hidden: readonly string[]): void {
    this.hiddenWidgetIds.set([...new Set(hidden)]);
    this.persistLayout(this.layout());
  }
}
