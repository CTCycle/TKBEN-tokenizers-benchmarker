import { DestroyRef, Injectable, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subject, catchError, debounceTime, forkJoin, of, switchMap } from 'rxjs';
import { BenchmarksApiService } from '../api/benchmarks-api.service';
import { DatasetsApiService } from '../api/datasets-api.service';
import { JobsApiService } from '../api/jobs-api.service';
import { TokenizersApiService } from '../api/tokenizers-api.service';
import { errorMessage } from '../api/error-utils';
import type {
  BenchmarkMetricCatalogCategory,
  BenchmarkReportListResponse,
  BenchmarkReportQuery,
  BenchmarkReportSort,
  BenchmarkReportSummary,
  BenchmarkRunRequest,
  BenchmarkRunResponse,
} from '../api/api.types';

@Injectable({ providedIn: 'root' })
export class BenchmarkStore {
  private readonly api = inject(BenchmarksApiService);
  private readonly datasetsApi = inject(DatasetsApiService);
  private readonly tokenizersApi = inject(TokenizersApiService);
  private readonly jobsApi = inject(JobsApiService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly reportRequests = new Subject<BenchmarkReportQuery>();
  private reportLoadSequence = 0;

  readonly reports = signal<readonly BenchmarkReportSummary[]>([]);
  readonly reportTotal = signal(0);
  readonly reportOffset = signal(0);
  readonly reportLimit = signal(25);
  readonly reportSearch = signal('');
  readonly reportSort = signal<BenchmarkReportSort>('newest');
  readonly reportsLoading = signal(true);
  readonly deletingReportId = signal<number | null>(null);
  readonly selectedReportId = signal<number | null>(null);
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
    this.reportRequests.pipe(
      debounceTime(250),
      switchMap((query) => {
        this.reportsLoading.set(true);
        this.error.set(null);
        return this.api.reports(query).pipe(
          catchError((error: unknown) => {
            this.error.set(errorMessage(error, 'Failed to fetch benchmark reports.'));
            return of<BenchmarkReportListResponse | null>(null);
          }),
        );
      }),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe((response) => {
      this.reportsLoading.set(false);
      if (!response) return;
      const reports = response.reports ?? [];
      this.reports.set(reports);
      this.reportTotal.set(response.total ?? 0);
      this.reportOffset.set(response.offset ?? 0);
      this.reportLimit.set(response.limit ?? 25);
      if (this.selectedReportId() === null && reports[0]) {
        this.selectReport(reports[0].report_id);
      }
    });
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
    this.requestReportPage(this.reportOffset());
  }

  setReportSearch(search: string): void {
    this.reportSearch.set(search);
    this.requestReportPage(0);
  }

  setReportSort(sort: BenchmarkReportSort): void {
    this.reportSort.set(sort);
    this.requestReportPage(0);
  }

  nextReportsPage(): void {
    if (this.reportOffset() + this.reportLimit() >= this.reportTotal()) return;
    this.requestReportPage(this.reportOffset() + this.reportLimit());
  }

  previousReportsPage(): void {
    if (this.reportOffset() <= 0) return;
    this.requestReportPage(Math.max(0, this.reportOffset() - this.reportLimit()));
  }

  selectReport(reportId: number): void {
    this.selectedReportId.set(reportId);
    this.loadReport(reportId);
  }

  loadReport(reportId: number): void {
    const sequence = ++this.reportLoadSequence;
    this.api.report(reportId).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (report) => {
        if (sequence !== this.reportLoadSequence || this.selectedReportId() !== reportId) return;
        this.report.set(report);
        if (!this.layout().length) {
          this.layout.set(report.dashboard.widgets.map((widget) => widget.widget_id));
          this.hiddenWidgetIds.set(report.dashboard.widgets.filter((widget) => !widget.default_visible).map((widget) => widget.widget_id));
        }
      },
      error: (error: unknown) => {
        if (sequence === this.reportLoadSequence) this.error.set(errorMessage(error, 'Failed to fetch benchmark report.'));
      },
    });
  }

  deleteReport(reportId: number): void {
    if (this.deletingReportId() !== null) return;
    this.deletingReportId.set(reportId);
    this.error.set(null);
    const visibleReports = [...this.reports()];
    const deletedIndex = visibleReports.findIndex((item) => item.report_id === reportId);
    const selected = this.selectedReportId() === reportId;
    this.api.deleteReport(reportId).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: () => {
        const remaining = visibleReports.filter((item) => item.report_id !== reportId);
        this.reports.set(remaining);
        this.reportTotal.update((total) => Math.max(0, total - 1));
        this.deletingReportId.set(null);

        let nextOffset = this.reportOffset();
        if (selected) {
          const fallback = deletedIndex >= 0
            ? visibleReports[deletedIndex + 1] ?? visibleReports[deletedIndex - 1]
            : undefined;
          this.clearDashboardReport();
          if (fallback) {
            this.selectedReportId.set(fallback.report_id);
            this.loadReport(fallback.report_id);
          } else {
            this.selectedReportId.set(null);
            if (nextOffset > 0) nextOffset = Math.max(0, nextOffset - this.reportLimit());
          }
        }
        this.requestReportPage(nextOffset);
      },
      error: (error: unknown) => {
        this.deletingReportId.set(null);
        this.error.set(errorMessage(error, 'Failed to delete benchmark report.'));
      },
    });
  }

  run(request: BenchmarkRunRequest): void {
    this.busy.set(true);
    this.progress.set(0);
    this.api.run(request, (status) => this.progress.set(status.progress), (job) => this.activeJobId.set(job.job_id)).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (report) => {
        this.report.set(report);
        if (report.report_id !== null) this.selectedReportId.set(report.report_id);
        this.busy.set(false);
        this.activeJobId.set(null);
        this.progress.set(100);
        this.refresh();
      },
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

  private requestReportPage(offset: number): void {
    this.reportOffset.set(Math.max(0, offset));
    this.reportRequests.next({
      search: this.reportSearch().trim() || undefined,
      sort: this.reportSort(),
      offset: Math.max(0, offset),
      limit: this.reportLimit(),
    });
  }

  private clearDashboardReport(): void {
    this.report.set(null);
    this.layout.set([]);
    this.hiddenWidgetIds.set([]);
  }
}
