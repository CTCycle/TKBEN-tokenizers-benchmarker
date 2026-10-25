// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { DestroyRef, Injectable, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subject, catchError, debounceTime, forkJoin, of, switchMap } from 'rxjs';
import { BenchmarksApiService } from '../api/benchmarks-api.service';
import { DatasetsApiService } from '../api/datasets-api.service';
import { TokenizersApiService } from '../api/tokenizers-api.service';
import { errorMessage } from '../api/error-utils';
import type {
  BenchmarkMetricCatalogCategory,
  BenchmarkVisualizationKind,
  DashboardUiPreferences,
  BenchmarkReportListResponse,
  BenchmarkReportQuery,
  BenchmarkReportSort,
  BenchmarkReportSummary,
  BenchmarkRunRequest,
  BenchmarkRunResponse,
} from '../api/api.models';
import { benchmarkComparisonTokenizers, benchmarkTokenizerColors } from '../utils/benchmark-dashboard-data';

@Injectable({ providedIn: 'root' })
export class BenchmarkStore {
  private readonly api = inject(BenchmarksApiService);
  private readonly datasetsApi = inject(DatasetsApiService);
  private readonly tokenizersApi = inject(TokenizersApiService);
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
  readonly layout = signal<readonly string[]>([]);
  readonly hiddenWidgetIds = signal<readonly string[]>([]);
  readonly visualizations = signal<Record<string, BenchmarkVisualizationKind>>({});
  readonly baselineTokenizer = signal<string | null>(null);
  readonly tokenizerColorOverrides = signal<Record<string, string>>({});
  readonly tokenizerColors = computed(() => {
    const report = this.report();
    const overrides = this.tokenizerColorOverrides();
    return report ? benchmarkTokenizerColors(report.dashboard, overrides) : overrides;
  });
  readonly preferencesSaving = signal(false);
  readonly preferencesSaved = signal(false);
  readonly preferencesError = signal<string | null>(null);
  readonly updatingReportTagsId = signal<number | null>(null);
  private baselinePreferences: Record<string, string> = {};
  private dashboardPreferencesRevision = 0;
  private dashboardPreferencesLoaded = false;
  private dashboardPreferencesWriting = false;
  private queuedDashboardPreferences: DashboardUiPreferences | null = null;

  constructor() {
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
    this.loadDashboardPreferences();
  }

  private loadDashboardPreferences(): void {
    this.api.dashboardPreferences().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (response) => {
        this.dashboardPreferencesRevision = response.revision;
        this.applyDashboardPreferences(response.preferences);
        this.dashboardPreferencesLoaded = true;
        this.refresh();
      },
      error: (error: unknown) => {
        this.dashboardPreferencesLoaded = true;
        this.preferencesError.set(errorMessage(
          error,
          'Failed to load dashboard preferences.',
        ));
        this.refresh();
      },
    });
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
    this.baselineTokenizer.set(null);
    this.loadReport(reportId);
  }

  loadReport(reportId: number): void {
    const sequence = ++this.reportLoadSequence;
    this.api.report(reportId).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (report) => {
        if (sequence !== this.reportLoadSequence || this.selectedReportId() !== reportId) return;
        this.report.set(report);
        this.restoreBaselineForReport(report);
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
        this.removeBaselinePreference(reportId);
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
    this.error.set(null);
    this.api.run(request).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (report) => {
        ++this.reportLoadSequence;
        this.report.set(report);
        if (report.report_id !== null) this.selectedReportId.set(report.report_id);
        this.restoreBaselineForReport(report);
        this.refresh();
      },
      error: (error: unknown) => this.error.set(errorMessage(error, 'Failed to run benchmarks.')),
    });
  }

  reorder(from: number, to: number): void {
    const next = [...this.layout()];
    const [item] = next.splice(from, 1);
    if (item) next.splice(to, 0, item);
    this.layout.set(next);
    this.persistDashboardPreferences();
  }

  reorderVisible(from: number, to: number): void {
    const next = [...this.layout()];
    const hidden = new Set(this.hiddenWidgetIds());
    const visiblePositions = next.flatMap((widgetId, index) => hidden.has(widgetId) ? [] : [index]);
    if (from < 0 || from >= visiblePositions.length || to < 0 || to >= visiblePositions.length || from === to) return;

    const visibleIds = visiblePositions.map((position) => next[position]);
    const [item] = visibleIds.splice(from, 1);
    if (!item) return;
    visibleIds.splice(to, 0, item);
    visiblePositions.forEach((position, index) => next[position] = visibleIds[index]);
    this.layout.set(next);
    this.persistDashboardPreferences();
  }

  resetLayout(): void {
    const widgets = this.report()?.dashboard.widgets ?? [];
    const defaults = widgets.map((widget) => widget.widget_id);
    this.layout.set(defaults);
    this.hiddenWidgetIds.set(widgets.filter((widget) => !widget.default_visible).map((widget) => widget.widget_id));
    this.visualizations.set({});
    this.persistDashboardPreferences();
  }

  private applyDashboardPreferences(preferences: DashboardUiPreferences): void {
    this.layout.set([...preferences.ordered_widget_ids]);
    this.hiddenWidgetIds.set([...preferences.hidden_widget_ids]);
    this.visualizations.set({ ...preferences.visualization_by_widget_id });
    this.baselinePreferences = { ...preferences.baseline_by_report_id };
    this.tokenizerColorOverrides.set({ ...preferences.tokenizer_colors });
  }

  private restoreBaselineForReport(report: BenchmarkRunResponse): void {
    const reportId = report.report_id;
    if (reportId === null) {
      this.baselineTokenizer.set(null);
      return;
    }
    const stored = this.baselinePreferences[String(reportId)] ?? null;
    if (stored && benchmarkComparisonTokenizers(report.dashboard).includes(stored)) {
      this.baselineTokenizer.set(stored);
      return;
    }
    this.baselineTokenizer.set(null);
    if (stored) {
      delete this.baselinePreferences[String(reportId)];
      this.persistDashboardPreferences();
    }
  }

  private removeBaselinePreference(reportId: number): void {
    delete this.baselinePreferences[String(reportId)];
    this.persistDashboardPreferences();
    if (this.selectedReportId() === reportId) this.baselineTokenizer.set(null);
  }

  private dashboardPreferencesSnapshot(): DashboardUiPreferences {
    return {
      ordered_widget_ids: [...this.layout()],
      hidden_widget_ids: [...this.hiddenWidgetIds()],
      visualization_by_widget_id: { ...this.visualizations() },
      baseline_by_report_id: { ...this.baselinePreferences },
      tokenizer_colors: { ...this.tokenizerColorOverrides() },
    };
  }

  private persistDashboardPreferences(): void {
    if (!this.dashboardPreferencesLoaded) return;
    this.queuedDashboardPreferences = this.dashboardPreferencesSnapshot();
    this.preferencesError.set(null);
    this.preferencesSaved.set(false);
    this.preferencesSaving.set(true);
    this.flushDashboardPreferences();
  }

  private flushDashboardPreferences(): void {
    if (this.dashboardPreferencesWriting || !this.queuedDashboardPreferences) return;
    const preferences = this.queuedDashboardPreferences;
    this.queuedDashboardPreferences = null;
    this.dashboardPreferencesWriting = true;
    this.api.saveDashboardPreferences({
      expected_revision: this.dashboardPreferencesRevision,
      preferences,
    }).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (response) => {
        this.dashboardPreferencesRevision = response.revision;
        this.dashboardPreferencesWriting = false;
        if (this.queuedDashboardPreferences) {
          this.flushDashboardPreferences();
        } else {
          this.preferencesSaving.set(false);
          this.preferencesSaved.set(true);
        }
      },
      error: (error: unknown) => {
        this.dashboardPreferencesWriting = false;
        this.preferencesSaving.set(false);
        this.preferencesSaved.set(false);
        this.preferencesError.set(errorMessage(
          error,
          'Failed to save dashboard preferences.',
        ));
      },
    });
  }

  setHiddenWidgetIds(hidden: readonly string[]): void {
    this.hiddenWidgetIds.set([...new Set(hidden)]);
    this.persistDashboardPreferences();
  }

  setVisualization(widgetId: string, visualization: BenchmarkVisualizationKind): void {
    this.visualizations.update((current) => ({ ...current, [widgetId]: visualization }));
    this.persistDashboardPreferences();
  }

  setTokenizerColor(tokenizer: string, color: string | null): void {
    this.tokenizerColorOverrides.update((current) => {
      const next = { ...current };
      if (color) next[tokenizer] = color;
      else delete next[tokenizer];
      return next;
    });
    this.persistDashboardPreferences();
  }

  setBaseline(tokenizer: string | null): void {
    const report = this.report();
    const reportId = this.selectedReportId();
    if (!report || reportId === null) {
      this.baselineTokenizer.set(null);
      return;
    }

    if (!tokenizer) {
      this.baselineTokenizer.set(null);
      delete this.baselinePreferences[String(reportId)];
      this.persistDashboardPreferences();
      return;
    }

    if (!benchmarkComparisonTokenizers(report.dashboard).includes(tokenizer)) return;
    this.baselineTokenizer.set(tokenizer);
    this.baselinePreferences[String(reportId)] = tokenizer;
    this.persistDashboardPreferences();
  }

  clearBaseline(): void {
    this.setBaseline(null);
  }

  updateReportTags(reportId: number, tags: readonly string[]): void {
    if (this.updatingReportTagsId() !== null) return;
    this.updatingReportTagsId.set(reportId);
    this.error.set(null);
    this.api.updateReportTags(reportId, [...tags]).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (response) => {
        this.reports.update((reports) => reports.map((item) => item.report_id === reportId
          ? { ...item, tags: [...response.tags] }
          : item));
        const current = this.report();
        if (current?.report_id === reportId) this.report.set({ ...current, tags: [...response.tags] });
        this.updatingReportTagsId.set(null);
      },
      error: (error: unknown) => {
        this.updatingReportTagsId.set(null);
        this.error.set(errorMessage(error, 'Failed to update benchmark report tags.'));
      },
    });
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
    this.baselineTokenizer.set(null);
    this.layout.set([]);
    this.hiddenWidgetIds.set([]);
    this.visualizations.set({});
  }
}
