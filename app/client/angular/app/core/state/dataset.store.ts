import { DestroyRef, Injectable, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subject, catchError, debounce, distinctUntilChanged, map, of, switchMap, timer } from 'rxjs';
import { DatasetsApiService, type DatasetCatalogFilters } from '../api/datasets-api.service';
import { errorMessage, isNotFoundError } from '../api/error-utils';
import type { DatasetAnalysisRequest, DatasetAnalysisResponse, DatasetDownloadRequest, DatasetMetricCatalogCategory, DatasetPreviewItem } from '../api/api.models';

interface DatasetRefreshRequest {
  readonly filters: DatasetCatalogFilters;
  readonly force: boolean;
}

@Injectable({ providedIn: 'root' })
export class DatasetStore {
  private readonly api = inject(DatasetsApiService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly refreshRequests = new Subject<DatasetRefreshRequest>();
  private lastFilters: DatasetCatalogFilters = {};
  private reportLoadSequence = 0;

  readonly datasets = signal<readonly DatasetPreviewItem[]>([]);
  readonly selectedDataset = signal<string | null>(null);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly metricCategories = signal<readonly DatasetMetricCatalogCategory[]>([]);
  readonly loadingMetricCatalog = signal(false);
  readonly report = signal<DatasetAnalysisResponse | null>(null);
  readonly jobProgress = signal<number | null>(null);
  readonly busyAction = signal<string | null>(null);

  constructor() {
    this.refreshRequests.pipe(
      debounce((request) => request.force ? of(0) : timer(250)),
      map(({ filters, force }) => ({
        filters: { ...filters, search: filters.search?.trim() || undefined },
        force,
      })),
      distinctUntilChanged((a, b) => !a.force && !b.force && JSON.stringify(a.filters) === JSON.stringify(b.filters)),
      switchMap(({ filters }) => {
        this.loading.set(true);
        this.error.set(null);
        return this.api.list(filters).pipe(
          map((response) => response.datasets ?? []),
          catchError((error: unknown) => {
            this.error.set(errorMessage(error, 'Failed to fetch datasets.'));
            return of<readonly DatasetPreviewItem[]>([]);
          }),
        );
      }),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe((datasets) => {
      this.datasets.set(datasets);
      this.loading.set(false);
      if (this.selectedDataset() && !datasets.some((dataset) => dataset.dataset_name === this.selectedDataset())) {
        this.selectedDataset.set(null);
      }
    });

    this.loadMetricCatalog();
    this.refresh();
  }

  loadMetricCatalog(): void {
    this.loadingMetricCatalog.set(true);
    this.api.metricsCatalog().pipe(
      catchError((error: unknown) => {
        this.error.set(errorMessage(error, 'Failed to fetch dataset metrics catalog.'));
        return of({ categories: [] as DatasetMetricCatalogCategory[] });
      }),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe((response) => {
      this.metricCategories.set(response.categories ?? []);
      this.loadingMetricCatalog.set(false);
    });
  }

  refresh(filters: DatasetCatalogFilters = this.lastFilters): void {
    this.lastFilters = { ...filters };
    this.refreshRequests.next({ filters: this.lastFilters, force: false });
  }

  private refreshAfterMutation(): void {
    this.refreshRequests.next({ filters: this.lastFilters, force: true });
  }

  select(datasetName: string): void {
    this.selectedDataset.set(datasetName);
  }

  loadLatest(datasetName: string, options: { suppressNotFoundError?: boolean } = {}): void {
    const sequence = ++this.reportLoadSequence;
    this.selectedDataset.set(datasetName);
    if (this.report()?.dataset_name !== datasetName) this.setReport(null);
    this.busyAction.set(`load:${datasetName}`);
    this.error.set(null);
    this.api.latestReport(datasetName).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (report) => {
        if (sequence !== this.reportLoadSequence) return;
        if (report) this.setReport(report);
        else {
          this.setReport(null);
          if (!options.suppressNotFoundError) this.error.set('No validation report found.');
        }
        this.busyAction.set(null);
      },
      error: (error: unknown) => {
        if (sequence !== this.reportLoadSequence) return;
        const message = errorMessage(error, 'Failed to load latest dataset report.');
        const isNoReportFound = isNotFoundError(error) || message.toLowerCase().includes('no validation report found');
        if (options.suppressNotFoundError && isNoReportFound) this.setReport(null);
        else this.error.set(message);
        this.busyAction.set(null);
      },
    });
  }

  analyze(request: DatasetAnalysisRequest): void {
    const sequence = ++this.reportLoadSequence;
    this.busyAction.set(`analyze:${request.dataset_name}`);
    this.jobProgress.set(0);
    this.error.set(null);
    this.api.analyze(request, (status) => this.jobProgress.set(status.progress)).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (report) => { if (sequence !== this.reportLoadSequence) return; this.setReport(report); this.busyAction.set(null); this.jobProgress.set(100); },
      error: (error: unknown) => { if (sequence !== this.reportLoadSequence) return; this.error.set(errorMessage(error, 'Failed to analyze dataset.')); this.busyAction.set(null); this.jobProgress.set(null); },
    });
  }

  download(request: DatasetDownloadRequest): void {
    this.busyAction.set('download');
    this.jobProgress.set(0);
    this.api.download(request, (status) => this.jobProgress.set(status.progress)).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (response) => { this.refreshAfterMutation(); this.selectedDataset.set(response.dataset_name); this.busyAction.set(null); this.jobProgress.set(100); },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to download dataset.')); this.busyAction.set(null); this.jobProgress.set(null); },
    });
  }

  upload(file: File): void {
    this.busyAction.set('upload');
    this.jobProgress.set(0);
    this.api.upload(file, (status) => this.jobProgress.set(status.progress)).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (response) => { this.refreshAfterMutation(); this.selectedDataset.set(response.dataset_name); this.busyAction.set(null); this.jobProgress.set(100); },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to upload dataset.')); this.busyAction.set(null); this.jobProgress.set(null); },
    });
  }

  remove(datasetName: string): void {
    if (this.busyAction() !== null) return;
    ++this.reportLoadSequence;
    this.busyAction.set(`remove:${datasetName}`);
    this.error.set(null);
    this.api.delete(datasetName).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: () => { this.removeFromState(datasetName); this.refreshAfterMutation(); this.busyAction.set(null); },
      error: (error: unknown) => {
        if (isNotFoundError(error)) {
          this.removeFromState(datasetName);
          this.refreshAfterMutation();
        } else this.error.set(errorMessage(error, 'Failed to delete dataset.'));
        this.busyAction.set(null);
      },
    });
  }

  private removeFromState(datasetName: string): void {
    this.datasets.update((datasets) => datasets.filter((dataset) => dataset.dataset_name !== datasetName));
    if (this.selectedDataset() === datasetName) this.selectedDataset.set(null);
    if (this.report()?.dataset_name === datasetName) {
      ++this.reportLoadSequence;
      this.setReport(null);
    }
  }

  private setReport(report: DatasetAnalysisResponse | null): void {
    this.report.set(report);
  }
}
