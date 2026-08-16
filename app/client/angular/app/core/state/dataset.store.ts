import { DestroyRef, Injectable, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subject, catchError, debounceTime, distinctUntilChanged, map, of, switchMap } from 'rxjs';
import { DatasetsApiService, type DatasetCatalogFilters } from '../api/datasets-api.service';
import { errorMessage } from '../api/error-utils';
import type { DatasetAnalysisRequest, DatasetAnalysisResponse, DatasetDownloadRequest, DatasetPreviewItem } from '../api/api.types';

@Injectable({ providedIn: 'root' })
export class DatasetStore {
  private readonly api = inject(DatasetsApiService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly refreshRequests = new Subject<DatasetCatalogFilters>();

  readonly datasets = signal<readonly DatasetPreviewItem[]>([]);
  readonly selectedDataset = signal<string | null>(null);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly report = signal<DatasetAnalysisResponse | null>(null);
  readonly jobProgress = signal<number | null>(null);
  readonly busyAction = signal<string | null>(null);

  constructor() {
    this.refreshRequests.pipe(
      debounceTime(250),
      map((filters) => ({ ...filters, search: filters.search?.trim() || undefined })),
      distinctUntilChanged((a, b) => JSON.stringify(a) === JSON.stringify(b)),
      switchMap((filters) => {
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

    this.restoreReport();
    this.refresh();
  }

  refresh(filters: DatasetCatalogFilters = {}): void {
    this.refreshRequests.next(filters);
  }

  select(datasetName: string): void {
    this.selectedDataset.set(datasetName);
  }

  loadLatest(datasetName: string): void {
    this.selectedDataset.set(datasetName);
    this.busyAction.set(`load:${datasetName}`);
    this.error.set(null);
    this.api.latestReport(datasetName).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (report) => { this.setReport(report); this.busyAction.set(null); },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to load latest dataset report.')); this.busyAction.set(null); },
    });
  }

  analyze(request: DatasetAnalysisRequest): void {
    this.busyAction.set(`analyze:${request.dataset_name}`);
    this.jobProgress.set(0);
    this.error.set(null);
    this.api.analyze(request, (status) => this.jobProgress.set(status.progress)).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (report) => { this.setReport(report); this.busyAction.set(null); this.jobProgress.set(100); },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to analyze dataset.')); this.busyAction.set(null); this.jobProgress.set(null); },
    });
  }

  download(request: DatasetDownloadRequest): void {
    this.busyAction.set('download');
    this.jobProgress.set(0);
    this.api.download(request, (status) => this.jobProgress.set(status.progress)).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (response) => { this.refresh(); this.selectedDataset.set(response.dataset_name); this.busyAction.set(null); this.jobProgress.set(100); },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to download dataset.')); this.busyAction.set(null); this.jobProgress.set(null); },
    });
  }

  upload(file: File): void {
    this.busyAction.set('upload');
    this.jobProgress.set(0);
    this.api.upload(file, (status) => this.jobProgress.set(status.progress)).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (response) => { this.refresh(); this.selectedDataset.set(response.dataset_name); this.busyAction.set(null); this.jobProgress.set(100); },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to upload dataset.')); this.busyAction.set(null); this.jobProgress.set(null); },
    });
  }

  remove(datasetName: string): void {
    this.busyAction.set(`remove:${datasetName}`);
    this.api.delete(datasetName).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: () => { if (this.selectedDataset() === datasetName) this.selectedDataset.set(null); if (this.report()?.dataset_name === datasetName) this.setReport(null); this.refresh(); this.busyAction.set(null); },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to delete dataset.')); this.busyAction.set(null); },
    });
  }

  private setReport(report: DatasetAnalysisResponse | null): void {
    this.report.set(report);
    try {
      if (report) localStorage.setItem('tkben:last-dataset-report', JSON.stringify(report));
      else localStorage.removeItem('tkben:last-dataset-report');
    } catch { /* storage is optional */ }
  }

  private restoreReport(): void {
    try {
      const raw = localStorage.getItem('tkben:last-dataset-report');
      if (!raw) return;
      const parsed: unknown = JSON.parse(raw);
      if (parsed && typeof parsed === 'object' && typeof (parsed as { dataset_name?: unknown }).dataset_name === 'string') this.report.set(parsed as DatasetAnalysisResponse);
    } catch { /* corrupted storage is ignored */ }
  }
}
