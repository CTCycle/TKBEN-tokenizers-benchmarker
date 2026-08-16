import { DestroyRef, Injectable, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { catchError, of } from 'rxjs';
import { BenchmarksApiService } from '../api/benchmarks-api.service';
import { errorMessage } from '../api/error-utils';
import type { BenchmarkReportSummary, BenchmarkRunRequest, BenchmarkRunResponse } from '../api/api.types';

@Injectable({ providedIn: 'root' })
export class BenchmarkStore {
  private readonly api = inject(BenchmarksApiService);
  private readonly destroyRef = inject(DestroyRef);

  readonly reports = signal<readonly BenchmarkReportSummary[]>([]);
  readonly selectedReportId = signal<number | null>(null);
  readonly loading = signal(true);
  readonly error = signal<string | null>(null);
  readonly report = signal<BenchmarkRunResponse | null>(null);
  readonly busy = signal(false);
  readonly progress = signal<number | null>(null);
  readonly layout = signal<readonly string[]>([]);

  constructor() {
    this.restoreLayout();
    this.refresh();
  }

  refresh(): void {
    this.loading.set(true);
    this.api.reports().pipe(
      catchError((error: unknown) => {
        this.error.set(errorMessage(error, 'Failed to fetch benchmark reports.'));
        return of({ reports: [] as readonly BenchmarkReportSummary[] });
      }),
    ).subscribe((response) => {
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
      next: (report) => { this.report.set(report); if (!this.layout().length) this.layout.set(report.dashboard.widgets.map((widget) => widget.widget_id)); },
      error: (error: unknown) => this.error.set(errorMessage(error, 'Failed to fetch benchmark report.')),
    });
  }

  run(request: BenchmarkRunRequest): void {
    this.busy.set(true);
    this.progress.set(0);
    this.api.run(request, (status) => this.progress.set(status.progress)).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (report) => { this.report.set(report); this.busy.set(false); this.progress.set(100); this.refresh(); },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to run benchmarks.')); this.busy.set(false); this.progress.set(null); },
    });
  }

  reorder(from: number, to: number): void {
    const next = [...this.layout()];
    const [item] = next.splice(from, 1);
    if (item) next.splice(to, 0, item);
    this.layout.set(next);
    try { localStorage.setItem('tkben:cross-benchmark-dashboard-layout:v3', JSON.stringify(next)); } catch { /* storage is optional */ }
  }

  resetLayout(): void {
    const defaults = this.report()?.dashboard.widgets.map((widget) => widget.widget_id) ?? [];
    this.layout.set(defaults);
    try { localStorage.setItem('tkben:cross-benchmark-dashboard-layout:v3', JSON.stringify(defaults)); } catch { /* storage is optional */ }
  }

  private restoreLayout(): void {
    try {
      const raw = localStorage.getItem('tkben:cross-benchmark-dashboard-layout:v3');
      const parsed: unknown = raw ? JSON.parse(raw) : null;
      if (Array.isArray(parsed) && parsed.every((item) => typeof item === 'string')) this.layout.set(parsed);
    } catch { /* corrupted storage is ignored */ }
  }
}
