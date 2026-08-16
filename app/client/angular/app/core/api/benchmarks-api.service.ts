import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, tap } from 'rxjs';
import type {
  BenchmarkMetricCatalogResponse,
  BenchmarkReportListResponse,
  BenchmarkRunRequest,
  BenchmarkRunResponse,
  JobStartResponse,
  JobStatusResponse,
} from './api.models';
import { JobsApiService } from './jobs-api.service';

@Injectable({ providedIn: 'root' })
export class BenchmarksApiService {
  private readonly http = inject(HttpClient);
  private readonly jobs = inject(JobsApiService);

  reports(limit = 200): Observable<BenchmarkReportListResponse> {
    return this.http.get<BenchmarkReportListResponse>('/api/benchmarks/reports', { params: { limit } });
  }

  metricsCatalog(): Observable<BenchmarkMetricCatalogResponse> {
    return this.http.get<BenchmarkMetricCatalogResponse>('/api/benchmarks/metrics/catalog');
  }

  report(reportId: number): Observable<BenchmarkRunResponse> {
    return this.http.get<BenchmarkRunResponse>(`/api/benchmarks/reports/${reportId}`);
  }

  run(request: BenchmarkRunRequest, onUpdate?: (status: JobStatusResponse) => void, onJobStart?: (job: JobStartResponse) => void): Observable<BenchmarkRunResponse> {
    const start$ = this.http.post<JobStartResponse>('/api/benchmarks/run', request);
    return this.jobs.startAndPoll<BenchmarkRunResponse>(start$.pipe(tap((job) => onJobStart?.(job))), { timeoutMs: 30 * 60 * 1000, onUpdate });
  }
}
