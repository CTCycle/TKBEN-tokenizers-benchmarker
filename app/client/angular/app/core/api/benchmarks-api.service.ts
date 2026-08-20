import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, tap } from 'rxjs';
import type {
  BenchmarkMetricCatalogResponse,
  BenchmarkReportListResponse,
  BenchmarkReportQuery,
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

  reports(query: BenchmarkReportQuery = {}): Observable<BenchmarkReportListResponse> {
    let params = new HttpParams();
    if (query.search?.trim()) params = params.set('search', query.search.trim());
    if (query.sort) params = params.set('sort', query.sort);
    if (query.offset !== undefined) params = params.set('offset', query.offset);
    if (query.limit !== undefined) params = params.set('limit', query.limit);
    return this.http.get<BenchmarkReportListResponse>('/api/benchmarks/reports', { params });
  }

  metricsCatalog(): Observable<BenchmarkMetricCatalogResponse> {
    return this.http.get<BenchmarkMetricCatalogResponse>('/api/benchmarks/metrics/catalog');
  }

  report(reportId: number): Observable<BenchmarkRunResponse> {
    return this.http.get<BenchmarkRunResponse>(`/api/benchmarks/reports/${reportId}`);
  }

  deleteReport(reportId: number): Observable<void> {
    return this.http.delete<void>(`/api/benchmarks/reports/${reportId}`);
  }

  run(request: BenchmarkRunRequest, onUpdate?: (status: JobStatusResponse) => void, onJobStart?: (job: JobStartResponse) => void): Observable<BenchmarkRunResponse> {
    const start$ = this.http.post<JobStartResponse>('/api/benchmarks/run', request);
    return this.jobs.startAndPoll<BenchmarkRunResponse>(start$.pipe(tap((job) => onJobStart?.(job))), { timeoutMs: 30 * 60 * 1000, onUpdate });
  }
}
