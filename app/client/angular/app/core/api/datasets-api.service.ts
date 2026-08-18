import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, catchError, of, throwError } from 'rxjs';
import type {
  CustomDatasetUploadResponse,
  DatasetAnalysisRequest,
  DatasetAnalysisResponse,
  DatasetDownloadRequest,
  DatasetDownloadResponse,
  DatasetListResponse,
  DatasetMetricCatalogResponse,
  JobStartResponse,
  JobStatusResponse,
} from './api.models';
import { JobsApiService } from './jobs-api.service';
import { isNotFoundError } from './error-utils';

export interface DatasetCatalogFilters {
  readonly search?: string;
  readonly source?: 'all' | 'public' | 'custom';
  readonly documentsOperator?: 'at_least' | 'at_most';
  readonly documents?: number;
}

@Injectable({ providedIn: 'root' })
export class DatasetsApiService {
  private readonly http = inject(HttpClient);
  private readonly jobs = inject(JobsApiService);

  list(filters: DatasetCatalogFilters = {}): Observable<DatasetListResponse> {
    let params = new HttpParams();
    if (filters.search?.trim()) params = params.set('search', filters.search.trim());
    if (filters.source) params = params.set('source', filters.source);
    if (filters.documentsOperator) params = params.set('document_count_operator', filters.documentsOperator);
    if (filters.documents !== undefined && Number.isFinite(filters.documents)) params = params.set('document_count', filters.documents);
    return this.http.get<DatasetListResponse>('/api/datasets/list', { params });
  }

  download(request: DatasetDownloadRequest, onUpdate?: (status: JobStatusResponse) => void): Observable<DatasetDownloadResponse> {
    const start$ = this.http.post<JobStartResponse>('/api/datasets/download', request);
    return this.jobs.startAndPoll<DatasetDownloadResponse>(start$, { timeoutMs: 10 * 60 * 1000, onUpdate });
  }

  upload(file: File, onUpdate?: (status: JobStatusResponse) => void): Observable<CustomDatasetUploadResponse> {
    const body = new FormData();
    body.append('file', file);
    const start$ = this.http.post<JobStartResponse>('/api/datasets/upload', body);
    return this.jobs.startAndPoll<CustomDatasetUploadResponse>(start$, { timeoutMs: 10 * 60 * 1000, onUpdate });
  }

  analyze(request: DatasetAnalysisRequest, onUpdate?: (status: JobStatusResponse) => void): Observable<DatasetAnalysisResponse> {
    const start$ = this.http.post<JobStartResponse>('/api/datasets/analyze', request);
    return this.jobs.startAndPoll<DatasetAnalysisResponse>(start$, { timeoutMs: 10 * 60 * 1000, onUpdate });
  }

  metricsCatalog(): Observable<DatasetMetricCatalogResponse> {
    return this.http.get<DatasetMetricCatalogResponse>('/api/datasets/metrics/catalog');
  }

  latestReport(datasetName: string): Observable<DatasetAnalysisResponse | null> {
    return this.http.get<DatasetAnalysisResponse>(`/api/datasets/reports/latest?dataset_name=${encodeURIComponent(datasetName)}`).pipe(
      catchError((error: unknown) => isNotFoundError(error) ? of(null) : throwError(() => error)),
    );
  }

  delete(datasetName: string): Observable<{ status: string; dataset_name: string; message: string }> {
    return this.http.delete<{ status: string; dataset_name: string; message: string }>(`/api/datasets/delete?dataset_name=${encodeURIComponent(datasetName)}`);
  }
}
