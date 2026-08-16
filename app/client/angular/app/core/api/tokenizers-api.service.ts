import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import type {
  JobStartResponse,
  JobStatusResponse,
  TokenizerDownloadRequest,
  TokenizerDownloadResponse,
  TokenizerListResponse,
  TokenizerReportResponse,
  TokenizerScanResponse,
  TokenizerUploadResponse,
  TokenizerValidationGenerateRequest,
  TokenizerVocabularyPageResponse,
} from './api.models';
import { JobsApiService } from './jobs-api.service';

export interface TokenizerCatalogFilters {
  readonly search?: string;
  readonly source?: 'all' | 'huggingface' | 'custom';
  readonly vocabularyOperator?: 'at_least' | 'at_most';
  readonly vocabulary?: number;
}

@Injectable({ providedIn: 'root' })
export class TokenizersApiService {
  private readonly http = inject(HttpClient);
  private readonly jobs = inject(JobsApiService);

  list(filters: TokenizerCatalogFilters = {}): Observable<TokenizerListResponse> {
    let params = new HttpParams();
    if (filters.search?.trim()) params = params.set('search', filters.search.trim());
    if (filters.source) params = params.set('source', filters.source);
    if (filters.vocabularyOperator) params = params.set('vocabulary_size_operator', filters.vocabularyOperator);
    if (filters.vocabulary !== undefined && Number.isFinite(filters.vocabulary)) params = params.set('vocabulary_size', filters.vocabulary);
    return this.http.get<TokenizerListResponse>('/api/tokenizers/list', { params });
  }

  scan(limit?: number): Observable<TokenizerScanResponse> {
    return this.http.get<TokenizerScanResponse>('/api/tokenizers/scan', { params: limit === undefined ? {} : { limit } });
  }

  download(request: TokenizerDownloadRequest, onUpdate?: (status: JobStatusResponse) => void): Observable<TokenizerDownloadResponse> {
    const start$ = this.http.post<JobStartResponse>('/api/tokenizers/download', request);
    return this.jobs.startAndPoll<TokenizerDownloadResponse>(start$, { onUpdate });
  }

  upload(file: File): Observable<TokenizerUploadResponse> {
    const body = new FormData();
    body.append('file', file);
    return this.http.post<TokenizerUploadResponse>('/api/tokenizers/upload', body);
  }

  clearCustom(): Observable<void> {
    return this.http.delete<void>('/api/tokenizers/custom');
  }

  delete(tokenizerName: string): Observable<void> {
    return this.http.delete<void>(`/api/tokenizers/delete?tokenizer_name=${encodeURIComponent(tokenizerName)}`);
  }

  generateReport(request: TokenizerValidationGenerateRequest, onUpdate?: (status: JobStatusResponse) => void): Observable<TokenizerReportResponse> {
    const start$ = this.http.post<JobStartResponse>('/api/tokenizers/reports/generate', request);
    return this.jobs.startAndPoll<TokenizerReportResponse>(start$, { onUpdate });
  }

  latestReport(tokenizerName: string): Observable<TokenizerReportResponse | null> {
    return this.http.get<TokenizerReportResponse>(`/api/tokenizers/reports/latest?tokenizer_name=${encodeURIComponent(tokenizerName)}`);
  }

  vocabularyPage(reportId: number, offset = 0, limit = 500): Observable<TokenizerVocabularyPageResponse> {
    return this.http.get<TokenizerVocabularyPageResponse>(`/api/tokenizers/reports/${reportId}/vocabulary`, { params: { offset, limit } });
  }
}
