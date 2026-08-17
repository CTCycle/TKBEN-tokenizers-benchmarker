import { DestroyRef, Injectable, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subject, catchError, debounceTime, distinctUntilChanged, map, of, switchMap } from 'rxjs';
import { TokenizersApiService, type TokenizerCatalogFilters } from '../api/tokenizers-api.service';
import { errorMessage } from '../api/error-utils';
import type { TokenizerDownloadRequest, TokenizerListItem, TokenizerReportResponse, TokenizerVocabularyPageResponse } from '../api/api.types';

@Injectable({ providedIn: 'root' })
export class TokenizersStore {
  private readonly api = inject(TokenizersApiService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly refreshRequests = new Subject<TokenizerCatalogFilters>();

  readonly tokenizers = signal<readonly TokenizerListItem[]>([]);
  readonly selectedTokenizer = signal<string | null>(null);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly report = signal<TokenizerReportResponse | null>(null);
  readonly vocabulary = signal<TokenizerVocabularyPageResponse | null>(null);
  readonly busyAction = signal<string | null>(null);
  readonly jobProgress = signal<number | null>(null);
  readonly scannedTokenizers = signal<readonly string[]>([]);
  readonly scanLoading = signal(false);
  readonly downloadWarning = signal<string | null>(null);

  constructor() {
    this.refreshRequests.pipe(
      debounceTime(250),
      map((filters) => ({ ...filters, search: filters.search?.trim() || undefined })),
      distinctUntilChanged((a, b) => JSON.stringify(a) === JSON.stringify(b)),
      switchMap((filters) => {
        this.loading.set(true);
        this.error.set(null);
        return this.api.list(filters).pipe(
          map((response) => response.tokenizers ?? []),
          catchError((error: unknown) => {
            this.error.set(errorMessage(error, 'Failed to fetch tokenizers.'));
            return of<readonly TokenizerListItem[]>([]);
          }),
        );
      }),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe((tokenizers) => {
      this.tokenizers.set(tokenizers);
      this.loading.set(false);
      if (this.selectedTokenizer() && !tokenizers.some((tokenizer) => tokenizer.tokenizer_name === this.selectedTokenizer())) {
        this.selectedTokenizer.set(null);
      }
    });

    this.restoreReport();
    this.refresh();
  }

  refresh(filters: TokenizerCatalogFilters = {}): void {
    this.refreshRequests.next(filters);
  }

  select(tokenizerName: string): void {
    this.selectedTokenizer.set(tokenizerName);
  }

  scan(): void {
    this.scanLoading.set(true);
    this.error.set(null);
    this.api.scan().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (response) => { this.scannedTokenizers.set(response.identifiers ?? []); this.scanLoading.set(false); },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to scan Hugging Face tokenizers.')); this.scanLoading.set(false); },
    });
  }

  generateReport(tokenizerName: string): void {
    this.select(tokenizerName);
    this.busyAction.set(`report:${tokenizerName}`);
    this.jobProgress.set(0);
    this.api.generateReport({ tokenizer_name: tokenizerName }, (status) => this.jobProgress.set(status.progress)).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (report) => { this.setReport(report); this.busyAction.set(null); this.jobProgress.set(100); },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to generate tokenizer report.')); this.busyAction.set(null); this.jobProgress.set(null); },
    });
  }

  openReport(tokenizerName: string): void {
    this.select(tokenizerName);
    this.busyAction.set(`report:${tokenizerName}`);
    this.jobProgress.set(0);
    this.error.set(null);
    this.api.latestReport(tokenizerName).pipe(
      switchMap((report) => report
        ? of(report)
        : this.api.generateReport({ tokenizer_name: tokenizerName }, (status) => this.jobProgress.set(status.progress))),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe({
      next: (report) => { this.setReport(report); this.busyAction.set(null); this.jobProgress.set(100); },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to open tokenizer report.')); this.busyAction.set(null); this.jobProgress.set(null); },
    });
  }

  loadLatest(tokenizerName: string): void {
    this.select(tokenizerName);
    this.busyAction.set(`load:${tokenizerName}`);
    this.api.latestReport(tokenizerName).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (report) => { if (report) this.setReport(report); else this.error.set('No persisted tokenizer report found.'); this.busyAction.set(null); },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to load latest tokenizer report.')); this.busyAction.set(null); },
    });
  }

  loadVocabulary(offset = 0, limit = 500): void {
    const reportId = this.report()?.report_id;
    if (reportId === undefined) return;
    this.api.vocabularyPage(reportId, offset, limit).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (page) => this.vocabulary.set(page),
      error: (error: unknown) => this.error.set(errorMessage(error, 'Failed to load tokenizer vocabulary.')),
    });
  }

  upload(file: File): void {
    this.busyAction.set('upload');
    this.api.upload(file).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: () => { this.refresh(); this.busyAction.set(null); },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to upload tokenizer.')); this.busyAction.set(null); },
    });
  }

  download(request: TokenizerDownloadRequest): void {
    this.busyAction.set('download');
    this.jobProgress.set(0);
    this.downloadWarning.set(null);
    this.api.download(request, (status) => this.jobProgress.set(status.progress)).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (response) => {
        this.refresh();
        this.busyAction.set(null);
        this.jobProgress.set(100);
        if (response.failed?.length) this.downloadWarning.set(`Some tokenizers could not be downloaded: ${response.failed.join(', ')}`);
      },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to download tokenizers.')); this.busyAction.set(null); this.jobProgress.set(null); },
    });
  }

  remove(tokenizerName: string): void {
    this.busyAction.set(`remove:${tokenizerName}`);
    this.api.delete(tokenizerName).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: () => { if (this.selectedTokenizer() === tokenizerName) this.selectedTokenizer.set(null); this.refresh(); this.busyAction.set(null); },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to remove tokenizer.')); this.busyAction.set(null); },
    });
  }

  private setReport(report: TokenizerReportResponse): void {
    this.report.set(report);
    try { localStorage.setItem('tkben.lastTokenizerReport', JSON.stringify(report)); } catch { /* storage is optional */ }
    this.loadVocabulary();
  }

  private restoreReport(): void {
    try {
      const raw = localStorage.getItem('tkben.lastTokenizerReport');
      if (!raw) return;
      const parsed: unknown = JSON.parse(raw);
      if (parsed && typeof parsed === 'object' && typeof (parsed as { tokenizer_name?: unknown }).tokenizer_name === 'string') this.report.set(parsed as TokenizerReportResponse);
    } catch { /* corrupted storage is ignored */ }
  }
}
