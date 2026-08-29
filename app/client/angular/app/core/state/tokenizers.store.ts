import { DestroyRef, Injectable, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subject, catchError, debounce, distinctUntilChanged, map, of, switchMap, timer } from 'rxjs';
import { TokenizersApiService, type TokenizerCatalogFilters } from '../api/tokenizers-api.service';
import { errorMessage, isNotFoundError } from '../api/error-utils';
import type {
  TokenizerDiscoveryItem,
  TokenizerDiscoveryQuery,
  TokenizerDownloadRequest,
  TokenizerListItem,
  TokenizerReportResponse,
  TokenizerVocabularyPageResponse,
} from '../api/api.models';

interface TokenizerRefreshRequest {
  readonly filters: TokenizerCatalogFilters;
  readonly force: boolean;
}

@Injectable({ providedIn: 'root' })
export class TokenizersStore {
  private readonly api = inject(TokenizersApiService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly refreshRequests = new Subject<TokenizerRefreshRequest>();
  private lastFilters: TokenizerCatalogFilters = {};

  readonly tokenizers = signal<readonly TokenizerListItem[]>([]);
  readonly selectedTokenizer = signal<string | null>(null);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly report = signal<TokenizerReportResponse | null>(null);
  readonly vocabulary = signal<TokenizerVocabularyPageResponse | null>(null);
  readonly busyAction = signal<string | null>(null);
  readonly jobProgress = signal<number | null>(null);
  readonly discoveryResults = signal<readonly TokenizerDiscoveryItem[]>([]);
  readonly discoveryLoading = signal(false);
  readonly discoveryError = signal<string | null>(null);
  readonly selectedDiscoveryIds = signal<readonly string[]>([]);
  readonly downloadWarning = signal<string | null>(null);
  private discoverySequence = 0;
  private reportLoadSequence = 0;

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

    this.refresh();
  }

  refresh(filters: TokenizerCatalogFilters = this.lastFilters): void {
    this.lastFilters = { ...filters };
    this.refreshRequests.next({ filters: this.lastFilters, force: false });
  }

  private refreshAfterMutation(): void {
    this.refreshRequests.next({ filters: this.lastFilters, force: true });
  }

  select(tokenizerName: string): void {
    this.selectedTokenizer.set(tokenizerName);
  }

  discover(query: TokenizerDiscoveryQuery): void {
    const sequence = ++this.discoverySequence;
    this.discoveryLoading.set(true);
    this.discoveryError.set(null);
    this.api.discover(query).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (response) => {
        if (sequence !== this.discoverySequence) return;
        const results = response.items ?? [];
        const availableIds = new Set(results.map((item) => item.identifier));
        this.discoveryResults.set(results);
        this.selectedDiscoveryIds.update((selected) => selected.filter((id) => availableIds.has(id)));
        this.discoveryLoading.set(false);
      },
      error: (error: unknown) => {
        if (sequence !== this.discoverySequence) return;
        this.discoveryError.set(errorMessage(error, 'Failed to discover Hugging Face tokenizers.'));
        this.discoveryLoading.set(false);
      },
    });
  }

  toggleDiscoverySelection(identifier: string, enabled: boolean): void {
    this.selectedDiscoveryIds.update((selected) => {
      const next = new Set(selected);
      if (enabled) next.add(identifier); else next.delete(identifier);
      return [...next];
    });
  }

  generateReport(tokenizerName: string): void {
    const sequence = ++this.reportLoadSequence;
    this.select(tokenizerName);
    this.busyAction.set(`report:${tokenizerName}`);
    this.jobProgress.set(0);
    this.api.generateReport({ tokenizer_name: tokenizerName }, (status) => this.jobProgress.set(status.progress)).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (report) => { if (sequence !== this.reportLoadSequence) return; this.setReport(report); this.busyAction.set(null); this.jobProgress.set(100); },
      error: (error: unknown) => { if (sequence !== this.reportLoadSequence) return; this.error.set(errorMessage(error, 'Failed to generate tokenizer report.')); this.busyAction.set(null); this.jobProgress.set(null); },
    });
  }

  openReport(tokenizerName: string): void {
    const sequence = ++this.reportLoadSequence;
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
      next: (report) => { if (sequence !== this.reportLoadSequence) return; this.setReport(report); this.busyAction.set(null); this.jobProgress.set(100); },
      error: (error: unknown) => { if (sequence !== this.reportLoadSequence) return; this.error.set(errorMessage(error, 'Failed to open tokenizer report.')); this.busyAction.set(null); this.jobProgress.set(null); },
    });
  }

  loadLatest(tokenizerName: string): void {
    const sequence = ++this.reportLoadSequence;
    this.select(tokenizerName);
    this.busyAction.set(`load:${tokenizerName}`);
    this.api.latestReport(tokenizerName).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (report) => { if (sequence !== this.reportLoadSequence) return; if (report) this.setReport(report); else { this.clearReport(); this.error.set('No persisted tokenizer report found.'); } this.busyAction.set(null); },
      error: (error: unknown) => { if (sequence !== this.reportLoadSequence) return; this.error.set(errorMessage(error, 'Failed to load latest tokenizer report.')); this.busyAction.set(null); },
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
      next: () => { this.refreshAfterMutation(); this.busyAction.set(null); },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to upload tokenizer.')); this.busyAction.set(null); },
    });
  }

  download(request: TokenizerDownloadRequest): void {
    this.busyAction.set('download');
    this.jobProgress.set(0);
    this.downloadWarning.set(null);
    this.api.download(request, (status) => this.jobProgress.set(status.progress)).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (response) => {
        this.refreshAfterMutation();
        this.busyAction.set(null);
        this.jobProgress.set(100);
        if (response.failed?.length) this.downloadWarning.set(`Some tokenizers could not be downloaded: ${response.failed.join(', ')}`);
      },
      error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to download tokenizers.')); this.busyAction.set(null); this.jobProgress.set(null); },
    });
  }

  remove(tokenizerName: string): void {
    if (this.busyAction() !== null) return;
    ++this.reportLoadSequence;
    this.busyAction.set(`remove:${tokenizerName}`);
    this.error.set(null);
    this.api.delete(tokenizerName).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: () => { this.removeFromState(tokenizerName); this.refreshAfterMutation(); this.busyAction.set(null); },
      error: (error: unknown) => {
        if (isNotFoundError(error)) {
          this.removeFromState(tokenizerName);
          this.refreshAfterMutation();
        } else this.error.set(errorMessage(error, 'Failed to remove tokenizer.'));
        this.busyAction.set(null);
      },
    });
  }

  private removeFromState(tokenizerName: string): void {
    this.tokenizers.update((tokenizers) => tokenizers.filter((tokenizer) => tokenizer.tokenizer_name !== tokenizerName));
    if (this.selectedTokenizer() === tokenizerName) this.selectedTokenizer.set(null);
    if (this.report()?.tokenizer_name === tokenizerName) this.clearReport();
  }

  private setReport(report: TokenizerReportResponse): void {
    this.report.set(report);
    this.loadVocabulary();
  }

  private clearReport(): void {
    this.report.set(null);
    this.vocabulary.set(null);
  }
}
