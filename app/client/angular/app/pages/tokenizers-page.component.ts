import { Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormControl, FormGroup, ReactiveFormsModule } from '@angular/forms';
import { debounceTime } from 'rxjs';
import { TokenizersStore } from '../core/state/tokenizers.store';
import { HistogramChartComponent } from '../components/histogram-chart.component';
import { ExportApiService } from '../core/api/export-api.service';
import { errorMessageAsync } from '../core/api/error-utils';
import { ModalA11yDirective } from '../core/ui/modal-a11y.directive';
import type {
  SupportedTokenizerPipeline,
  TokenizerDiscoverySort,
  VocabularySort,
} from '../core/api/api.models';

type TokenizerManagerTab = 'discover' | 'add-by-name' | 'upload-json';

@Component({
  selector: 'app-tokenizers-page',
  imports: [ReactiveFormsModule, HistogramChartComponent, ModalA11yDirective],
  templateUrl: './tokenizers-page.component.html',
})
export class TokenizersPageComponent {
  protected readonly store = inject(TokenizersStore);
  private readonly exportApi = inject(ExportApiService);
  private readonly destroyRef = inject(DestroyRef);
  protected readonly addTokenizerOpen = signal(false);
  protected readonly activeTokenizerTab = signal<TokenizerManagerTab>('discover');
  protected readonly manualTokenizerInput = signal('');
  protected readonly manualTokenizerIds = computed(() => this.manualTokenizerInput().split(/\r?\n|,/).map((item) => item.trim()).filter(Boolean));
  protected readonly discoveryAdvancedOpen = signal(false);
  protected readonly downloadProgressVisible = computed(() => this.store.busyAction() === 'download');
  protected readonly exportError = signal<string | null>(null);
  protected readonly filters = new FormGroup({
    search: new FormControl('', { nonNullable: true }),
    source: new FormControl('', { nonNullable: true }),
    vocabularyOperator: new FormControl<'at_least' | 'at_most'>('at_least', { nonNullable: true }),
    vocabulary: new FormControl<number | null>(null),
  });
  protected readonly discoveryForm = new FormGroup({
    search: new FormControl('', { nonNullable: true }),
    limit: new FormControl(50, { nonNullable: true }),
    pipelineTag: new FormControl<SupportedTokenizerPipeline | ''>('', { nonNullable: true }),
    sort: new FormControl<TokenizerDiscoverySort>('downloads', { nonNullable: true }),
    author: new FormControl('', { nonNullable: true }),
    access: new FormControl<'all' | 'public' | 'gated'>('all', { nonNullable: true }),
    includeTags: new FormControl('', { nonNullable: true }),
    excludeTags: new FormControl('', { nonNullable: true }),
    vocabularyOperator: new FormControl<'at_least' | 'at_most' | ''>('', { nonNullable: true }),
    vocabularySize: new FormControl<number | null>(null),
    vocabularySort: new FormControl<VocabularySort>('none', { nonNullable: true }),
  });

  constructor() {
    this.filters.valueChanges.pipe(debounceTime(10), takeUntilDestroyed(this.destroyRef)).subscribe(() => this.refresh());
  }

  protected refresh(): void {
    const value = this.filters.getRawValue();
    this.store.refresh({
      search: value.search,
      source: value.source === 'hugging_face' ? 'huggingface' : value.source === 'custom' ? 'custom' : undefined,
      vocabularyOperator: value.vocabularyOperator,
      vocabulary: value.vocabulary ?? undefined,
    });
  }

  protected selectTokenizer(name: string): void {
    this.store.select(name);
  }

  protected openAddTokenizer(): void {
    this.activeTokenizerTab.set('discover');
    this.addTokenizerOpen.set(true);
    this.discoverTokenizers();
  }
  protected closeAddTokenizer(): void { this.addTokenizerOpen.set(false); }
  protected selectTokenizerTab(tab: TokenizerManagerTab): void { this.activeTokenizerTab.set(tab); }
  protected toggleDiscoveryAdvanced(): void { this.discoveryAdvancedOpen.update((open) => !open); }
  protected handleTokenizerTabKeydown(event: KeyboardEvent, tab: TokenizerManagerTab): void {
    const tabs: readonly TokenizerManagerTab[] = ['discover', 'add-by-name', 'upload-json'];
    const index = tabs.indexOf(tab);
    let nextIndex: number | null = null;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') nextIndex = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') nextIndex = (index - 1 + tabs.length) % tabs.length;
    if (event.key === 'Home') nextIndex = 0;
    if (event.key === 'End') nextIndex = tabs.length - 1;
    if (nextIndex === null) return;
    event.preventDefault();
    const nextTab = tabs[nextIndex];
    this.activeTokenizerTab.set(nextTab);
    document.getElementById(`tokenizer-tab-${nextTab}`)?.focus();
  }
  protected discoverTokenizers(): void {
    const value = this.discoveryForm.getRawValue();
    const splitTags = (raw: string): string[] => [...new Set(raw.split(/[\n,]/).map((tag) => tag.trim()).filter(Boolean))];
    this.store.discover({
      search: value.search,
      limit: value.limit,
      pipeline_tag: value.pipelineTag || undefined,
      author: value.author,
      include_tags: splitTags(value.includeTags),
      exclude_tags: splitTags(value.excludeTags),
      access: value.access,
      sort: value.sort,
      vocabulary_operator: value.vocabularyOperator || undefined,
      vocabulary_size: value.vocabularySize ?? undefined,
      vocabulary_sort: value.vocabularySort,
    });
  }
  protected downloadManualTokenizers(): void {
    const tokenizers = this.manualTokenizerInput().split(/\r?\n|,/).map((item) => item.trim()).filter(Boolean);
    if (tokenizers.length) this.store.download({ tokenizers });
  }
  protected downloadSelectedDiscoveryTokenizers(): void {
    const tokenizers = [...this.store.selectedDiscoveryIds()];
    if (tokenizers.length) this.store.download({ tokenizers });
  }
  protected toggleDiscoveryTokenizer(identifier: string, enabled: boolean): void {
    this.store.toggleDiscoverySelection(identifier, enabled);
  }
  protected openReport(name: string): void { this.store.openReport(name); }
  protected loadLatest(name: string): void { this.store.loadLatest(name); }
  protected dashboardDescription(): string {
    const report = this.store.report();
    return report ? `Report ${report.report_id} for ${report.tokenizer_name}` : 'Open a tokenizer report from the preview list to populate this dashboard.';
  }
  protected formatNumber(value: number | null | undefined, digits = 2): string {
    if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toFixed(digits);
  }
  protected formatPercent(value: number | null | undefined): string {
    return value === null || value === undefined || Number.isNaN(value) ? 'N/A' : `${this.formatNumber(value, 2)}%`;
  }
  protected vocabularyOffset(): number { return this.store.vocabulary()?.offset ?? 0; }
  protected vocabularyLimit(): number { return this.store.vocabulary()?.limit ?? 500; }
  protected vocabularyTotal(): number { return this.store.vocabulary()?.total ?? 0; }
  protected vocabularyStart(): number { return this.vocabularyTotal() > 0 ? this.vocabularyOffset() + 1 : 0; }
  protected vocabularyEnd(): number { return this.vocabularyTotal() > 0 ? Math.min(this.vocabularyOffset() + (this.store.vocabulary()?.items.length ?? 0), this.vocabularyTotal()) : 0; }
  protected canGoPrevious(): boolean { return this.vocabularyOffset() > 0; }
  protected canGoNext(): boolean { return this.vocabularyOffset() + this.vocabularyLimit() < this.vocabularyTotal(); }
  protected changeVocabularyPageSize(event: Event): void {
    const value = Number((event.target as HTMLSelectElement).value);
    this.store.loadVocabulary(0, Math.max(1, Math.min(5000, Math.floor(value || 500))));
  }
  protected previousVocabularyPage(): void {
    if (!this.store.report() || !this.canGoPrevious()) return;
    this.store.loadVocabulary(Math.max(0, this.vocabularyOffset() - this.vocabularyLimit()), this.vocabularyLimit());
  }
  protected nextVocabularyPage(): void {
    if (!this.store.report() || !this.canGoNext()) return;
    this.store.loadVocabulary(this.vocabularyOffset() + this.vocabularyLimit(), this.vocabularyLimit());
  }
  protected exportDashboard(): void {
    const report = this.store.report();
    if (!report) return;
    this.exportError.set(null);
    this.exportApi.dashboardPdf({
      dashboardType: 'tokenizer',
      reportName: `tokenizer-${report.tokenizer_name}-report-${report.report_id}`,
      fileName: `tokenizer-${report.tokenizer_name}-report-${report.report_id}.pdf`,
      dashboardPayload: { report, vocabulary_items: this.store.vocabulary()?.items ?? [] } as unknown as Record<string, unknown>,
    }).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (result) => { const url = URL.createObjectURL(result.blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = result.fileName; anchor.click(); URL.revokeObjectURL(url); },
      error: (error: unknown) => { void errorMessageAsync(error, 'Failed to export dashboard.').then((message) => this.exportError.set(message)); },
    });
  }
  protected removeTokenizer(name: string): void { if (window.confirm(`Remove ${name} from the database?`)) this.store.remove(name); }
  protected uploadFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) { this.store.upload(file); this.addTokenizerOpen.set(false); input.value = ''; }
  }
}
