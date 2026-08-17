import { Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { debounceTime } from 'rxjs';
import { DatasetStore } from '../core/state/dataset.store';
import { HistogramChartComponent } from '../components/histogram-chart.component';
import { ExportApiService } from '../core/api/export-api.service';
import { errorMessageAsync } from '../core/api/error-utils';
import { WordCloudComponent } from '../components/word-cloud.component';
import { ModalA11yDirective } from '../core/ui/modal-a11y.directive';
import {
  buildWordCloudFromWordFrequencies,
  buildZipfCurveFromWordFrequencies,
  hasMetricValue,
  isRecord,
  metricDisplayValue,
  normalizeCount,
  normalizePercent,
  parseWordCloudTerms,
  parseWordFrequencyItems,
  parseZipfCurve,
  toNumber,
} from '../core/utils/dataset-dashboard-data';

interface DatasetFiltersForm {
  search: FormControl<string>;
  source: FormControl<string>;
  documentsOperator: FormControl<'at_least' | 'at_most'>;
  documents: FormControl<number | null>;
}

@Component({
  selector: 'app-dataset-page',
  imports: [ReactiveFormsModule, HistogramChartComponent, WordCloudComponent, ModalA11yDirective],
  templateUrl: './dataset-page.component.html',
})
export class DatasetPageComponent {
  protected readonly store = inject(DatasetStore);
  private readonly exportApi = inject(ExportApiService);
  private readonly destroyRef = inject(DestroyRef);
  protected readonly addDatasetOpen = signal(false);
  protected readonly validationOpen = signal(false);
  protected readonly validationDataset = signal<string | null>(null);
  protected readonly validationStep = signal<0 | 1 | 2>(0);
  protected readonly selectedMetricKeys = signal<readonly string[]>([]);
  protected readonly banner = signal<string | null>(null);
  protected readonly presets = [
    { id: 'wikitext', label: 'wikitext', description: 'Clean Wikipedia articles, multiple sizes, common baseline.', configuration: 'wikitext-2-v1' },
    { id: 'c4', label: 'c4', description: 'Colossal Clean Crawled Corpus, large filtered web crawl.' },
    { id: 'ag_news', label: 'ag_news', description: 'Short news classification dataset.' },
    { id: 'imdb', label: 'imdb', description: 'Long-form movie reviews.' },
  ] as const;
  protected readonly downloadForm = new FormGroup({
    corpus: new FormControl('wikitext', { nonNullable: true, validators: [Validators.required] }),
    configuration: new FormControl('wikitext-2-v1', { nonNullable: true }),
  });
  protected readonly validationForm = new FormGroup({
    sessionName: new FormControl('', { nonNullable: true }),
    samplingMode: new FormControl<'fraction' | 'count'>('fraction', { nonNullable: true }),
    samplingFraction: new FormControl(1, { nonNullable: true, validators: [Validators.min(0.01), Validators.max(1)] }),
    samplingCount: new FormControl(1000, { nonNullable: true, validators: [Validators.min(1), Validators.max(100000)] }),
    minLength: new FormControl<number | null>(null, { validators: [Validators.min(0)] }),
    maxLength: new FormControl<number | null>(null, { validators: [Validators.min(0)] }),
    excludeEmpty: new FormControl(true, { nonNullable: true }),
  });
  protected readonly filters = new FormGroup<DatasetFiltersForm>({
    search: new FormControl('', { nonNullable: true }),
    source: new FormControl('', { nonNullable: true }),
    documentsOperator: new FormControl('at_least', { nonNullable: true }),
    documents: new FormControl<number | null>(null),
  });
  protected readonly aggregate = computed<Record<string, unknown>>(() => {
    const value = this.store.report()?.aggregate_statistics;
    return isRecord(value) ? value : {};
  });
  protected readonly characterSlices = computed(() => [
    { key: 'Whitespace', value: toNumber(this.aggregate()['chars.whitespace_ratio']) },
    { key: 'Punctuation', value: toNumber(this.aggregate()['chars.punctuation_ratio']) },
    { key: 'Digits', value: toNumber(this.aggregate()['chars.digit_ratio']) },
    { key: 'Uppercase', value: toNumber(this.aggregate()['chars.uppercase_ratio']) },
    { key: 'Non-ASCII', value: toNumber(this.aggregate()['chars.non_ascii_ratio']) },
    { key: 'Control', value: toNumber(this.aggregate()['chars.control_ratio']) },
    { key: 'Other', value: toNumber(this.aggregate()['chars.other_ratio']) },
  ].filter((item) => item.value > 0));
  protected readonly characterTotal = computed(() => Math.max(1, this.characterSlices().reduce((sum, item) => sum + item.value, 0)));
  protected readonly zipfCurve = computed(() => {
    const aggregate = this.aggregate();
    const parsed = parseZipfCurve(aggregate['words.zipf_curve']);
    if (parsed.length) return parsed;
    return buildZipfCurveFromWordFrequencies(this.wordFrequencies());
  });
  protected readonly wordFrequencies = computed(() => {
    const report = this.store.report();
    if (!report) return [];
    if (report.most_common_words?.length) return parseWordFrequencyItems(report.most_common_words);
    return parseWordFrequencyItems(this.aggregate()['words.most_common']);
  });
  protected readonly wordCloudTerms = computed(() => {
    const report = this.store.report();
    if (!report) return [];
    const terms = report.word_cloud_terms?.length ? parseWordCloudTerms(report.word_cloud_terms) : parseWordCloudTerms(this.aggregate()['words.word_cloud']);
    return terms.length ? terms : buildWordCloudFromWordFrequencies(this.wordFrequencies());
  });
  protected readonly documentCount = computed(() => hasMetricValue(this.aggregate()['corpus.document_count']) ? toNumber(this.aggregate()['corpus.document_count']) : this.store.report()?.document_count ?? 0);
  protected readonly emptyCount = computed(() => {
    const raw = this.aggregate()['quality.empty_rate'];
    return hasMetricValue(raw) ? Math.round(toNumber(raw) * this.documentCount()) : null;
  });
  protected readonly aggregateRows = computed(() => {
    const aggregate = this.aggregate();
    return [
      { label: 'Num documents', value: this.store.report() ? normalizeCount(this.documentCount()) : '—' },
      { label: 'Mean length', value: metricDisplayValue(aggregate['doc.length_mean'], (value) => value.toFixed(2)) },
      { label: 'Min length', value: metricDisplayValue(aggregate['doc.length_min'], normalizeCount) },
      { label: 'Max length', value: metricDisplayValue(aggregate['doc.length_max'], normalizeCount) },
      { label: 'Empty count', value: this.emptyCount() === null ? '—' : normalizeCount(this.emptyCount() ?? 0) },
      { label: 'Length CV', value: metricDisplayValue(aggregate['doc.length_cv'], (value) => value.toFixed(4)) },
      { label: 'p50', value: metricDisplayValue(aggregate['doc.length_p50'], normalizeCount) },
      { label: 'p90', value: metricDisplayValue(aggregate['doc.length_p90'], normalizeCount) },
      { label: 'p99', value: metricDisplayValue(aggregate['doc.length_p99'], normalizeCount) },
    ];
  });
  protected readonly wordMetricRows = computed(() => {
    const aggregate = this.aggregate();
    return [
      { label: 'Vocabulary size', value: metricDisplayValue(aggregate['corpus.unique_words'], normalizeCount) },
      { label: 'MATTR', value: metricDisplayValue(aggregate['corpus.mattr'], (value) => value.toFixed(4)) },
      { label: 'Entropy', value: metricDisplayValue(aggregate['words.shannon_entropy'], (value) => value.toFixed(4)) },
      { label: 'Hapax ratio', value: metricDisplayValue(aggregate['words.hapax_ratio'], (value) => value.toFixed(4)) },
      { label: 'Zipf slope', value: metricDisplayValue(aggregate['words.zipf_slope'], (value) => value.toFixed(4)) },
      { label: 'Gini', value: metricDisplayValue(aggregate['words.frequency_gini'], (value) => value.toFixed(4)) },
      { label: 'HHI', value: metricDisplayValue(aggregate['words.hhi'], (value) => value.toFixed(6)) },
    ];
  });
  protected readonly entropy = computed(() => toNumber(this.aggregate()['words.normalized_entropy']));
  protected readonly shannonEntropy = computed(() => toNumber(this.aggregate()['words.shannon_entropy']));
  protected readonly duplicateRate = computed(() => toNumber(this.aggregate()['quality.duplicate_rate']));
  protected readonly nearDuplicateRate = computed(() => toNumber(this.aggregate()['quality.near_duplicate_rate']));
  protected readonly topKConcentration = computed(() => toNumber(this.aggregate()['words.topk_concentration']));
  protected readonly rareTailMass = computed(() => toNumber(this.aggregate()['words.rare_tail_mass']));
  protected readonly allMetricKeys = computed(() => this.store.metricCategories().flatMap((category) => category.metrics.map((metric) => metric.key)));

  protected readonly donutColors = ['#f59e0b', '#fb7185', '#38bdf8', '#34d399', '#a78bfa', '#f97316', '#64748b'];
  protected readonly zipfPoints = computed(() => {
    const values = this.zipfCurve();
    const max = Math.max(1, ...values.map((item) => item.frequency));
    return values.map((item, index) => `${24 + (index / Math.max(values.length - 1, 1)) * 572},${166 - (item.frequency / max) * 140}`).join(' ');
  });

  protected donutPath(index: number): string {
    const slices = this.characterSlices();
    const total = this.characterTotal();
    const start = slices.slice(0, index).reduce((sum, item) => sum + item.value, 0) / total * Math.PI * 2 - Math.PI / 2;
    const end = start + slices[index].value / total * Math.PI * 2;
    const large = end - start > Math.PI ? 1 : 0;
    const outer = (angle: number) => `${100 + Math.cos(angle) * 96} ${140 + Math.sin(angle) * 96}`;
    const inner = (angle: number) => `${100 + Math.cos(angle) * 58} ${140 + Math.sin(angle) * 58}`;
    return `M ${outer(start)} A 96 96 0 ${large} 1 ${outer(end)} L ${inner(end)} A 58 58 0 ${large} 0 ${inner(start)} Z`;
  }

  protected displayPercent(value: unknown): string { return normalizePercent(toNumber(value)); }
  protected hasAggregateMetric(key: string): boolean { return hasMetricValue(this.aggregate()[key]); }
  protected dashboardDescription(): string {
    const report = this.store.report();
    if (!report) return 'Load a saved report or run validation to populate this dashboard.';
    const timestamp = report.created_at ? ` (${new Date(report.created_at).toLocaleString()})` : '';
    return `Latest persisted session for ${report.dataset_name}${timestamp}`;
  }

  constructor() {
    this.filters.valueChanges.pipe(debounceTime(10), takeUntilDestroyed(this.destroyRef)).subscribe(() => this.refresh());
  }

  protected refresh(): void {
    const value = this.filters.getRawValue();
    this.store.refresh({
      search: value.search,
      source: value.source === 'public' || value.source === 'custom' ? value.source : undefined,
      documentsOperator: value.documentsOperator,
      documents: value.documents ?? undefined,
    });
  }

  protected selectDataset(datasetName: string): void {
    this.store.select(datasetName);
    this.banner.set(null);
    this.store.loadLatest(datasetName, { suppressNotFoundError: true });
  }

  protected openValidation(datasetName: string): void {
    this.validationDataset.set(datasetName);
    this.validationStep.set(0);
    this.selectedMetricKeys.set([...this.allMetricKeys()]);
    this.validationForm.reset({ sessionName: '', samplingMode: 'fraction', samplingFraction: 1, samplingCount: 1000, minLength: null, maxLength: null, excludeEmpty: true });
    this.validationOpen.set(true);
  }

  protected toggleMetric(metricKey: string, enabled: boolean): void {
    const current = new Set(this.selectedMetricKeys());
    if (enabled) current.add(metricKey); else current.delete(metricKey);
    this.selectedMetricKeys.set([...current]);
  }

  protected toggleMetricCategory(category: { metrics: readonly { key: string }[] }, enabled: boolean): void {
    const current = new Set(this.selectedMetricKeys());
    for (const metric of category.metrics) {
      if (enabled) current.add(metric.key); else current.delete(metric.key);
    }
    this.selectedMetricKeys.set([...current]);
  }

  protected categorySelected(category: { metrics: readonly { key: string }[] }): boolean {
    return category.metrics.length > 0 && category.metrics.every((metric) => this.selectedMetricKeys().includes(metric.key));
  }

  protected nextValidationStep(): void {
    if (this.validationStep() === 0 && this.store.metricCategories().length > 0 && this.selectedMetricKeys().length === 0) return;
    if (this.validationStep() < 2) this.validationStep.update((step) => (step + 1) as 0 | 1 | 2);
  }

  protected previousValidationStep(): void {
    if (this.validationStep() > 0) this.validationStep.update((step) => (step - 1) as 0 | 1 | 2);
  }

  protected runValidation(): void {
    const datasetName = this.validationDataset();
    if (!datasetName) return;
    const value = this.validationForm.getRawValue();
    const sessionName = value.sessionName.trim();
    const minLength = value.minLength === null || !Number.isFinite(value.minLength) ? null : Math.max(0, Math.floor(value.minLength));
    const maxLength = value.maxLength === null || !Number.isFinite(value.maxLength) ? null : Math.max(0, Math.floor(value.maxLength));
    this.store.analyze({
      dataset_name: datasetName,
      session_name: sessionName || null,
      selected_metric_keys: this.selectedMetricKeys().length ? [...this.selectedMetricKeys()] : null,
      sampling: value.samplingMode === 'fraction'
        ? { fraction: Math.min(1, Math.max(0.01, value.samplingFraction)) }
        : { count: Math.min(100000, Math.max(1, Math.floor(value.samplingCount))) },
      filters: { min_length: minLength, max_length: maxLength, exclude_empty: value.excludeEmpty },
      metric_parameters: {},
    });
    this.validationOpen.set(false);
  }

  protected loadLatest(datasetName: string): void {
    this.store.select(datasetName);
    this.store.loadLatest(datasetName);
  }

  protected removeDataset(datasetName: string): void {
    if (window.confirm(`Remove ${datasetName} from the database?`)) this.store.remove(datasetName);
  }

  protected choosePreset(preset: { id: string; configuration?: string }): void {
    this.downloadForm.patchValue({ corpus: preset.id, configuration: preset.configuration ?? '' });
  }

  protected downloadSelected(): void {
    const value = this.downloadForm.getRawValue();
    this.store.download({ corpus: value.corpus, configs: { configuration: value.configuration || null } });
    this.addDatasetOpen.set(false);
  }

  protected uploadFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) {
      this.store.upload(file);
      this.addDatasetOpen.set(false);
      input.value = '';
    }
  }

  protected exportDashboard(): void {
    const report = this.store.report();
    if (!report) return;
    this.exportApi.dashboardPdf({ dashboardType: 'dataset', reportName: `dataset-${report.dataset_name}-report-${report.report_id ?? 'latest'}`, fileName: `dataset-${report.dataset_name}-report.pdf`, dashboardPayload: report as unknown as Record<string, unknown> }).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (result) => { const url = URL.createObjectURL(result.blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = result.fileName; anchor.click(); URL.revokeObjectURL(url); },
      error: (error: unknown) => { void errorMessageAsync(error, 'Failed to export dashboard.').then((message) => this.banner.set(message)); },
    });
  }

  protected notify(message: string): void {
    this.banner.set(message);
  }

  protected closeAddDataset(): void {
    this.addDatasetOpen.set(false);
  }

  protected closeValidation(): void {
    this.validationOpen.set(false);
    this.validationStep.set(0);
  }
}
