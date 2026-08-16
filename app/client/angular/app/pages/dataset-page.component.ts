import { Component, DestroyRef, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { debounceTime } from 'rxjs';
import { DatasetStore } from '../core/state/dataset.store';
import { HistogramChartComponent } from '../components/histogram-chart.component';
import { ExportApiService } from '../core/api/export-api.service';
import { errorMessageAsync } from '../core/api/error-utils';

interface DatasetFiltersForm {
  search: FormControl<string>;
  source: FormControl<string>;
  documentsOperator: FormControl<'at_least' | 'at_most'>;
  documents: FormControl<number | null>;
}

@Component({
  selector: 'app-dataset-page',
  imports: [ReactiveFormsModule, HistogramChartComponent],
  templateUrl: './dataset-page.component.html',
})
export class DatasetPageComponent {
  protected readonly store = inject(DatasetStore);
  private readonly exportApi = inject(ExportApiService);
  private readonly destroyRef = inject(DestroyRef);
  protected readonly addDatasetOpen = signal(false);
  protected readonly validationOpen = signal(false);
  protected readonly validationDataset = signal<string | null>(null);
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
  });
  protected readonly filters = new FormGroup<DatasetFiltersForm>({
    search: new FormControl('', { nonNullable: true }),
    source: new FormControl('', { nonNullable: true }),
    documentsOperator: new FormControl('at_least', { nonNullable: true }),
    documents: new FormControl<number | null>(null),
  });

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
    this.store.loadLatest(datasetName);
  }

  protected openValidation(datasetName: string): void {
    this.validationDataset.set(datasetName);
    this.validationForm.reset({ sessionName: '' });
    this.validationOpen.set(true);
  }

  protected runValidation(): void {
    const datasetName = this.validationDataset();
    if (!datasetName) return;
    const sessionName = this.validationForm.controls.sessionName.value.trim();
    this.store.analyze({ dataset_name: datasetName, session_name: sessionName || null });
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
}
