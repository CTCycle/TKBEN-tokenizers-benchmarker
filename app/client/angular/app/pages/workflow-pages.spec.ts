// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { FormGroup } from '@angular/forms';
import { of } from 'rxjs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ExportApiService } from '../core/api/export-api.service';
import { DatasetsApiService } from '../core/api/datasets-api.service';
import { DatasetStore } from '../core/state/dataset.store';
import { TokenizersStore } from '../core/state/tokenizers.store';
import { SettingsStore } from '../core/state/settings.store';
import { BenchmarkStore } from '../core/state/benchmark.store';
import type { DatasetMetricCatalogCategory, RuntimeSettingsValues } from '../core/api/api.models';
import { DatasetPageComponent } from './dataset-page.component';
import { TokenizersPageComponent } from './tokenizers-page.component';
import { CrossBenchmarkPageComponent } from './cross-benchmark-page.component';
import { SettingsPageComponent } from './settings-page.component';


const metricCategories: DatasetMetricCatalogCategory[] = [{
  category_key: 'quality',
  category_label: 'Quality',
  metrics: [{
    key: 'quality.empty_rate',
    label: 'Empty rate',
    description: 'Empty document rate',
    scope: 'aggregate',
    value_kind: 'percent',
    core: true,
  }],
}];

afterEach(() => {
  TestBed.resetTestingModule();
});

describe('DatasetPageComponent workflow', () => {
  it('normalizes validation form input into the backend request', () => {
    const store = {
      report: signal(null),
      metricCategories: signal(metricCategories),
      analyze: vi.fn(),
      refresh: vi.fn(),
      select: vi.fn(),
      loadLatest: vi.fn(),
      download: vi.fn(),
      upload: vi.fn(),
      remove: vi.fn(),
    };
    TestBed.configureTestingModule({ providers: [
      { provide: DatasetStore, useValue: store },
      { provide: ExportApiService, useValue: {} },
      { provide: DatasetsApiService, useValue: { validationPreview: vi.fn() } },
    ] });
    const page = TestBed.runInInjectionContext(() => new DatasetPageComponent()) as unknown as {
      openValidation: (datasetName: string) => void;
      validationForm: FormGroup;
      runValidation: () => void;
    };

    page.openValidation('custom/demo');
    page.validationForm.patchValue({
      sessionName: '  focused session  ',
      samplingMode: 'count',
      samplingCount: 100_001,
      minLength: -4,
      maxLength: 3.9,
      excludeEmpty: false,
    });
    page.runValidation();

    expect(store.analyze).toHaveBeenCalledWith({
      dataset_name: 'custom/demo',
      session_name: 'focused session',
      selected_metric_keys: ['quality.empty_rate'],
      sampling: { count: 100_000 },
      filters: { min_length: 0, max_length: 3, exclude_empty: false },
      metric_parameters: {},
    });
  });

  it('blocks inverted document length filters before submission', () => {
    const store = {
      report: signal(null),
      metricCategories: signal(metricCategories),
      analyze: vi.fn(),
      refresh: vi.fn(),
      select: vi.fn(),
      loadLatest: vi.fn(),
      download: vi.fn(),
      upload: vi.fn(),
      remove: vi.fn(),
    };
    TestBed.configureTestingModule({ providers: [
      { provide: DatasetStore, useValue: store },
      { provide: ExportApiService, useValue: {} },
      { provide: DatasetsApiService, useValue: { validationPreview: vi.fn() } },
    ] });
    const page = TestBed.runInInjectionContext(() => new DatasetPageComponent()) as unknown as {
      openValidation: (datasetName: string) => void;
      validationForm: FormGroup;
      validationStep: { set: (value: 0 | 1 | 2) => void; (): number };
      nextValidationStep: () => void;
      runValidation: () => void;
    };

    page.openValidation('custom/demo');
    page.validationForm.patchValue({ minLength: 100, maxLength: 1 });
    page.validationStep.set(1);
    page.nextValidationStep();
    page.runValidation();

    expect(page.validationForm.hasError('minLengthExceedsMaxLength')).toBe(true);
    expect(page.validationStep()).toBe(1);
    expect(store.analyze).not.toHaveBeenCalled();
  });

  it('previews the validation scope with the run payload and blocks empty matches', () => {
    vi.useFakeTimers();
    const validationPreview = vi.fn().mockReturnValue(of({
      dataset_name: 'custom/demo',
      total_documents: 10,
      empty_documents: 2,
      min_length: 0,
      max_length: 40,
      mean_length: 12.5,
      length_histogram: [{ start: 0, end: 40, count: 10 }],
      matching_documents: 0,
      sampled_documents: 0,
    }));
    const store = {
      report: signal(null),
      metricCategories: signal(metricCategories),
      analyze: vi.fn(),
      refresh: vi.fn(),
    };
    TestBed.configureTestingModule({ providers: [
      { provide: DatasetStore, useValue: store },
      { provide: ExportApiService, useValue: {} },
      { provide: DatasetsApiService, useValue: { validationPreview } },
    ] });
    const page = TestBed.runInInjectionContext(() => new DatasetPageComponent()) as unknown as {
      openValidation: (datasetName: string) => void;
      validationForm: FormGroup;
      validationStep: () => number;
      nextValidationStep: () => void;
      runValidation: () => void;
    };

    try {
      page.openValidation('custom/demo');
      page.nextValidationStep();
      page.validationForm.patchValue({ samplingMode: 'count', samplingCount: 50, minLength: 100 });
      vi.advanceTimersByTime(300);
      expect(validationPreview).toHaveBeenCalledTimes(1);
      expect(validationPreview).toHaveBeenCalledWith({
        dataset_name: 'custom/demo',
        sampling: { count: 50 },
        filters: { min_length: 100, max_length: null, exclude_empty: true },
      });

      page.nextValidationStep();
      page.runValidation();
      expect(page.validationStep()).toBe(1);
      expect(store.analyze).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it('restores the grouped preset catalogue and keeps manual/upload actions available', () => {
    const store = {
      report: signal(null),
      metricCategories: signal(metricCategories),
      busyAction: signal<string | null>(null),
      refresh: vi.fn(),
      select: vi.fn(),
      loadLatest: vi.fn(),
      download: vi.fn(),
      upload: vi.fn(),
      remove: vi.fn(),
    };
    TestBed.configureTestingModule({ providers: [
      { provide: DatasetStore, useValue: store },
      { provide: ExportApiService, useValue: {} },
      { provide: DatasetsApiService, useValue: { validationPreview: vi.fn() } },
    ] });
    const page = TestBed.runInInjectionContext(() => new DatasetPageComponent()) as unknown as {
      presets: readonly { group: string; datasets: readonly { id: string; configuration?: string }[] }[];
      selectedPreset: () => string | null;
      addDatasetOpen: () => boolean;
      activeDatasetTab: () => string;
      openAddDataset: () => void;
      selectDatasetTab: (tab: string) => void;
      handleDatasetTabKeydown: (event: KeyboardEvent, tab: string) => void;
      isPresetGroupCollapsed: (group: string) => boolean;
      togglePresetGroup: (group: string) => void;
      choosePreset: (preset: { id: string; label: string; description: string; configuration?: string }) => void;
      downloadSelected: () => void;
      downloadForm: FormGroup;
      uploadFile: (event: Event) => void;
    };

    expect(page.presets).toHaveLength(7);
    expect(page.presets.flatMap((group) => group.datasets)).toHaveLength(24);
    expect(page.presets.every((group) => !page.isPresetGroupCollapsed(group.group))).toBe(true);
    expect(page.activeDatasetTab()).toBe('predefined');

    page.openAddDataset();
    expect(page.addDatasetOpen()).toBe(true);
    expect(page.activeDatasetTab()).toBe('predefined');
    page.downloadForm.patchValue({ corpus: 'organization/custom-dataset', configuration: 'main' });
    page.selectDatasetTab('add-by-name');
    page.selectDatasetTab('predefined');
    expect(page.downloadForm.getRawValue()).toMatchObject({ corpus: 'organization/custom-dataset', configuration: 'main' });
    page.handleDatasetTabKeydown(new KeyboardEvent('keydown', { key: 'ArrowRight' }), 'predefined');
    expect(page.activeDatasetTab()).toBe('add-by-name');
    page.handleDatasetTabKeydown(new KeyboardEvent('keydown', { key: 'End' }), 'add-by-name');
    expect(page.activeDatasetTab()).toBe('custom');
    page.handleDatasetTabKeydown(new KeyboardEvent('keydown', { key: 'Home' }), 'custom');
    expect(page.activeDatasetTab()).toBe('predefined');
    page.togglePresetGroup('General Corpora');
    expect(page.isPresetGroupCollapsed('General Corpora')).toBe(true);

    page.choosePreset({ id: 'squad', label: 'squad', description: 'Wikipedia-based QA dataset.' });
    expect(page.selectedPreset()).toBe('squad');
    expect(page.downloadForm.getRawValue()).toMatchObject({ corpus: 'squad', configuration: '' });
    page.downloadSelected();
    expect(store.download).toHaveBeenCalledWith({ corpus: 'squad', configs: { configuration: null } });

    const file = new File(['text'], 'custom.csv', { type: 'text/csv' });
    page.uploadFile({ target: { files: [file], value: 'custom.csv' } } as unknown as Event);
    expect(store.upload).toHaveBeenCalledWith(file);
  });
});

describe('TokenizersPageComponent workflow', () => {
  it('parses manual downloads and maps catalog filters', () => {
    const store = {
      report: signal(null),
      vocabulary: signal(null),
      discoveryResults: signal([{ identifier: 'alpha/model' }, { identifier: 'beta/model' }]),
      busyAction: signal<string | null>(null),
      refresh: vi.fn(),
      select: vi.fn(),
      discover: vi.fn(),
      selectedDiscoveryIds: signal<readonly string[]>([]),
      toggleDiscoverySelection: vi.fn(),
      download: vi.fn(),
      openReport: vi.fn(),
      loadLatest: vi.fn(),
      loadVocabulary: vi.fn(),
      remove: vi.fn(),
      upload: vi.fn(),
    };
    TestBed.configureTestingModule({ providers: [
      { provide: TokenizersStore, useValue: store },
      { provide: ExportApiService, useValue: {} },
    ] });
    const page = TestBed.runInInjectionContext(() => new TokenizersPageComponent()) as unknown as {
      manualTokenizerInput: { set: (value: string) => void };
      downloadManualTokenizers: () => void;
      filters: FormGroup;
      refresh: () => void;
    };

    page.manualTokenizerInput.set(' alpha/model,\nbeta/model ');
    page.downloadManualTokenizers();
    page.filters.patchValue({
      search: 'alpha',
      source: 'hugging_face',
      vocabularyOperator: 'at_most',
      vocabulary: 5000,
    });
    page.refresh();

    expect(store.download).toHaveBeenCalledWith({ tokenizers: ['alpha/model', 'beta/model'] });
    expect(store.refresh).toHaveBeenLastCalledWith({
      search: 'alpha',
      source: 'huggingface',
      vocabularyOperator: 'at_most',
      vocabulary: 5000,
    });
  });

  it('caps playground tokenizer selection and sends preview requests', () => {
    const store = {
      tokenizers: signal(['alpha', 'beta', 'gamma', 'delta'].map((tokenizer_name) => ({ tokenizer_name, source: 'huggingface', has_report: false, vocabulary_size: null }))),
      previewLoading: signal(false),
      preview: vi.fn(),
      refresh: vi.fn(),
    };
    TestBed.configureTestingModule({ providers: [
      { provide: TokenizersStore, useValue: store },
      { provide: ExportApiService, useValue: {} },
    ] });
    const page = TestBed.runInInjectionContext(() => new TokenizersPageComponent()) as unknown as {
      playgroundText: { set: (value: string) => void };
      playgroundSelectedTokenizers: () => readonly string[];
      togglePlaygroundTokenizer: (name: string, enabled: boolean) => void;
      runPlayground: () => void;
    };

    for (const name of ['alpha', 'beta', 'gamma', 'delta']) page.togglePlaygroundTokenizer(name, true);
    expect(page.playgroundSelectedTokenizers()).toEqual(['alpha', 'beta', 'gamma']);
    page.togglePlaygroundTokenizer('beta', false);
    page.runPlayground();
    expect(store.preview).not.toHaveBeenCalled();

    page.playgroundText.set('alpha beta');
    page.runPlayground();
    expect(store.preview).toHaveBeenCalledWith({ text: 'alpha beta', tokenizers: ['alpha', 'gamma'], add_special_tokens: true });
  });

  it('opens on Discover, preserves tab form state, and supports keyboard tab navigation', () => {
    const store = {
      report: signal(null),
      vocabulary: signal(null),
      discoveryResults: signal([]),
      discoveryLoading: signal(false),
      discoveryError: signal<string | null>(null),
      error: signal<string | null>(null),
      downloadWarning: signal<string | null>(null),
      busyAction: signal<string | null>(null),
      selectedDiscoveryIds: signal<readonly string[]>(['alpha/model']),
      refresh: vi.fn(),
      select: vi.fn(),
      discover: vi.fn(),
      toggleDiscoverySelection: vi.fn(),
      download: vi.fn(),
      openReport: vi.fn(),
      loadLatest: vi.fn(),
      loadVocabulary: vi.fn(),
      remove: vi.fn(),
      upload: vi.fn(),
    };
    TestBed.configureTestingModule({ providers: [
      { provide: TokenizersStore, useValue: store },
      { provide: ExportApiService, useValue: {} },
    ] });
    const page = TestBed.runInInjectionContext(() => new TokenizersPageComponent()) as unknown as {
      activeTokenizerTab: () => string;
      openAddTokenizer: () => void;
      selectTokenizerTab: (tab: string) => void;
      handleTokenizerTabKeydown: (event: KeyboardEvent, tab: string) => void;
      discoveryForm: FormGroup;
      discoveryAdvancedOpen: () => boolean;
      toggleDiscoveryAdvanced: () => void;
      downloadSelectedDiscoveryTokenizers: () => void;
    };

    expect(page.activeTokenizerTab()).toBe('discover');
    page.discoveryForm.patchValue({ search: 'long/repository', author: 'owner' });
    page.selectTokenizerTab('add-by-name');
    expect(page.activeTokenizerTab()).toBe('add-by-name');
    page.selectTokenizerTab('discover');
    expect(page.discoveryForm.getRawValue()).toMatchObject({ search: 'long/repository', author: 'owner' });
    expect(page.discoveryAdvancedOpen()).toBe(false);
    page.toggleDiscoveryAdvanced();
    expect(page.discoveryAdvancedOpen()).toBe(true);

    page.handleTokenizerTabKeydown(new KeyboardEvent('keydown', { key: 'ArrowRight' }), 'discover');
    expect(page.activeTokenizerTab()).toBe('add-by-name');
    page.handleTokenizerTabKeydown(new KeyboardEvent('keydown', { key: 'End' }), 'add-by-name');
    expect(page.activeTokenizerTab()).toBe('upload-json');
    page.openAddTokenizer();
    expect(page.activeTokenizerTab()).toBe('discover');
    expect(store.discover).toHaveBeenCalled();

    page.selectTokenizerTab('discover');
    page.downloadSelectedDiscoveryTokenizers();
    expect(store.download).toHaveBeenCalledWith({ tokenizers: ['alpha/model'] });
  });

  it('hydrates discovery defaults and constraints from SettingsStore', () => {
    const store = {
      report: signal(null),
      vocabulary: signal(null),
      discoveryResults: signal([]),
      busyAction: signal<string | null>(null),
      refresh: vi.fn(),
      select: vi.fn(),
      discover: vi.fn(),
      selectedDiscoveryIds: signal<readonly string[]>([]),
      toggleDiscoverySelection: vi.fn(),
      download: vi.fn(),
      remove: vi.fn(),
      upload: vi.fn(),
    };
    const settingsStore = {
      settings: signal({
        tokenizers: {
          default_discovery_limit: 7,
          max_discovery_limit: 9,
        },
      }),
    };
    TestBed.configureTestingModule({ providers: [
      { provide: TokenizersStore, useValue: store },
      { provide: SettingsStore, useValue: settingsStore },
      { provide: ExportApiService, useValue: {} },
    ] });
    const page = TestBed.runInInjectionContext(() => new TokenizersPageComponent()) as unknown as {
      discoveryForm: FormGroup;
      discoverTokenizers: () => void;
      maxDiscoveryLimit: () => number | null;
    };
    TestBed.tick();

    expect(page.discoveryForm.controls['limit'].value).toBe(7);
    expect(page.maxDiscoveryLimit()).toBe(9);
    page.discoveryForm.controls['limit'].setValue(10);
    expect(page.discoveryForm.controls['limit'].invalid).toBe(true);
    page.discoveryForm.controls['limit'].setValue(7);
    page.discoverTokenizers();
    expect(store.discover).toHaveBeenCalledWith(expect.objectContaining({ limit: 7 }));
  });
});

describe('CrossBenchmarkPageComponent workflow', () => {
  it('builds a trimmed benchmark request, applies settings defaults, and updates widget visibility', () => {
    const store = {
      report: signal({
        dashboard: {
          widgets: [
            { widget_id: 'visible', category_label: 'Core' },
            { widget_id: 'hidden', category_label: 'Core' },
          ],
        },
      }),
      layout: signal<readonly string[]>(['visible', 'hidden']),
      hiddenWidgetIds: signal<readonly string[]>([]),
      availableDatasets: signal(['custom/default']),
      availableTokenizers: signal(['alpha', 'beta']),
      metricCategories: signal([{ category_key: 'efficiency', category_label: 'Efficiency', metrics: [{ key: 'eff.speed', label: 'Speed' }] }]),
      busy: signal(false),
      run: vi.fn(),
      setHiddenWidgetIds: vi.fn(),
      reorder: vi.fn(),
      resetLayout: vi.fn(),
      cancel: vi.fn(),
    };
    const settingsStore = {
      settings: signal({
        benchmarks: {
          default_max_documents: 2500,
          default_batch_size: 32,
          default_parallelism: 4,
        },
      }),
    };
    TestBed.configureTestingModule({ providers: [
      { provide: BenchmarkStore, useValue: store },
      { provide: SettingsStore, useValue: settingsStore },
      { provide: ExportApiService, useValue: {} },
    ] });
    const page = TestBed.runInInjectionContext(() => new CrossBenchmarkPageComponent()) as unknown as {
      runForm: FormGroup;
      runSelectedTokenizers: { set: (value: readonly string[]) => void };
      runSelectedMetricKeys: { set: (value: readonly string[]) => void };
      openRun: () => void;
      runBenchmark: () => void;
      openCustomize: () => void;
      toggleCustomizeWidget: (widgetId: string, enabled: boolean) => void;
    };

    page.openRun();
    expect(page.runForm.getRawValue()).toMatchObject({
      maxDocuments: 2500,
      batchSize: 32,
      parallelism: 4,
    });
    page.runForm.patchValue({ dataset: '  custom/default  ', runName: '  quick run  ' });
    page.runSelectedTokenizers.set(['alpha', 'beta']);
    page.runSelectedMetricKeys.set(['eff.speed']);
    page.runBenchmark();

    expect(store.run).toHaveBeenCalledWith(expect.objectContaining({
      tokenizers: ['alpha', 'beta'],
      dataset_name: 'custom/default',
      run_name: 'quick run',
      selected_metric_keys: ['eff.speed'],
      config: expect.objectContaining({
        max_documents: 2500,
        batch_size: 32,
        parallelism: 4,
      }),
    }));

    page.openCustomize();
    page.toggleCustomizeWidget('hidden', false);
    expect(store.setHiddenWidgetIds).toHaveBeenCalledWith(['hidden']);
  });

  it('clones an eligible benchmark configuration and keeps new-run initialization separate', () => {
    const report = {
      report_id: 42,
      run_name: 'production benchmark',
      dataset_name: 'custom/default',
      documents_processed: 900,
      tokenizers_processed: ['alpha', 'beta'],
      selected_metric_keys: ['eff.speed'],
      config: {
        max_documents: 750,
        warmup_trials: 4,
        timed_trials: 12,
        batch_size: 64,
        seed: 99,
        parallelism: 3,
        add_special_tokens: true,
        padding: true,
        truncation: true,
        max_length: 256,
        store_per_document_stats: false,
        per_document_sample_size: 321,
      },
      dashboard: { widgets: [] },
    };
    const store = {
      report: signal(report),
      layout: signal<readonly string[]>([]),
      hiddenWidgetIds: signal<readonly string[]>([]),
      availableDatasets: signal(['custom/default']),
      availableTokenizers: signal(['alpha', 'beta']),
      metricCategories: signal([{ category_key: 'efficiency', category_label: 'Efficiency', metrics: [{ key: 'eff.speed', label: 'Speed' }] }]),
      busy: signal(false),
      run: vi.fn(),
      setHiddenWidgetIds: vi.fn(),
      reorder: vi.fn(),
      resetLayout: vi.fn(),
      cancel: vi.fn(),
    };
    const settingsStore = {
      settings: signal({ benchmarks: { default_max_documents: 2500, default_batch_size: 32, default_parallelism: 4 } }),
    };
    TestBed.configureTestingModule({ providers: [
      { provide: BenchmarkStore, useValue: store },
      { provide: SettingsStore, useValue: settingsStore },
      { provide: ExportApiService, useValue: {} },
    ] });
    const page = TestBed.runInInjectionContext(() => new CrossBenchmarkPageComponent()) as unknown as {
      runForm: FormGroup;
      runMode: () => 'new' | 'clone';
      runOpen: () => boolean;
      runStep: () => 1 | 2 | 3;
      runSelectedTokenizers: () => readonly string[];
      runSelectedMetricKeys: () => readonly string[];
      openCloneRun: () => void;
      openRun: () => void;
      runBenchmark: () => void;
    };

    page.openCloneRun();
    expect(page.runMode()).toBe('clone');
    expect(page.runOpen()).toBe(true);
    expect(page.runStep()).toBe(3);
    expect(page.runForm.getRawValue()).toMatchObject({
      dataset: 'custom/default',
      tokenizers: 'alpha,beta',
      runName: 'Clone of production benchmark',
      maxDocuments: 750,
      warmupTrials: 4,
      timedTrials: 12,
      batchSize: 64,
      seed: 99,
      parallelism: 3,
      addSpecialTokens: true,
      padding: true,
      truncation: true,
      maxLength: 256,
      storePerDocumentStats: false,
      perDocumentSampleSize: 321,
    });
    page.runBenchmark();
    expect(store.run).toHaveBeenCalledWith(expect.objectContaining({
      config: {
        max_documents: 750,
        warmup_trials: 4,
        timed_trials: 12,
        batch_size: 64,
        seed: 99,
        parallelism: 3,
        add_special_tokens: true,
        padding: true,
        truncation: true,
        max_length: 256,
        store_per_document_stats: false,
        per_document_sample_size: 321,
      },
    }));

    page.openRun();
    expect(page.runMode()).toBe('new');
    expect(page.runForm.getRawValue()).toMatchObject({ tokenizers: '', runName: '', maxLength: null, maxDocuments: 2500 });
    expect(page.runSelectedTokenizers()).toEqual([]);
  });

  it('keeps an ineligible clone inline instead of opening the wizard', () => {
    const store = {
      report: signal({
        report_id: 7,
        run_name: 'stale run',
        dataset_name: 'missing/dataset',
        tokenizers_processed: ['alpha'],
        selected_metric_keys: ['eff.speed'],
        dashboard: { widgets: [] },
      }),
      layout: signal<readonly string[]>([]),
      hiddenWidgetIds: signal<readonly string[]>([]),
      availableDatasets: signal(['custom/default']),
      availableTokenizers: signal(['alpha']),
      metricCategories: signal([{ category_key: 'efficiency', category_label: 'Efficiency', metrics: [{ key: 'eff.speed', label: 'Speed' }] }]),
      busy: signal(false),
      run: vi.fn(),
      setHiddenWidgetIds: vi.fn(),
      reorder: vi.fn(),
      resetLayout: vi.fn(),
      cancel: vi.fn(),
    };
    TestBed.configureTestingModule({ providers: [
      { provide: BenchmarkStore, useValue: store },
      { provide: SettingsStore, useValue: { settings: signal({}) } },
      { provide: ExportApiService, useValue: {} },
    ] });
    const page = TestBed.runInInjectionContext(() => new CrossBenchmarkPageComponent()) as unknown as {
      openCloneRun: () => void;
      runOpen: () => boolean;
      cloneError: () => string | null;
    };

    page.openCloneRun();
    expect(page.runOpen()).toBe(false);
    expect(page.cloneError()).toContain('dataset');
  });
});

const defaults: RuntimeSettingsValues = {
  tokenizers: {
    default_discovery_limit: 50,
    max_discovery_limit: 250,
    max_discovery_candidates: 750,
    metadata_candidate_multiplier: 3,
    max_upload_bytes: 10 * 1024 * 1024,
  },
  datasets: {
    histogram_bins: 20,
    streaming_batch_size: 10_000,
    max_upload_bytes: 25 * 1024 * 1024,
    download_timeout_seconds: 180,
    download_retry_attempts: 3,
    download_retry_backoff_seconds: 2,
  },
  benchmarks: {
    default_max_documents: 1000,
    default_batch_size: 16,
    default_parallelism: 1,
    streaming_batch_size: 1000,
  },
  jobs: { polling_interval: 1 },
};

const current: RuntimeSettingsValues = {
  ...defaults,
  datasets: { ...defaults.datasets, histogram_bins: 30, max_upload_bytes: 32 * 1024 * 1024 },
  benchmarks: { ...defaults.benchmarks, default_max_documents: 2500, default_batch_size: 32, default_parallelism: 2 },
  jobs: { polling_interval: 2 },
};

afterEach(() => {
  TestBed.resetTestingModule();
  vi.useRealTimers();
});

describe('SettingsPageComponent', () => {
  function createPage(overriddenKeys: string[] = ['datasets.histogram_bins']) {
    const store = {
      settings: signal<RuntimeSettingsValues | null>(current),
      defaults: signal<RuntimeSettingsValues | null>(defaults),
      revision: signal(4),
      overriddenKeys: signal(overriddenKeys),
      warning: signal<string | null>(null),
      loading: signal(false),
      saving: signal(false),
      error: signal<string | null>(null),
      conflict: signal(false),
      save: vi.fn(),
      reset: vi.fn(),
      reload: vi.fn(),
      notePendingEdit: vi.fn(),
      isOverridden: vi.fn((key: string) => overriddenKeys.includes(key)),
    };
    TestBed.configureTestingModule({ providers: [
      { provide: SettingsStore, useValue: store },
    ] });
    const page = TestBed.runInInjectionContext(() => new SettingsPageComponent()) as unknown as {
      activeTab: () => string;
      form: FormGroup;
      selectTab: (tab: string) => void;
      handleTabKeydown: (event: KeyboardEvent, tab: string) => void;
      persistCurrentSettings: () => void;
      ngOnDestroy: () => void;
      resetSetting: (key: string) => void;
      resetAll: () => void;
      reloadAfterConflict: () => void;
      tokenizerLimitsInvalid: () => boolean;
      fieldError: (controlName: string) => string | null;
      fieldDescribedBy: (helpId: string, controlName: string) => string;
    };
    TestBed.tick();
    return { page, store };
  }

  it('hydrates the form, supports tab keyboard navigation, and validates tokenizer limits', () => {
    const { page } = createPage();

    expect(page.form.getRawValue()).toMatchObject({
      histogramBins: 30,
      datasetMaxUploadMiB: 32,
      defaultDiscoveryLimit: 50,
      benchmarkDefaultMaxDocuments: 2500,
      benchmarkDefaultBatchSize: 32,
      benchmarkDefaultParallelism: 2,
      jobPollingInterval: 2,
    });
    expect(page.form.pristine).toBe(true);

    page.selectTab('tokenizers');
    expect(page.activeTab()).toBe('tokenizers');
    page.handleTabKeydown(new KeyboardEvent('keydown', { key: 'ArrowRight' }), 'tokenizers');
    expect(page.activeTab()).toBe('benchmarks');
    page.handleTabKeydown(new KeyboardEvent('keydown', { key: 'ArrowRight' }), 'benchmarks');
    expect(page.activeTab()).toBe('runtime');

    page.form.patchValue({ defaultDiscoveryLimit: 200, maxDiscoveryLimit: 100 });
    expect(page.tokenizerLimitsInvalid()).toBe(true);
    expect(page.form.invalid).toBe(true);
  });

  it('explains field-level and cross-field validation failures inline', () => {
    const { page } = createPage();

    page.form.controls['histogramBins'].setValue(4);
    page.form.controls['histogramBins'].markAsTouched();
    expect(page.fieldError('histogramBins')).toBe('Must be at least 5.');
    expect(page.fieldDescribedBy('settings-help-histogram-bins', 'histogramBins'))
      .toBe('settings-help-histogram-bins settings-error-histogramBins');

    page.form.controls['downloadRetryAttempts'].setValue(1.5);
    page.form.controls['downloadRetryAttempts'].markAsTouched();
    expect(page.fieldError('downloadRetryAttempts')).toBe('Use a whole number.');

    page.form.controls['metadataCandidateMultiplier'].setValue(11);
    page.form.controls['metadataCandidateMultiplier'].markAsTouched();
    expect(page.fieldError('metadataCandidateMultiplier')).toBe('Must be no more than 10.');

    page.form.patchValue({ defaultDiscoveryLimit: 200, maxDiscoveryLimit: 100 });
    page.form.controls['defaultDiscoveryLimit'].markAsTouched();
    page.form.controls['maxDiscoveryLimit'].markAsTouched();
    expect(page.fieldError('defaultDiscoveryLimit')).toBe('Must not exceed Maximum discovery limit.');
    expect(page.fieldError('maxDiscoveryLimit')).toBe('Must be at least Default discovery limit.');

    page.form.patchValue({ defaultDiscoveryLimit: 200, maxDiscoveryLimit: 100, maxDiscoveryCandidates: 50 });
    page.form.controls['defaultDiscoveryLimit'].markAsTouched();
    page.form.controls['maxDiscoveryLimit'].markAsTouched();
    page.form.controls['maxDiscoveryCandidates'].markAsTouched();
    expect(page.fieldError('defaultDiscoveryLimit')).toBe('Must not exceed Maximum discovery limit.');
    expect(page.fieldError('maxDiscoveryLimit')).toBe('Must be at least Default discovery limit and no more than Discovery candidate cap.');
    expect(page.fieldError('maxDiscoveryCandidates')).toBe('Must be at least Maximum discovery limit.');
  });

  it('converts MiB to bytes and sends a typed autosave patch', () => {
    const { page, store } = createPage([]);
    page.form.patchValue({
      datasetMaxUploadMiB: 40,
      tokenizerMaxUploadMiB: 12,
      histogramBins: 25,
      benchmarkDefaultMaxDocuments: 5000,
      benchmarkDefaultBatchSize: 64,
      benchmarkDefaultParallelism: 4,
    });
    page.form.markAsDirty();
    page.persistCurrentSettings();

    expect(store.save).toHaveBeenCalledWith(expect.objectContaining({
      datasets: expect.objectContaining({
        max_upload_bytes: 40 * 1024 * 1024,
        histogram_bins: 25,
      }),
      tokenizers: expect.objectContaining({ max_upload_bytes: 12 * 1024 * 1024 }),
      benchmarks: expect.objectContaining({
        default_max_documents: 5000,
        default_batch_size: 64,
        default_parallelism: 4,
      }),
    }));
  });

  it('debounces rapid control changes and autosaves the latest valid value', () => {
    vi.useFakeTimers();
    const { page, store } = createPage();

    page.form.controls['histogramBins'].setValue(31);
    vi.advanceTimersByTime(250);
    page.form.controls['histogramBins'].setValue(32);
    vi.advanceTimersByTime(399);
    expect(store.save).not.toHaveBeenCalled();

    vi.advanceTimersByTime(1);
    expect(store.save).toHaveBeenCalledTimes(1);
    expect(store.save).toHaveBeenCalledWith(expect.objectContaining({
      datasets: expect.objectContaining({ histogram_bins: 32 }),
    }));
  });

  it('persists a valid pending edit when leaving the settings page', () => {
    const { page, store } = createPage();

    page.form.controls['histogramBins'].setValue(33);
    page.form.markAsDirty();
    page.ngOnDestroy();

    expect(store.save).toHaveBeenCalledWith(expect.objectContaining({
      datasets: expect.objectContaining({ histogram_bins: 33 }),
    }));
  });

  it('resets overridden values through the autosave form and offers safe conflict reload', () => {
    const { page, store } = createPage();

    page.resetSetting('datasets.histogram_bins');
    expect(page.form.controls['histogramBins'].value).toBe(defaults.datasets.histogram_bins);
    expect(store.reset).not.toHaveBeenCalled();
    expect(store.save).toHaveBeenCalledTimes(1);
    expect(store.save).toHaveBeenCalledTimes(1);

    page.resetAll();
    expect(page.form.controls['histogramBins'].value).toBe(defaults.datasets.histogram_bins);
    expect(page.form.controls['benchmarkDefaultMaxDocuments'].value)
      .toBe(defaults.benchmarks.default_max_documents);
    expect(store.save).toHaveBeenCalledTimes(2);
    expect(store.save).toHaveBeenCalledTimes(2);

    store.conflict.set(true);
    page.reloadAfterConflict();
    expect(store.reload).toHaveBeenCalledTimes(1);
  });
});
