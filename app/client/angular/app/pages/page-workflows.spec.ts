import { signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { FormGroup } from '@angular/forms';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ExportApiService } from '../core/api/export-api.service';
import { DatasetStore } from '../core/state/dataset.store';
import { TokenizersStore } from '../core/state/tokenizers.store';
import { BenchmarkStore } from '../core/state/benchmark.store';
import type { DatasetMetricCatalogCategory } from '../core/api/api.types';
import { DatasetPageComponent } from './dataset-page.component';
import { TokenizersPageComponent } from './tokenizers-page.component';
import { CrossBenchmarkPageComponent } from './cross-benchmark-page.component';

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
      downloadProgressVisible: () => boolean;
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
    expect(page.downloadProgressVisible()).toBe(false);
    store.busyAction.set('download');
    expect(page.downloadProgressVisible()).toBe(true);
  });
});

describe('CrossBenchmarkPageComponent workflow', () => {
  it('builds a trimmed benchmark request and applies widget visibility', () => {
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
    TestBed.configureTestingModule({ providers: [
      { provide: BenchmarkStore, useValue: store },
      { provide: ExportApiService, useValue: {} },
    ] });
    const page = TestBed.runInInjectionContext(() => new CrossBenchmarkPageComponent()) as unknown as {
      runForm: FormGroup;
      runSelectedTokenizers: { set: (value: readonly string[]) => void };
      runSelectedMetricKeys: { set: (value: readonly string[]) => void };
      runBenchmark: () => void;
      customizeDraft: { set: (value: readonly string[]) => void };
      applyCustomize: () => void;
    };

    page.runForm.patchValue({ dataset: '  custom/default  ', runName: '  quick run  ' });
    page.runSelectedTokenizers.set(['alpha', 'beta']);
    page.runSelectedMetricKeys.set(['eff.speed']);
    page.runBenchmark();

    expect(store.run).toHaveBeenCalledWith(expect.objectContaining({
      tokenizers: ['alpha', 'beta'],
      dataset_name: 'custom/default',
      run_name: 'quick run',
      selected_metric_keys: ['eff.speed'],
    }));

    page.customizeDraft.set(['visible']);
    page.applyCustomize();
    expect(store.setHiddenWidgetIds).toHaveBeenCalledWith(['hidden']);
  });
});
