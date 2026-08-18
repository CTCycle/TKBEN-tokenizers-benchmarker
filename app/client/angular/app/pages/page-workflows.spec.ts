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
});

describe('TokenizersPageComponent workflow', () => {
  it('parses manual downloads and maps catalog filters', () => {
    const store = {
      report: signal(null),
      vocabulary: signal(null),
      discoveryResults: signal([{ identifier: 'alpha/model' }, { identifier: 'beta/model' }]),
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
