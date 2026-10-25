// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { of, Subject, throwError } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { BenchmarksApiService } from '../api/benchmarks-api.service';
import { DatasetsApiService } from '../api/datasets-api.service';
import { TokenizersApiService } from '../api/tokenizers-api.service';
import type { BenchmarkRunResponse, DashboardUiPreferencesResponse, DashboardUiPreferencesUpdate, DatasetAnalysisResponse, RuntimeSettingsResponse, TokenizerReportResponse } from '../api/api.models';
import { BenchmarkStore } from './benchmark.store';
import { DatasetStore } from './dataset.store';
import { SettingsApiService } from '../api/settings-api.service';
import { SettingsStore } from './settings.store';
import { TokenizersStore } from './tokenizers.store';


const report = {
  status: 'success',
  report_id: 5,
  dashboard: {
    widgets: [
      {
        widget_id: 'visible', default_visible: true,
        points: [{ tokenizer: 'alpha', value: 100 }],
        distributions: [], buckets: [], histogram_bins: [],
      },
      { widget_id: 'hidden', default_visible: false, points: [], distributions: [], buckets: [], histogram_bins: [] },
    ],
  },
} as unknown as BenchmarkRunResponse;

describe('BenchmarkStore', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    TestBed.resetTestingModule();
    vi.useRealTimers();
  });

  function createApi() {
    return {
      dashboardPreferences: vi.fn().mockReturnValue(of({
        revision: 0,
        preferences: {
          ordered_widget_ids: [],
          hidden_widget_ids: [],
          visualization_by_widget_id: {},
          baseline_by_report_id: {},
          tokenizer_colors: {},
        },
      })),
      saveDashboardPreferences: vi.fn((request: DashboardUiPreferencesUpdate) => of({
        revision: request.expected_revision + 1,
        preferences: request.preferences,
      })),
      reports: vi.fn().mockReturnValue(of({ reports: [{ report_id: 5 }], total: 1, offset: 0, limit: 25 })),
      metricsCatalog: vi.fn().mockReturnValue(of({ categories: [] })),
      report: vi.fn().mockReturnValue(of(report)),
      deleteReport: vi.fn().mockReturnValue(of(undefined)),
      updateReportTags: vi.fn((reportId: number, tags: readonly string[]) => of({ report_id: reportId, tags: [...tags] })),
      run: vi.fn().mockReturnValue(of(report)),
    };
  }

  function createStore(api = createApi()) {
    const datasetsApi = { list: vi.fn().mockReturnValue(of({ datasets: [{ dataset_name: 'custom/demo' }] })) };
    const tokenizersApi = { list: vi.fn().mockReturnValue(of({ tokenizers: [{ tokenizer_name: 'CUSTOM_demo' }] })) };
    TestBed.configureTestingModule({ providers: [
      { provide: BenchmarksApiService, useValue: api },
      { provide: DatasetsApiService, useValue: datasetsApi },
      { provide: TokenizersApiService, useValue: tokenizersApi },
    ] });
    const store = TestBed.inject(BenchmarkStore);
    vi.advanceTimersByTime(250);
    return { api, datasetsApi, tokenizersApi, store };
  }

  it('loads workspace metadata and initializes layout from the first report', () => {
    const { store } = createStore();

    expect(store.availableDatasets()).toEqual(['custom/demo']);
    expect(store.availableTokenizers()).toEqual(['CUSTOM_demo']);
    expect(store.selectedReportId()).toBe(5);
    expect(store.layout()).toEqual(['visible', 'hidden']);
    expect(store.hiddenWidgetIds()).toEqual(['hidden']);
    expect(store.reportsLoading()).toBe(false);
  });

  it('refreshes after a successful run and surfaces failures', () => {
    const { api, store } = createStore();

    store.run({ dataset_name: 'custom/demo' } as never);
    vi.advanceTimersByTime(250);

    expect(store.report()).toBe(report);
    expect(api.reports).toHaveBeenCalledTimes(2);

    api.run.mockImplementation(() => throwError(() => new Error('benchmark failed')));
    store.run({ dataset_name: 'custom/demo' } as never);
    expect(store.error()).toBe('benchmark failed');
  });

  it('shows the report of the run that completes last, regardless of start order', () => {
    const { api, store } = createStore();
    const first = new Subject<typeof report>();
    const second = new Subject<typeof report>();
    const firstReport = { ...report, report_id: 11 };
    const secondReport = { ...report, report_id: 12 };
    api.run.mockReturnValueOnce(first).mockReturnValueOnce(second);

    store.run({ dataset_name: 'a' } as never);
    store.run({ dataset_name: 'b' } as never);
    second.next(secondReport);
    first.next(firstReport);

    expect(store.report()).toBe(firstReport);
    expect(store.selectedReportId()).toBe(11);
  });

  it('persists dashboard layout, hidden IDs, and visualizations in one snapshot', () => {
    const { api, store } = createStore();
    store.setVisualization('visible', 'bar');

    store.setHiddenWidgetIds(['hidden', 'hidden']);
    store.reorder(0, 1);

    expect(store.hiddenWidgetIds()).toEqual(['hidden']);
    expect(api.saveDashboardPreferences).toHaveBeenLastCalledWith({
      expected_revision: 2,
      preferences: {
        ordered_widget_ids: ['hidden', 'visible'],
        hidden_widget_ids: ['hidden'],
        visualization_by_widget_id: { visible: 'bar' },
        baseline_by_report_id: {},
        tokenizer_colors: {},
      },
    });
  });

  it('serializes rapid dashboard edits and writes the newest snapshot', () => {
    const api = createApi();
    const firstWrite = new Subject<DashboardUiPreferencesResponse>();
    api.saveDashboardPreferences.mockReturnValueOnce(firstWrite).mockReturnValueOnce(of({
      revision: 2,
      preferences: {
        ordered_widget_ids: ['visible', 'hidden'],
        hidden_widget_ids: ['hidden'],
        visualization_by_widget_id: { visible: 'horizontal_bar' as const },
        baseline_by_report_id: {},
        tokenizer_colors: {},
      },
    }));
    const { api: storeApi, store } = createStore(api);

    store.setVisualization('visible', 'bar');
    store.setHiddenWidgetIds(['hidden']);
    store.setVisualization('visible', 'horizontal_bar');
    expect(storeApi.saveDashboardPreferences).toHaveBeenCalledTimes(1);

    firstWrite.next({
      revision: 1,
      preferences: {
        ordered_widget_ids: ['visible', 'hidden'],
        hidden_widget_ids: [],
        visualization_by_widget_id: { visible: 'bar' },
        baseline_by_report_id: {},
        tokenizer_colors: {},
      },
    });

    expect(storeApi.saveDashboardPreferences).toHaveBeenCalledTimes(2);
    expect(storeApi.saveDashboardPreferences).toHaveBeenLastCalledWith({
      expected_revision: 1,
      preferences: {
        ordered_widget_ids: ['visible', 'hidden'],
        hidden_widget_ids: ['hidden'],
        visualization_by_widget_id: { visible: 'horizontal_bar' },
        baseline_by_report_id: {},
        tokenizer_colors: {},
      },
    });
    expect(store.preferencesSaving()).toBe(false);
    expect(store.preferencesSaved()).toBe(true);
  });

  it('keeps the immediate dashboard change visible and reports a failed write', () => {
    const api = createApi();
    api.saveDashboardPreferences.mockReturnValue(throwError(() => ({
      status: 503,
      error: { detail: 'Database is read-only.' },
    })));
    const { store } = createStore(api);

    store.setVisualization('visible', 'bar');

    expect(store.visualizations()).toEqual({ visible: 'bar' });
    expect(store.preferencesSaved()).toBe(false);
    expect(store.preferencesSaving()).toBe(false);
    expect(store.preferencesError()).toBe('Database is read-only.');
  });

  it('loads dashboard preferences from the settings API before the first report', () => {
    const api = createApi();
    api.dashboardPreferences.mockReturnValue(of({
      revision: 3,
      preferences: {
        ordered_widget_ids: ['hidden', 'visible'],
        hidden_widget_ids: ['visible'],
        visualization_by_widget_id: { visible: 'horizontal_bar' },
        baseline_by_report_id: {},
        tokenizer_colors: {},
      },
    }));
    const { store } = createStore(api);

    expect(store.layout()).toEqual(['hidden', 'visible']);
    expect(store.hiddenWidgetIds()).toEqual(['visible']);
    expect(store.visualizations()).toEqual({ visible: 'horizontal_bar' });
  });

  it('reorders visible widgets without moving hidden panel slots', () => {
    const { store } = createStore();
    store.layout.set(['first', 'hidden-a', 'second', 'hidden-b', 'third']);
    store.hiddenWidgetIds.set(['hidden-a', 'hidden-b']);

    store.reorderVisible(0, 1);

    expect(store.layout()).toEqual(['second', 'hidden-a', 'first', 'hidden-b', 'third']);
  });

  it('persists and restores report-specific baselines while switching reports', () => {
    const reportFive = {
      ...report,
      dashboard: {
        widgets: [{
          widget_id: 'speed', default_visible: true,
          points: [
            { tokenizer: 'alpha', value: 100 },
            { tokenizer: 'beta', value: 80 },
          ],
          distributions: [], buckets: [], histogram_bins: [],
        }],
      },
    } as unknown as BenchmarkRunResponse;
    const reportSix = {
      ...reportFive,
      report_id: 6,
      dashboard: {
        widgets: [{
          widget_id: 'speed', default_visible: true,
          points: [
            { tokenizer: 'gamma', value: 100 },
            { tokenizer: 'delta', value: 80 },
          ],
          distributions: [], buckets: [], histogram_bins: [],
        }],
      },
    } as unknown as BenchmarkRunResponse;
    const api = createApi();
    api.report.mockImplementation((reportId: number) => of(reportId === 5 ? reportFive : reportSix));
    const { store } = createStore(api);

    store.setBaseline('alpha');
    expect(store.baselineTokenizer()).toBe('alpha');
    expect(api.saveDashboardPreferences).toHaveBeenLastCalledWith(expect.objectContaining({
      preferences: expect.objectContaining({ baseline_by_report_id: { '5': 'alpha' } }),
    }));

    store.selectReport(6);
    expect(store.baselineTokenizer()).toBeNull();
    store.setBaseline('gamma');
    store.selectReport(5);
    expect(store.baselineTokenizer()).toBe('alpha');
    store.selectReport(6);
    expect(store.baselineTokenizer()).toBe('gamma');
  });

  it('restores a valid report-specific baseline from the active database', () => {
    const api = createApi();
    api.dashboardPreferences.mockReturnValue(of({
      revision: 2,
      preferences: {
        ordered_widget_ids: [],
        hidden_widget_ids: [],
        visualization_by_widget_id: {},
        baseline_by_report_id: { '5': 'alpha' },
        tokenizer_colors: {},
      },
    }));
    const { store } = createStore(api);

    expect(store.baselineTokenizer()).toBe('alpha');
  });

  it('clears an invalid report baseline from persisted preferences', () => {
    const api = createApi();
    api.dashboardPreferences.mockReturnValue(of({
      revision: 2,
      preferences: {
        ordered_widget_ids: [],
        hidden_widget_ids: [],
        visualization_by_widget_id: {},
        baseline_by_report_id: { '5': 'unavailable' },
        tokenizer_colors: {},
      },
    }));
    const { store } = createStore(api);
    expect(store.baselineTokenizer()).toBeNull();
    expect(api.saveDashboardPreferences).toHaveBeenLastCalledWith(expect.objectContaining({
      expected_revision: 2,
      preferences: expect.objectContaining({ baseline_by_report_id: {} }),
    }));
  });

  it('removes a deleted report baseline and updates tags without reloading the report', () => {
    const { api, store } = createStore();
    store.setBaseline('alpha');
    expect(store.baselineTokenizer()).toBe('alpha');
    store.updateReportTags(5, ['Production', 'cpu']);

    expect(store.reports()[0]?.tags).toEqual(['Production', 'cpu']);
    expect(store.report()?.tags).toEqual(['Production', 'cpu']);
    expect(api.report).toHaveBeenCalledTimes(1);

    store.deleteReport(5);
    expect(api.saveDashboardPreferences).toHaveBeenLastCalledWith(expect.objectContaining({
      preferences: expect.objectContaining({ baseline_by_report_id: {} }),
    }));
  });

  it('leaves existing tags unchanged when the tag API fails', () => {
    const api = createApi();
    api.updateReportTags.mockImplementation(() => throwError(() => new Error('tag update failed')));
    const { store } = createStore(api);
    store.updateReportTags(5, ['new']);

    expect(store.reports()[0]?.tags).toBeUndefined();
    expect(store.report()?.tags).toBeUndefined();
    expect(store.error()).toBe('tag update failed');
    expect(store.updatingReportTagsId()).toBeNull();
  });
});

const report__dataset_store = {
  status: 'success',
  dataset_name: 'custom/demo',
  report_id: 7,
  aggregate_statistics: {},
} as unknown as DatasetAnalysisResponse;

describe('DatasetStore', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    TestBed.resetTestingModule();
    vi.useRealTimers();
    localStorage.clear();
  });

  function createApi() {
    return {
      list: vi.fn().mockReturnValue(of({ datasets: [{ dataset_name: 'custom/demo' }], count: 1 })),
      metricsCatalog: vi.fn().mockReturnValue(of({ categories: [] })),
      latestReport: vi.fn().mockReturnValue(of(null)),
      analyze: vi.fn().mockReturnValue(of(report__dataset_store)),
      download: vi.fn(),
      upload: vi.fn(),
      delete: vi.fn(),
    };
  }

  function createStore(api = createApi()) {
    TestBed.configureTestingModule({ providers: [
      { provide: DatasetsApiService, useValue: api },
    ] });
    const store = TestBed.inject(DatasetStore);
    vi.advanceTimersByTime(250);
    return { api, store };
  }

  it('debounces and normalizes catalog refreshes, clearing invalid selection', () => {
    const { api, store } = createStore();
    store.select('custom/missing');
    store.refresh({ search: '  demo  ' });

    vi.advanceTimersByTime(249);
    expect(api.list).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(1);

    expect(api.list).toHaveBeenLastCalledWith({ search: 'demo' });
    expect(store.datasets()).toEqual([{ dataset_name: 'custom/demo' }]);
    expect(store.selectedDataset()).toBeNull();
    expect(store.loading()).toBe(false);
  });

  it('suppresses only the expected missing-report error for row selection', () => {
    const { api, store } = createStore();
    api.latestReport.mockReturnValue(throwError(() => new Error('No validation report found')));

    store.loadLatest('custom/demo', { suppressNotFoundError: true });
    expect(store.error()).toBeNull();

    store.loadLatest('custom/demo');
    expect(store.error()).toBe('No validation report found');
    expect(store.busyAction()).toBeNull();
  });

  it('stores the analysis report in memory and resets failures', () => {
    const { api, store } = createStore();

    store.analyze({ dataset_name: 'custom/demo' } as never);

    expect(api.analyze).toHaveBeenCalled();
    expect(store.report()).toBe(report__dataset_store);
    expect(store.busyAction()).toBeNull();
    expect(localStorage.length).toBe(0);

    api.analyze.mockReturnValue(throwError(() => new Error('analysis failed')));
    store.analyze({ dataset_name: 'custom/demo' } as never);

    expect(store.error()).toBe('analysis failed');
    expect(store.busyAction()).toBeNull();
  });

  it('runs analyses concurrently and shows the one that completes last', () => {
    const { api, store } = createStore();
    const first = new Subject<typeof report__dataset_store>();
    const second = new Subject<typeof report__dataset_store>();
    const firstReport = { ...report__dataset_store, dataset_name: 'custom/a' };
    const secondReport = { ...report__dataset_store, dataset_name: 'custom/b' };
    api.analyze.mockReturnValueOnce(first).mockReturnValueOnce(second);

    store.analyze({ dataset_name: 'custom/a' } as never);
    store.analyze({ dataset_name: 'custom/b' } as never);
    expect(store.analyzingDatasets()).toEqual(['custom/a', 'custom/b']);

    second.next(secondReport);
    expect(store.report()).toBe(secondReport);
    expect(store.analyzingDatasets()).toEqual(['custom/a']);

    first.next(firstReport);
    expect(store.report()).toBe(firstReport);
    expect(store.analyzingDatasets()).toEqual([]);
  });

  it('removes a dataset immediately, clears its report, and ignores duplicate deletes', () => {
    const api = createApi();
    let authoritative = [{ dataset_name: 'custom/demo' }, { dataset_name: 'custom/other' }];
    api.list.mockImplementation(() => of({ datasets: authoritative, count: authoritative.length }));
    const { store } = createStore(api);
    store.analyze({ dataset_name: 'custom/demo' } as never);

    const deletion = new Subject<void>();
    api.delete.mockReturnValue(deletion.asObservable());
    store.remove('custom/demo');
    store.remove('custom/demo');

    expect(api.delete).toHaveBeenCalledTimes(1);
    expect(store.busyAction()).toBe('remove:custom/demo');

    authoritative = [{ dataset_name: 'custom/other' }];
    deletion.next();
    deletion.complete();

    expect(store.datasets()).toEqual([{ dataset_name: 'custom/other' }]);
    expect(store.selectedDataset()).toBeNull();
    expect(store.report()).toBeNull();
    expect(store.busyAction()).toBeNull();
  });

  it('forces a refetch using the active catalog filters after deletion', () => {
    const api = createApi();
    let authoritative = [{ dataset_name: 'custom/demo' }];
    api.list.mockImplementation((filters: { search?: string; source?: string }) => of({ datasets: authoritative, count: authoritative.length, filters }));
    const { store } = createStore(api);

    store.refresh({ search: ' demo ', source: 'custom' });
    vi.advanceTimersByTime(250);
    expect(api.list).toHaveBeenLastCalledWith({ search: 'demo', source: 'custom' });

    api.delete.mockReturnValue(of({ status: 'success', dataset_name: 'custom/demo', message: 'removed' }));
    authoritative = [];
    store.remove('custom/demo');

    expect(api.list).toHaveBeenLastCalledWith({ search: 'demo', source: 'custom' });
    expect(store.datasets()).toEqual([]);
  });
});

const response: RuntimeSettingsResponse = {
  revision: 2,
  settings: {
    tokenizers: {
      default_discovery_limit: 40,
      max_discovery_limit: 200,
      max_discovery_candidates: 600,
      metadata_candidate_multiplier: 3,
      max_upload_bytes: 10 * 1024 * 1024,
    },
    datasets: {
      histogram_bins: 30,
      streaming_batch_size: 12_000,
      max_upload_bytes: 25 * 1024 * 1024,
      download_timeout_seconds: 180,
      download_retry_attempts: 3,
      download_retry_backoff_seconds: 2,
    },
    benchmarks: {
      default_max_documents: 2500,
      default_batch_size: 32,
      default_parallelism: 2,
      streaming_batch_size: 1000,
    },
    jobs: { polling_interval: 2, max_stored_jobs: 1000 },
  },
  defaults: {
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
    jobs: { polling_interval: 1, max_stored_jobs: 1000 },
  },
  overridden_keys: [
    'datasets.histogram_bins',
    'datasets.streaming_batch_size',
    'benchmarks.default_max_documents',
    'benchmarks.default_batch_size',
    'benchmarks.default_parallelism',
    'jobs.polling_interval',
    'jobs.max_stored_jobs',
  ],
  warning: null,
};

describe('SettingsStore', () => {
  afterEach(() => TestBed.resetTestingModule());

  function createApi() {
    return {
      get: vi.fn().mockReturnValue(of(response)),
      patch: vi.fn().mockReturnValue(of(response)),
      reset: vi.fn().mockReturnValue(of(response)),
    };
  }

  function createStore(api = createApi()) {
    TestBed.configureTestingModule({ providers: [
      { provide: SettingsApiService, useValue: api },
    ] });
    return { api, store: TestBed.inject(SettingsStore) };
  }

  it('loads typed server state and tracks overridden keys', () => {
    const { api, store } = createStore();
    store.load();
    store.load();

    expect(api.get).toHaveBeenCalledTimes(1);
    expect(store.settings()?.datasets.histogram_bins).toBe(30);
    expect(store.defaults()?.datasets.histogram_bins).toBe(20);
    expect(store.settings()?.benchmarks.default_max_documents).toBe(2500);
    expect(store.revision()).toBe(2);
    expect(store.isOverridden('datasets.histogram_bins')).toBe(true);
    expect(store.isOverridden('benchmarks.default_batch_size')).toBe(true);
    expect(store.isOverridden('benchmarks.streaming_batch_size')).toBe(false);
    expect(store.loading()).toBe(false);
  });

  it('sends the current revision and replaces local state with the response', () => {
    const api = createApi();
    const serverResponse = { ...response, revision: 3, settings: {
      ...response.settings,
      jobs: { polling_interval: 4, max_stored_jobs: 1000 },
    }, overridden_keys: ['jobs.polling_interval'] as const };
    api.get.mockReturnValue(of(response));
    api.patch.mockReturnValue(of(serverResponse));
    const { store } = createStore(api);
    store.load();

    store.save({ jobs: { polling_interval: 4.5 } });

    expect(api.patch).toHaveBeenCalledWith({
      expected_revision: 2,
      jobs: { polling_interval: 4.5 },
    });
    expect(store.revision()).toBe(3);
    expect(store.settings()?.jobs.polling_interval).toBe(4);
    expect(store.saving()).toBe(false);
    expect(store.saveStatus()).toBe('saved');
  });

  it('serializes rapid edits and sends the newest snapshot at the updated revision', () => {
    const api = createApi();
    const firstWrite = new Subject<RuntimeSettingsResponse>();
    const secondResponse: RuntimeSettingsResponse = {
      ...response,
      revision: 4,
      settings: { ...response.settings, jobs: { polling_interval: 3.5, max_stored_jobs: 1000 } },
    };
    api.patch.mockReturnValueOnce(firstWrite).mockReturnValueOnce(of(secondResponse));
    const { store } = createStore(api);
    store.load();

    store.save({ jobs: { polling_interval: 2.5 } });
    store.save({ jobs: { polling_interval: 3.5 } });
    expect(api.patch).toHaveBeenCalledTimes(1);

    firstWrite.next({
      ...response,
      revision: 3,
      settings: { ...response.settings, jobs: { polling_interval: 2.5, max_stored_jobs: 1000 } },
    });

    expect(api.patch).toHaveBeenCalledTimes(2);
    expect(api.patch).toHaveBeenLastCalledWith({
      expected_revision: 3,
      jobs: { polling_interval: 3.5 },
    });
    expect(store.settings()?.jobs.polling_interval).toBe(3.5);
    expect(store.revision()).toBe(4);
    expect(store.saving()).toBe(false);
    expect(store.saveStatus()).toBe('saved');
  });

  it('keeps persisted state unchanged and reports a failed write', () => {
    const api = createApi();
    api.patch.mockReturnValue(throwError(() => ({
      status: 503,
      error: { detail: 'Database is read-only.' },
    })));
    const { store } = createStore(api);
    store.load();

    store.save({ jobs: { polling_interval: 3 } });

    expect(store.settings()?.jobs.polling_interval).toBe(response.settings.jobs.polling_interval);
    expect(store.saveStatus()).toBe('error');
    expect(store.error()).toBe('Database is read-only.');
    expect(store.saving()).toBe(false);
  });

  it('resets through the API and handles errors and conflicts', () => {
    const api = createApi();
    api.reset.mockReturnValue(of(response));
    const { store } = createStore(api);
    store.load();
    store.reset(['datasets.histogram_bins']);

    expect(api.reset).toHaveBeenCalledWith({
      expected_revision: 2,
      keys: ['datasets.histogram_bins'],
    });

    api.patch.mockReturnValue(throwError(() => ({ status: 409, error: { detail: 'stale' } })));
    store.save({ jobs: { polling_interval: 3 } });
    expect(store.conflict()).toBe(true);
    expect(store.error()).toBe('Settings changed elsewhere. Reload before trying again.');
    expect(store.saving()).toBe(false);

    api.get.mockReturnValue(throwError(() => new Error('settings unavailable')));
    store.reload();
    expect(store.error()).toBe('settings unavailable');
  });
});

const report__tokenizers_store = {
  status: 'success',
  report_id: 12,
  tokenizer_name: 'CUSTOM_demo',
} as unknown as TokenizerReportResponse;

describe('TokenizersStore', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    TestBed.resetTestingModule();
    vi.useRealTimers();
    localStorage.clear();
  });

  function createApi() {
    return {
      list: vi.fn().mockReturnValue(of({ tokenizers: [{ tokenizer_name: 'CUSTOM_demo' }], count: 1 })),
      discover: vi.fn().mockReturnValue(of({ items: [{ identifier: 'alpha/model' }], count: 1, fetched_count: 1 })),
      latestReport: vi.fn().mockReturnValue(of(report__tokenizers_store)),
      generateReport: vi.fn().mockReturnValue(of(report__tokenizers_store)),
      vocabularyPage: vi.fn().mockReturnValue(of({ report_id: 12, offset: 0, limit: 2, total: 1, items: [] })),
      download: vi.fn(),
      upload: vi.fn(),
      delete: vi.fn(),
    };
  }

  function createStore(api = createApi()) {
    TestBed.configureTestingModule({ providers: [
      { provide: TokenizersApiService, useValue: api },
    ] });
    const store = TestBed.inject(TokenizersStore);
    vi.advanceTimersByTime(250);
    return { api, store };
  }

  it('opens a generated report only when no persisted report exists', () => {
    const { api, store } = createStore();
    api.latestReport.mockReturnValueOnce(of(null));

    store.openReport('CUSTOM_demo');

    expect(api.generateReport).toHaveBeenCalledWith({ tokenizer_name: 'CUSTOM_demo' });
    expect(store.report()).toBe(report__tokenizers_store);
    expect(store.vocabulary()).toEqual({ report_id: 12, offset: 0, limit: 2, total: 1, items: [] });
  });

  it('generates reports concurrently and shows the one that completes last', () => {
    const { api, store } = createStore();
    const first = new Subject<typeof report__tokenizers_store>();
    const second = new Subject<typeof report__tokenizers_store>();
    const firstReport = { ...report__tokenizers_store, tokenizer_name: 'a' };
    const secondReport = { ...report__tokenizers_store, tokenizer_name: 'b' };
    api.generateReport.mockReturnValueOnce(first).mockReturnValueOnce(second);

    store.generateReport('a');
    store.generateReport('b');
    expect(store.reportingTokenizers()).toEqual(['a', 'b']);

    second.next(secondReport);
    first.next(firstReport);

    expect(store.report()).toBe(firstReport);
    expect(store.reportingTokenizers()).toEqual([]);
  });

  it('surfaces partial download failures', () => {
    const { api, store } = createStore();
    api.download.mockReturnValue(of({ failed: ['broken/model'] }));

    store.download({ tokenizers: ['good/model', 'broken/model'] });

    expect(store.downloadWarning()).toBe('Some tokenizers could not be downloaded: broken/model');
    expect(store.busyAction()).toBeNull();
  });

  it('loads vocabulary pages from the active report and reports API failures', () => {
    const { api, store } = createStore();
    store.openReport('CUSTOM_demo');
    store.loadVocabulary(4, 10);

    expect(api.vocabularyPage).toHaveBeenLastCalledWith(12, 4, 10);

    api.vocabularyPage.mockReturnValue(throwError(() => new Error('vocabulary failed')));
    store.loadVocabulary(14, 10);
    expect(store.error()).toBe('vocabulary failed');
  });

  it('removes a custom tokenizer immediately, clears report state, and ignores duplicate deletes', () => {
    const api = createApi();
    let authoritative = [{ tokenizer_name: 'CUSTOM_demo' }];
    api.list.mockImplementation(() => of({ tokenizers: authoritative, count: authoritative.length }));
    const { store } = createStore(api);
    store.openReport('CUSTOM_demo');

    const deletion = new Subject<void>();
    api.delete.mockReturnValue(deletion.asObservable());
    store.remove('CUSTOM_demo');
    store.remove('CUSTOM_demo');

    expect(api.delete).toHaveBeenCalledTimes(1);
    expect(store.busyAction()).toBe('remove:CUSTOM_demo');

    authoritative = [];
    deletion.next();
    deletion.complete();

    expect(store.tokenizers()).toEqual([]);
    expect(store.selectedTokenizer()).toBeNull();
    expect(store.report()).toBeNull();
    expect(store.vocabulary()).toBeNull();
    expect(store.busyAction()).toBeNull();
  });

  it('forces a refetch using the active catalog filters after tokenizer deletion', () => {
    const api = createApi();
    let authoritative = [{ tokenizer_name: 'CUSTOM_demo' }];
    api.list.mockImplementation((filters: { search?: string; source?: string }) => of({ tokenizers: authoritative, count: authoritative.length, filters }));
    const { store } = createStore(api);

    store.refresh({ search: ' demo ', source: 'custom' });
    vi.advanceTimersByTime(250);
    expect(api.list).toHaveBeenLastCalledWith({ search: 'demo', source: 'custom' });

    api.delete.mockReturnValue(of(undefined));
    authoritative = [];
    store.remove('CUSTOM_demo');

    expect(api.list).toHaveBeenLastCalledWith({ search: 'demo', source: 'custom' });
    expect(store.tokenizers()).toEqual([]);
  });
});
