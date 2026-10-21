// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { of, Subject, throwError } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { BenchmarksApiService } from '../api/benchmarks-api.service';
import { DatasetsApiService } from '../api/datasets-api.service';
import { JobsApiService } from '../api/jobs-api.service';
import { TokenizersApiService } from '../api/tokenizers-api.service';
import type {
  BenchmarkRunResponse,
  DashboardUiPreferencesResponse,
  DashboardUiPreferencesUpdate,
} from '../api/api.models';
import { BenchmarkStore } from './benchmark.store';

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
      run: vi.fn((
        _request: unknown,
        onUpdate: (status: { progress: number }) => void,
        onJobStart: (job: { job_id: string }) => void,
      ) => {
        onJobStart({ job_id: 'job-5' });
        onUpdate({ progress: 60 });
        return of(report);
      }),
    };
  }

  function createStore(api = createApi()) {
    const datasetsApi = { list: vi.fn().mockReturnValue(of({ datasets: [{ dataset_name: 'custom/demo' }] })) };
    const tokenizersApi = { list: vi.fn().mockReturnValue(of({ tokenizers: [{ tokenizer_name: 'CUSTOM_demo' }] })) };
    const jobsApi = { cancel: vi.fn().mockReturnValue(of({})) };
    TestBed.configureTestingModule({ providers: [
      { provide: BenchmarksApiService, useValue: api },
      { provide: DatasetsApiService, useValue: datasetsApi },
      { provide: TokenizersApiService, useValue: tokenizersApi },
      { provide: JobsApiService, useValue: jobsApi },
    ] });
    const store = TestBed.inject(BenchmarkStore);
    vi.advanceTimersByTime(250);
    return { api, datasetsApi, tokenizersApi, jobsApi, store };
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

  it('propagates job progress and refreshes after success', () => {
    const { api, jobsApi, store } = createStore();

    store.run({ dataset_name: 'custom/demo' } as never);
    vi.advanceTimersByTime(250);

    expect(store.report()).toBe(report);
    expect(store.progress()).toBe(100);
    expect(store.activeJobId()).toBeNull();
    expect(store.busy()).toBe(false);
    expect(api.reports).toHaveBeenCalledTimes(2);

    api.run.mockImplementation(() => throwError(() => new Error('benchmark failed')));
    store.run({ dataset_name: 'custom/demo' } as never);
    expect(store.error()).toBe('benchmark failed');
    expect(store.progress()).toBeNull();

    store.cancel();
    expect(jobsApi.cancel).not.toHaveBeenCalled();
  });

  it('requests cancellation once and allows an immediate rerun after cancellation', () => {
    const api = createApi();
    const { jobsApi, store } = createStore(api);
    const pendingRun = new Subject<BenchmarkRunResponse>();
    api.run.mockImplementation((_request, _onUpdate, onJobStart) => {
      onJobStart({ job_id: 'job-cancelled' });
      return pendingRun;
    });

    store.run({ dataset_name: 'custom/demo' } as never);
    expect(store.busy()).toBe(true);
    expect(store.activeJobId()).toBe('job-cancelled');

    store.cancel();
    store.cancel();
    expect(jobsApi.cancel).toHaveBeenCalledTimes(1);
    expect(jobsApi.cancel).toHaveBeenCalledWith('job-cancelled');
    expect(store.cancellationRequested()).toBe(true);

    pendingRun.error(new Error('Job was cancelled.'));
    expect(store.busy()).toBe(false);
    expect(store.activeJobId()).toBeNull();
    expect(store.cancellationRequested()).toBe(false);
    expect(store.error()).toBe('Job was cancelled.');

    api.run.mockImplementation((_request, _onUpdate, onJobStart) => {
      onJobStart({ job_id: 'job-rerun' });
      return of(report);
    });
    store.run({ dataset_name: 'custom/demo' } as never);

    expect(store.busy()).toBe(false);
    expect(store.activeJobId()).toBeNull();
    expect(store.error()).toBeNull();
    expect(store.report()).toBe(report);
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
