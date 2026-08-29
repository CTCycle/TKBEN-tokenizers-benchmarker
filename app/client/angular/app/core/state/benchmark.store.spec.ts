import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { BenchmarksApiService } from '../api/benchmarks-api.service';
import { DatasetsApiService } from '../api/datasets-api.service';
import { JobsApiService } from '../api/jobs-api.service';
import { TokenizersApiService } from '../api/tokenizers-api.service';
import type { BenchmarkRunResponse } from '../api/api.models';
import { BenchmarkStore } from './benchmark.store';

const report = {
  status: 'success',
  report_id: 5,
  dashboard: {
    widgets: [
      { widget_id: 'visible', default_visible: true },
      { widget_id: 'hidden', default_visible: false },
    ],
  },
} as unknown as BenchmarkRunResponse;

describe('BenchmarkStore', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
  });

  afterEach(() => {
    TestBed.resetTestingModule();
    vi.useRealTimers();
    localStorage.clear();
  });

  function createApi() {
    return {
      reports: vi.fn().mockReturnValue(of({ reports: [{ report_id: 5 }], total: 1, offset: 0, limit: 25 })),
      metricsCatalog: vi.fn().mockReturnValue(of({ categories: [] })),
      report: vi.fn().mockReturnValue(of(report)),
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

  it('propagates job progress, refreshes after success, and cancels the active job', () => {
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

  it('deduplicates hidden widgets and preserves visualization settings while reordering', () => {
    const { store } = createStore();
    store.setVisualization('visible', 'bar');

    store.setHiddenWidgetIds(['hidden', 'hidden']);
    store.reorder(0, 1);

    const saved = JSON.parse(localStorage.getItem('tkben:cross-benchmark-dashboard-layout:v3') ?? '{}');
    expect(store.hiddenWidgetIds()).toEqual(['hidden']);
    expect(saved.ordered_widget_ids).toEqual(['hidden', 'visible']);
    expect(saved.visualization_by_widget_id).toEqual({ visible: 'bar' });
  });

  it('ignores the old array preference shape', () => {
    localStorage.setItem(
      'tkben:cross-benchmark-dashboard-layout:v3',
      JSON.stringify(['legacy-only']),
    );
    const { store } = createStore();

    expect(store.layout()).toEqual(['visible', 'hidden']);
  });

  it('reorders visible widgets without moving hidden panel slots', () => {
    const { store } = createStore();
    store.layout.set(['first', 'hidden-a', 'second', 'hidden-b', 'third']);
    store.hiddenWidgetIds.set(['hidden-a', 'hidden-b']);

    store.reorderVisible(0, 1);

    expect(store.layout()).toEqual(['second', 'hidden-a', 'first', 'hidden-b', 'third']);
  });
});
