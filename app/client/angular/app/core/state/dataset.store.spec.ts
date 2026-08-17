import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DatasetsApiService } from '../api/datasets-api.service';
import type { DatasetAnalysisResponse } from '../api/api.types';
import { DatasetStore } from './dataset.store';

const report = {
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
      analyze: vi.fn((
        _request: unknown,
        onUpdate: (status: { progress: number }) => void,
      ) => {
        onUpdate({ progress: 45 });
        return of(report);
      }),
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

  it('publishes analysis progress, persists the report, and resets failures', () => {
    const { api, store } = createStore();

    store.analyze({ dataset_name: 'custom/demo' } as never);

    expect(api.analyze).toHaveBeenCalled();
    expect(store.report()).toBe(report);
    expect(store.jobProgress()).toBe(100);
    expect(store.busyAction()).toBeNull();
    expect(JSON.parse(localStorage.getItem('tkben:last-dataset-report') ?? '{}')).toEqual(report);

    api.analyze.mockReturnValue(throwError(() => new Error('analysis failed')));
    store.analyze({ dataset_name: 'custom/demo' } as never);

    expect(store.error()).toBe('analysis failed');
    expect(store.jobProgress()).toBeNull();
    expect(store.busyAction()).toBeNull();
  });
});
