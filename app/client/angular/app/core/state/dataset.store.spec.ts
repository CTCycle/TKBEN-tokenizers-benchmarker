import { TestBed } from '@angular/core/testing';
import { Subject, of, throwError } from 'rxjs';
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
