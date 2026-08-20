import { TestBed } from '@angular/core/testing';
import { Subject, of, throwError } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { TokenizersApiService } from '../api/tokenizers-api.service';
import type { TokenizerReportResponse } from '../api/api.models';
import { TokenizersStore } from './tokenizers.store';

const report = {
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
      latestReport: vi.fn().mockReturnValue(of(report)),
      generateReport: vi.fn().mockReturnValue(of(report)),
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

    expect(api.generateReport).toHaveBeenCalledWith(
      { tokenizer_name: 'CUSTOM_demo' },
      expect.any(Function),
    );
    expect(store.report()).toBe(report);
    expect(store.vocabulary()).toEqual({ report_id: 12, offset: 0, limit: 2, total: 1, items: [] });
    expect(store.jobProgress()).toBe(100);
  });

  it('surfaces partial download failures without losing completed progress', () => {
    const { api, store } = createStore();
    api.download.mockImplementation((
      _request: unknown,
      onUpdate: (status: { progress: number }) => void,
    ) => {
      onUpdate({ progress: 65 });
      return of({ failed: ['broken/model'] });
    });

    store.download({ tokenizers: ['good/model', 'broken/model'] });

    expect(store.downloadWarning()).toBe('Some tokenizers could not be downloaded: broken/model');
    expect(store.jobProgress()).toBe(100);
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
