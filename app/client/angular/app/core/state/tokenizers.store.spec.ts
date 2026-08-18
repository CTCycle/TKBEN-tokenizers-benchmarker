import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { TokenizersApiService } from '../api/tokenizers-api.service';
import type { TokenizerReportResponse } from '../api/api.types';
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
});
