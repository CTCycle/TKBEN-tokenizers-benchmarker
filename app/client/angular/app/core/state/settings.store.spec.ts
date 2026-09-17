import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { SettingsApiService } from '../api/settings-api.service';
import type { RuntimeSettingsResponse } from '../api/api.models';
import { SettingsStore } from './settings.store';

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
    jobs: { polling_interval: 2 },
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
    jobs: { polling_interval: 1 },
  },
  overridden_keys: [
    'datasets.histogram_bins',
    'datasets.streaming_batch_size',
    'benchmarks.default_max_documents',
    'benchmarks.default_batch_size',
    'benchmarks.default_parallelism',
    'jobs.polling_interval',
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
      jobs: { polling_interval: 4 },
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
