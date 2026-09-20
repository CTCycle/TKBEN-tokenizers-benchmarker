import { signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { FormGroup } from '@angular/forms';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { SettingsStore } from '../core/state/settings.store';
import type { RuntimeSettingsValues } from '../core/api/api.models';
import { SettingsPageComponent } from './settings-page.component';

const defaults: RuntimeSettingsValues = {
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
};

const current: RuntimeSettingsValues = {
  ...defaults,
  datasets: { ...defaults.datasets, histogram_bins: 30, max_upload_bytes: 32 * 1024 * 1024 },
  benchmarks: { ...defaults.benchmarks, default_max_documents: 2500, default_batch_size: 32, default_parallelism: 2 },
  jobs: { polling_interval: 2 },
};

afterEach(() => TestBed.resetTestingModule());

describe('SettingsPageComponent', () => {
  function createPage(overriddenKeys: string[] = ['datasets.histogram_bins']) {
    const store = {
      settings: signal<RuntimeSettingsValues | null>(current),
      defaults: signal<RuntimeSettingsValues | null>(defaults),
      revision: signal(4),
      overriddenKeys: signal(overriddenKeys),
      warning: signal<string | null>(null),
      loading: signal(false),
      saving: signal(false),
      error: signal<string | null>(null),
      conflict: signal(false),
      save: vi.fn(),
      reset: vi.fn(),
      reload: vi.fn(),
      isOverridden: vi.fn((key: string) => overriddenKeys.includes(key)),
    };
    TestBed.configureTestingModule({ providers: [
      { provide: SettingsStore, useValue: store },
    ] });
    const page = TestBed.runInInjectionContext(() => new SettingsPageComponent()) as unknown as {
      activeTab: () => string;
      form: FormGroup;
      selectTab: (tab: string) => void;
      handleTabKeydown: (event: KeyboardEvent, tab: string) => void;
      saveChanges: () => void;
      resetSetting: (key: string) => void;
      resetAll: () => void;
      reloadAfterConflict: () => void;
      tokenizerLimitsInvalid: () => boolean;
      fieldError: (controlName: string) => string | null;
      fieldDescribedBy: (helpId: string, controlName: string) => string;
    };
    TestBed.tick();
    return { page, store };
  }

  it('hydrates the form, supports tab keyboard navigation, and validates tokenizer limits', () => {
    const { page } = createPage();

    expect(page.form.getRawValue()).toMatchObject({
      histogramBins: 30,
      datasetMaxUploadMiB: 32,
      defaultDiscoveryLimit: 50,
      benchmarkDefaultMaxDocuments: 2500,
      benchmarkDefaultBatchSize: 32,
      benchmarkDefaultParallelism: 2,
      jobPollingInterval: 2,
    });
    expect(page.form.pristine).toBe(true);

    page.selectTab('tokenizers');
    expect(page.activeTab()).toBe('tokenizers');
    page.handleTabKeydown(new KeyboardEvent('keydown', { key: 'ArrowRight' }), 'tokenizers');
    expect(page.activeTab()).toBe('benchmarks');
    page.handleTabKeydown(new KeyboardEvent('keydown', { key: 'ArrowRight' }), 'benchmarks');
    expect(page.activeTab()).toBe('runtime');

    page.form.patchValue({ defaultDiscoveryLimit: 200, maxDiscoveryLimit: 100 });
    expect(page.tokenizerLimitsInvalid()).toBe(true);
    expect(page.form.invalid).toBe(true);
  });

  it('explains field-level and cross-field validation failures inline', () => {
    const { page } = createPage();

    page.form.controls['histogramBins'].setValue(4);
    page.form.controls['histogramBins'].markAsTouched();
    expect(page.fieldError('histogramBins')).toBe('Must be at least 5.');
    expect(page.fieldDescribedBy('settings-help-histogram-bins', 'histogramBins'))
      .toBe('settings-help-histogram-bins settings-error-histogramBins');

    page.form.controls['downloadRetryAttempts'].setValue(1.5);
    page.form.controls['downloadRetryAttempts'].markAsTouched();
    expect(page.fieldError('downloadRetryAttempts')).toBe('Use a whole number.');

    page.form.controls['metadataCandidateMultiplier'].setValue(11);
    page.form.controls['metadataCandidateMultiplier'].markAsTouched();
    expect(page.fieldError('metadataCandidateMultiplier')).toBe('Must be no more than 10.');

    page.form.patchValue({ defaultDiscoveryLimit: 200, maxDiscoveryLimit: 100 });
    page.form.controls['defaultDiscoveryLimit'].markAsTouched();
    page.form.controls['maxDiscoveryLimit'].markAsTouched();
    expect(page.fieldError('defaultDiscoveryLimit')).toBe('Must not exceed Maximum discovery limit.');
    expect(page.fieldError('maxDiscoveryLimit')).toBe('Must be at least Default discovery limit.');
  });

  it('converts MiB to bytes and sends an explicit typed save patch', () => {
    const { page, store } = createPage([]);
    page.form.patchValue({
      datasetMaxUploadMiB: 40,
      tokenizerMaxUploadMiB: 12,
      histogramBins: 25,
      benchmarkDefaultMaxDocuments: 5000,
      benchmarkDefaultBatchSize: 64,
      benchmarkDefaultParallelism: 4,
    });
    page.form.markAsDirty();
    page.saveChanges();

    expect(store.save).toHaveBeenCalledWith(expect.objectContaining({
      datasets: expect.objectContaining({
        max_upload_bytes: 40 * 1024 * 1024,
        histogram_bins: 25,
      }),
      tokenizers: expect.objectContaining({ max_upload_bytes: 12 * 1024 * 1024 }),
      benchmarks: expect.objectContaining({
        default_max_documents: 5000,
        default_batch_size: 64,
        default_parallelism: 4,
      }),
    }));
  });

  it('resets overridden values and offers safe conflict reload', () => {
    const { page, store } = createPage();

    page.resetSetting('datasets.histogram_bins');
    expect(store.reset).toHaveBeenCalledWith(['datasets.histogram_bins']);

    page.resetAll();
    expect(store.reset).toHaveBeenCalledWith();

    store.conflict.set(true);
    page.reloadAfterConflict();
    expect(store.reload).toHaveBeenCalledTimes(1);
  });
});
