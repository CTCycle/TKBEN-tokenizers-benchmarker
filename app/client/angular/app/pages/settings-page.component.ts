// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, DestroyRef, effect, inject, OnDestroy, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subject, of, switchMap, timer } from 'rxjs';
import {
  AbstractControl,
  FormControl,
  FormGroup,
  ReactiveFormsModule,
  ValidatorFn,
  ValidationErrors,
  Validators,
} from '@angular/forms';
import { SettingsStore } from '../core/state/settings.store';
import { HfAccessKeyManagerComponent } from '../components/hf-access-key-manager.component';
import { DatabaseSettingsComponent } from '../components/database-settings.component';
import { DirectorySettingsComponent } from '../components/directory-settings.component';
import type {
  RuntimeSettingKey,
  RuntimeSettingsValues,
} from '../core/api/api.models';

type SettingsTab = 'data' | 'tokenizers' | 'benchmarks' | 'runtime' | 'storage' | 'database' | 'keys';

const SETTINGS_TABS: readonly SettingsTab[] = ['data', 'tokenizers', 'benchmarks', 'runtime', 'storage', 'database', 'keys'];

const BYTES_PER_MIB = 1024 * 1024;

const wholeNumber: ValidatorFn = (control: AbstractControl): ValidationErrors | null => {
  const value = control.value;
  return value === null || value === '' || Number.isInteger(value) ? null : { wholeNumber: true };
};

const tokenizerLimits: ValidatorFn = (control: AbstractControl): ValidationErrors | null => {
  const group = control as FormGroup;
  const defaultLimit = group.controls['defaultDiscoveryLimit']?.value;
  const maxLimit = group.controls['maxDiscoveryLimit']?.value;
  const candidateCap = group.controls['maxDiscoveryCandidates']?.value;
  if (![defaultLimit, maxLimit, candidateCap].every((value) => typeof value === 'number' && Number.isFinite(value))) return null;
  const errors: ValidationErrors = {};
  if (defaultLimit > maxLimit) errors['tokenizerDefaultExceedsMaximum'] = true;
  if (maxLimit > candidateCap) errors['tokenizerMaximumExceedsCandidates'] = true;
  return Object.keys(errors).length ? errors : null;
};

@Component({
  selector: 'app-settings-page',
  imports: [ReactiveFormsModule, HfAccessKeyManagerComponent, DatabaseSettingsComponent, DirectorySettingsComponent],
  templateUrl: './settings-page.component.html',
})
export class SettingsPageComponent implements OnDestroy {
  protected readonly store = inject(SettingsStore);
  private readonly destroyRef = inject(DestroyRef);
  private readonly persistenceRequests = new Subject<boolean>();
  protected readonly activeTab = signal<SettingsTab>('data');
  private hydrateFromServer = true;

  protected readonly form = new FormGroup({
    histogramBins: new FormControl<number | null>(null, {
      validators: [Validators.required, wholeNumber, Validators.min(5), Validators.max(100)],
    }),
    datasetMaxUploadMiB: new FormControl<number | null>(null, {
      validators: [Validators.required, wholeNumber, Validators.min(1)],
    }),
    downloadTimeoutSeconds: new FormControl<number | null>(null, {
      validators: [Validators.required, Validators.min(1)],
    }),
    downloadRetryAttempts: new FormControl<number | null>(null, {
      validators: [Validators.required, wholeNumber, Validators.min(1), Validators.max(10)],
    }),
    downloadRetryBackoffSeconds: new FormControl<number | null>(null, {
      validators: [Validators.required, Validators.min(0), Validators.max(60)],
    }),
    defaultDiscoveryLimit: new FormControl<number | null>(null, {
      validators: [Validators.required, wholeNumber, Validators.min(1), Validators.max(250)],
    }),
    maxDiscoveryLimit: new FormControl<number | null>(null, {
      validators: [Validators.required, wholeNumber, Validators.min(1), Validators.max(250)],
    }),
    maxDiscoveryCandidates: new FormControl<number | null>(null, {
      validators: [Validators.required, wholeNumber, Validators.min(1)],
    }),
    metadataCandidateMultiplier: new FormControl<number | null>(null, {
      validators: [Validators.required, wholeNumber, Validators.min(1), Validators.max(10)],
    }),
    tokenizerMaxUploadMiB: new FormControl<number | null>(null, {
      validators: [Validators.required, wholeNumber, Validators.min(1)],
    }),
    benchmarkDefaultMaxDocuments: new FormControl<number | null>(null, {
      validators: [Validators.required, wholeNumber, Validators.min(1), Validators.max(100_000)],
    }),
    benchmarkDefaultBatchSize: new FormControl<number | null>(null, {
      validators: [Validators.required, wholeNumber, Validators.min(1), Validators.max(4096)],
    }),
    benchmarkDefaultParallelism: new FormControl<number | null>(null, {
      validators: [Validators.required, wholeNumber, Validators.min(1), Validators.max(128)],
    }),
    datasetStreamingBatchSize: new FormControl<number | null>(null, {
      validators: [Validators.required, wholeNumber, Validators.min(100)],
    }),
    benchmarkStreamingBatchSize: new FormControl<number | null>(null, {
      validators: [Validators.required, wholeNumber, Validators.min(100)],
    }),
    jobPollingInterval: new FormControl<number | null>(null, {
      validators: [Validators.required, Validators.min(0.25)],
    }),
  }, { validators: tokenizerLimits });

  constructor() {
    const initialSettings = this.store.settings();
    if (initialSettings) {
      this.hydrate(initialSettings);
      this.hydrateFromServer = false;
    }
    effect(() => {
      const settings = this.store.settings();
      if (!settings) return;
      if (!this.hydrateFromServer && this.form.dirty && !this.matchesSettings(settings)) return;
      this.hydrate(settings);
      this.hydrateFromServer = false;
    });

    this.form.valueChanges.pipe(takeUntilDestroyed(this.destroyRef)).subscribe(() => {
      this.store.notePendingEdit();
      this.persistenceRequests.next(false);
    });
    this.persistenceRequests.pipe(
      switchMap((immediate) => immediate ? of(null) : timer(400)),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe(() => this.persistCurrentSettings());

  }

  ngOnDestroy(): void {
    if (this.form.dirty) this.persistCurrentSettings();
  }

  protected selectTab(tab: SettingsTab): void {
    this.activeTab.set(tab);
  }

  protected handleTabKeydown(event: KeyboardEvent, tab: SettingsTab): void {
    const index = SETTINGS_TABS.indexOf(tab);
    let nextIndex: number | null = null;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') nextIndex = (index + 1) % SETTINGS_TABS.length;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') nextIndex = (index - 1 + SETTINGS_TABS.length) % SETTINGS_TABS.length;
    if (event.key === 'Home') nextIndex = 0;
    if (event.key === 'End') nextIndex = SETTINGS_TABS.length - 1;
    if (nextIndex === null) return;
    event.preventDefault();
    const nextTab = SETTINGS_TABS[nextIndex];
    this.activeTab.set(nextTab);
    document.getElementById(`settings-tab-${nextTab}`)?.focus();
  }

  protected persistCurrentSettings(): void {
    if (this.form.invalid || this.store.loading() || !this.store.settings()) return;
    const value = this.form.getRawValue();
    const datasetMaxUploadBytes = this.mibToBytes(value.datasetMaxUploadMiB);
    const tokenizerMaxUploadBytes = this.mibToBytes(value.tokenizerMaxUploadMiB);
    if (datasetMaxUploadBytes === null || tokenizerMaxUploadBytes === null) {
      this.form.controls.datasetMaxUploadMiB.updateValueAndValidity();
      this.form.controls.tokenizerMaxUploadMiB.updateValueAndValidity();
      return;
    }

    this.store.save({
      datasets: {
        histogram_bins: this.requiredNumber(value.histogramBins),
        streaming_batch_size: this.requiredNumber(value.datasetStreamingBatchSize),
        max_upload_bytes: datasetMaxUploadBytes,
        download_timeout_seconds: this.requiredNumber(value.downloadTimeoutSeconds),
        download_retry_attempts: this.requiredNumber(value.downloadRetryAttempts),
        download_retry_backoff_seconds: this.requiredNumber(value.downloadRetryBackoffSeconds),
      },
      tokenizers: {
        default_discovery_limit: this.requiredNumber(value.defaultDiscoveryLimit),
        max_discovery_limit: this.requiredNumber(value.maxDiscoveryLimit),
        max_discovery_candidates: this.requiredNumber(value.maxDiscoveryCandidates),
        metadata_candidate_multiplier: this.requiredNumber(value.metadataCandidateMultiplier),
        max_upload_bytes: tokenizerMaxUploadBytes,
      },
      benchmarks: {
        default_max_documents: this.requiredNumber(value.benchmarkDefaultMaxDocuments),
        default_batch_size: this.requiredNumber(value.benchmarkDefaultBatchSize),
        default_parallelism: this.requiredNumber(value.benchmarkDefaultParallelism),
        streaming_batch_size: this.requiredNumber(value.benchmarkStreamingBatchSize),
      },
      jobs: {
        polling_interval: this.requiredNumber(value.jobPollingInterval),
      },
    });
  }

  protected resetSetting(key: RuntimeSettingKey): void {
    const defaults = this.store.defaults();
    if (!defaults || !this.store.isOverridden(key)) return;
    switch (key) {
      case 'tokenizers.default_discovery_limit': this.form.controls.defaultDiscoveryLimit.setValue(defaults.tokenizers.default_discovery_limit); break;
      case 'tokenizers.max_discovery_limit': this.form.controls.maxDiscoveryLimit.setValue(defaults.tokenizers.max_discovery_limit); break;
      case 'tokenizers.max_discovery_candidates': this.form.controls.maxDiscoveryCandidates.setValue(defaults.tokenizers.max_discovery_candidates); break;
      case 'tokenizers.metadata_candidate_multiplier': this.form.controls.metadataCandidateMultiplier.setValue(defaults.tokenizers.metadata_candidate_multiplier); break;
      case 'tokenizers.max_upload_bytes': this.form.controls.tokenizerMaxUploadMiB.setValue(this.bytesToMib(defaults.tokenizers.max_upload_bytes)); break;
      case 'datasets.histogram_bins': this.form.controls.histogramBins.setValue(defaults.datasets.histogram_bins); break;
      case 'datasets.streaming_batch_size': this.form.controls.datasetStreamingBatchSize.setValue(defaults.datasets.streaming_batch_size); break;
      case 'datasets.max_upload_bytes': this.form.controls.datasetMaxUploadMiB.setValue(this.bytesToMib(defaults.datasets.max_upload_bytes)); break;
      case 'datasets.download_timeout_seconds': this.form.controls.downloadTimeoutSeconds.setValue(defaults.datasets.download_timeout_seconds); break;
      case 'datasets.download_retry_attempts': this.form.controls.downloadRetryAttempts.setValue(defaults.datasets.download_retry_attempts); break;
      case 'datasets.download_retry_backoff_seconds': this.form.controls.downloadRetryBackoffSeconds.setValue(defaults.datasets.download_retry_backoff_seconds); break;
      case 'benchmarks.default_max_documents': this.form.controls.benchmarkDefaultMaxDocuments.setValue(defaults.benchmarks.default_max_documents); break;
      case 'benchmarks.default_batch_size': this.form.controls.benchmarkDefaultBatchSize.setValue(defaults.benchmarks.default_batch_size); break;
      case 'benchmarks.default_parallelism': this.form.controls.benchmarkDefaultParallelism.setValue(defaults.benchmarks.default_parallelism); break;
      case 'benchmarks.streaming_batch_size': this.form.controls.benchmarkStreamingBatchSize.setValue(defaults.benchmarks.streaming_batch_size); break;
      case 'jobs.polling_interval': this.form.controls.jobPollingInterval.setValue(defaults.jobs.polling_interval); break;
    }
    this.form.markAsDirty();
    this.persistenceRequests.next(true);
  }

  protected resetAll(): void {
    const defaults = this.store.defaults();
    if (!defaults || !this.store.overriddenKeys().length) return;
    this.form.setValue(this.formValues(defaults));
    this.form.markAsDirty();
    this.persistenceRequests.next(true);
  }

  protected reloadAfterConflict(): void {
    if (!this.store.conflict() || this.store.saving()) return;
    this.hydrateFromServer = true;
    this.store.reload();
  }

  protected isOverridden(key: RuntimeSettingKey): boolean {
    return this.store.isOverridden(key);
  }

  protected tokenizerLimitsInvalid(): boolean {
    return this.form.hasError('tokenizerDefaultExceedsMaximum')
      || this.form.hasError('tokenizerMaximumExceedsCandidates');
  }

  protected fieldError(controlName: string): string | null {
    const control = this.form.get(controlName);
    if (!control || (!control.dirty && !control.touched)) return null;

    if (controlName === 'defaultDiscoveryLimit'
      && this.form.hasError('tokenizerDefaultExceedsMaximum')) {
      return 'Must not exceed Maximum discovery limit.';
    }
    if (control.hasError('required')) return 'Enter a value.';
    if (control.hasError('wholeNumber')) return 'Use a whole number.';

    const minimum = control.getError('min') as { min?: number } | null;
    if (minimum?.min !== undefined) return `Must be at least ${minimum.min}.`;

    const maximum = control.getError('max') as { max?: number } | null;
    if (maximum?.max !== undefined) return `Must be no more than ${maximum.max}.`;

    if (controlName === 'maxDiscoveryLimit') {
      const lowerBoundError = this.form.hasError('tokenizerDefaultExceedsMaximum');
      const upperBoundError = this.form.hasError('tokenizerMaximumExceedsCandidates');
      if (lowerBoundError && upperBoundError) {
        return 'Must be at least Default discovery limit and no more than Discovery candidate cap.';
      }
      if (lowerBoundError) return 'Must be at least Default discovery limit.';
      if (upperBoundError) return 'Must not exceed Discovery candidate cap.';
    }
    if (controlName === 'maxDiscoveryCandidates'
      && this.form.hasError('tokenizerMaximumExceedsCandidates')) {
      return 'Must be at least Maximum discovery limit.';
    }

    return null;
  }

  protected fieldDescribedBy(helpId: string, controlName: string): string {
    const errorId = `settings-error-${controlName}`;
    return this.fieldError(controlName) ? `${helpId} ${errorId}` : helpId;
  }

  protected defaultValue(key: RuntimeSettingKey): string {
    const defaults = this.store.defaults();
    if (!defaults) return '—';
    switch (key) {
      case 'datasets.max_upload_bytes': return `${this.bytesToMib(defaults.datasets.max_upload_bytes)} MiB`;
      case 'tokenizers.max_upload_bytes': return `${this.bytesToMib(defaults.tokenizers.max_upload_bytes)} MiB`;
      case 'tokenizers.default_discovery_limit': return `${defaults.tokenizers.default_discovery_limit}`;
      case 'tokenizers.max_discovery_limit': return `${defaults.tokenizers.max_discovery_limit}`;
      case 'tokenizers.max_discovery_candidates': return `${defaults.tokenizers.max_discovery_candidates}`;
      case 'tokenizers.metadata_candidate_multiplier': return `${defaults.tokenizers.metadata_candidate_multiplier}`;
      case 'datasets.histogram_bins': return `${defaults.datasets.histogram_bins}`;
      case 'datasets.streaming_batch_size': return `${defaults.datasets.streaming_batch_size}`;
      case 'datasets.download_timeout_seconds': return `${defaults.datasets.download_timeout_seconds} seconds`;
      case 'datasets.download_retry_attempts': return `${defaults.datasets.download_retry_attempts}`;
      case 'datasets.download_retry_backoff_seconds': return `${defaults.datasets.download_retry_backoff_seconds} seconds`;
      case 'benchmarks.default_max_documents': return `${defaults.benchmarks.default_max_documents}`;
      case 'benchmarks.default_batch_size': return `${defaults.benchmarks.default_batch_size}`;
      case 'benchmarks.default_parallelism': return `${defaults.benchmarks.default_parallelism}`;
      case 'benchmarks.streaming_batch_size': return `${defaults.benchmarks.streaming_batch_size}`;
      case 'jobs.polling_interval': return `${defaults.jobs.polling_interval} seconds`;
    }
  }

  private hydrate(settings: RuntimeSettingsValues): void {
    this.form.reset(this.formValues(settings), { emitEvent: false });
    this.form.markAsPristine();
    this.form.markAsUntouched();
    this.form.updateValueAndValidity({ emitEvent: false });
  }

  private formValues(settings: RuntimeSettingsValues) {
    return {
      histogramBins: settings.datasets.histogram_bins,
      datasetMaxUploadMiB: this.bytesToMib(settings.datasets.max_upload_bytes),
      downloadTimeoutSeconds: settings.datasets.download_timeout_seconds,
      downloadRetryAttempts: settings.datasets.download_retry_attempts,
      downloadRetryBackoffSeconds: settings.datasets.download_retry_backoff_seconds,
      defaultDiscoveryLimit: settings.tokenizers.default_discovery_limit,
      maxDiscoveryLimit: settings.tokenizers.max_discovery_limit,
      maxDiscoveryCandidates: settings.tokenizers.max_discovery_candidates,
      metadataCandidateMultiplier: settings.tokenizers.metadata_candidate_multiplier,
      tokenizerMaxUploadMiB: this.bytesToMib(settings.tokenizers.max_upload_bytes),
      benchmarkDefaultMaxDocuments: settings.benchmarks.default_max_documents,
      benchmarkDefaultBatchSize: settings.benchmarks.default_batch_size,
      benchmarkDefaultParallelism: settings.benchmarks.default_parallelism,
      datasetStreamingBatchSize: settings.datasets.streaming_batch_size,
      benchmarkStreamingBatchSize: settings.benchmarks.streaming_batch_size,
      jobPollingInterval: settings.jobs.polling_interval,
    };
  }

  private matchesSettings(settings: RuntimeSettingsValues): boolean {
    const expected = this.formValues(settings);
    const current = this.form.getRawValue();
    return (Object.keys(expected) as (keyof typeof expected)[])
      .every((key) => current[key] === expected[key]);
  }

  private requiredNumber(value: number | null): number {
    return value ?? 0;
  }

  private mibToBytes(value: number | null): number | null {
    if (value === null || !Number.isFinite(value) || value <= 0 || !Number.isInteger(value)) return null;
    return value * BYTES_PER_MIB;
  }

  private bytesToMib(value: number): number {
    return value / BYTES_PER_MIB;
  }
}
