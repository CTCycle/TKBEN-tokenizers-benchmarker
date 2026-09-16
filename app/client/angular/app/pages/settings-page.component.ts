import { Component, effect, inject, signal } from '@angular/core';
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
import type {
  RuntimeSettingKey,
  RuntimeSettingsPatchRequest,
  RuntimeSettingsValues,
} from '../core/api/api.models';

type SettingsTab = 'data' | 'tokenizers' | 'benchmark' | 'runtime';

const BYTES_PER_MIB = 1024 * 1024;
const SECONDS_PER_MINUTE = 60;

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
  if (defaultLimit > maxLimit) return { tokenizerDefaultExceedsMaximum: true };
  if (maxLimit > candidateCap) return { tokenizerMaximumExceedsCandidates: true };
  return null;
};

@Component({
  selector: 'app-settings-page',
  imports: [ReactiveFormsModule],
  templateUrl: './settings-page.component.html',
})
export class SettingsPageComponent {
  protected readonly store = inject(SettingsStore);
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
    benchmarkDefaultWarmupTrials: new FormControl<number | null>(null, {
      validators: [Validators.required, wholeNumber, Validators.min(0), Validators.max(100)],
    }),
    benchmarkDefaultTimedTrials: new FormControl<number | null>(null, {
      validators: [Validators.required, wholeNumber, Validators.min(1), Validators.max(200)],
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
    jobRetentionMinutes: new FormControl<number | null>(null, {
      validators: [Validators.required, Validators.min(0)],
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
      if (!settings || (!this.hydrateFromServer && this.form.dirty)) return;
      this.hydrate(settings);
      this.hydrateFromServer = false;
    });

  }

  protected selectTab(tab: SettingsTab): void {
    this.activeTab.set(tab);
  }

  protected handleTabKeydown(event: KeyboardEvent, tab: SettingsTab): void {
    const tabs: readonly SettingsTab[] = ['data', 'tokenizers', 'benchmark', 'runtime'];
    const index = tabs.indexOf(tab);
    let nextIndex: number | null = null;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') nextIndex = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') nextIndex = (index - 1 + tabs.length) % tabs.length;
    if (event.key === 'Home') nextIndex = 0;
    if (event.key === 'End') nextIndex = tabs.length - 1;
    if (nextIndex === null) return;
    event.preventDefault();
    const nextTab = tabs[nextIndex];
    this.activeTab.set(nextTab);
    document.getElementById(`settings-tab-${nextTab}`)?.focus();
  }

  protected saveChanges(): void {
    if (this.form.invalid || !this.form.dirty || this.store.saving()) {
      this.form.markAllAsTouched();
      return;
    }

    const value = this.form.getRawValue();
    const datasetMaxUploadBytes = this.mibToBytes(value.datasetMaxUploadMiB);
    const tokenizerMaxUploadBytes = this.mibToBytes(value.tokenizerMaxUploadMiB);
    if (datasetMaxUploadBytes === null || tokenizerMaxUploadBytes === null) {
      this.form.controls.datasetMaxUploadMiB.updateValueAndValidity();
      this.form.controls.tokenizerMaxUploadMiB.updateValueAndValidity();
      return;
    }

    const patch: RuntimeSettingsPatchRequest = {
      expected_revision: this.store.revision(),
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
        streaming_batch_size: this.requiredNumber(value.benchmarkStreamingBatchSize),
        default_max_documents: this.requiredNumber(value.benchmarkDefaultMaxDocuments),
        default_warmup_trials: this.requiredNumber(value.benchmarkDefaultWarmupTrials),
        default_timed_trials: this.requiredNumber(value.benchmarkDefaultTimedTrials),
        default_batch_size: this.requiredNumber(value.benchmarkDefaultBatchSize),
        default_parallelism: this.requiredNumber(value.benchmarkDefaultParallelism),
      },
      jobs: {
        polling_interval: this.requiredNumber(value.jobPollingInterval),
        terminal_retention_seconds: this.requiredNumber(value.jobRetentionMinutes) * SECONDS_PER_MINUTE,
      },
    };
    this.hydrateFromServer = true;
    this.store.save({
      datasets: patch.datasets,
      tokenizers: patch.tokenizers,
      benchmarks: patch.benchmarks,
      jobs: patch.jobs,
    });
  }

  protected resetSetting(key: RuntimeSettingKey): void {
    if (!this.store.isOverridden(key) || this.store.saving()) return;
    this.hydrateFromServer = true;
    this.store.reset([key]);
  }

  protected resetAll(): void {
    if (!this.store.overriddenKeys().length || this.store.saving()) return;
    this.hydrateFromServer = true;
    this.store.reset();
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
      case 'benchmarks.streaming_batch_size': return `${defaults.benchmarks.streaming_batch_size}`;
      case 'benchmarks.default_max_documents': return `${defaults.benchmarks.default_max_documents}`;
      case 'benchmarks.default_warmup_trials': return `${defaults.benchmarks.default_warmup_trials}`;
      case 'benchmarks.default_timed_trials': return `${defaults.benchmarks.default_timed_trials}`;
      case 'benchmarks.default_batch_size': return `${defaults.benchmarks.default_batch_size}`;
      case 'benchmarks.default_parallelism': return `${defaults.benchmarks.default_parallelism}`;
      case 'jobs.polling_interval': return `${defaults.jobs.polling_interval} seconds`;
      case 'jobs.terminal_retention_seconds': return `${defaults.jobs.terminal_retention_seconds / SECONDS_PER_MINUTE} minutes`;
    }
  }

  private hydrate(settings: RuntimeSettingsValues): void {
    this.form.reset({
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
      benchmarkDefaultWarmupTrials: settings.benchmarks.default_warmup_trials,
      benchmarkDefaultTimedTrials: settings.benchmarks.default_timed_trials,
      benchmarkDefaultBatchSize: settings.benchmarks.default_batch_size,
      benchmarkDefaultParallelism: settings.benchmarks.default_parallelism,
      datasetStreamingBatchSize: settings.datasets.streaming_batch_size,
      benchmarkStreamingBatchSize: settings.benchmarks.streaming_batch_size,
      jobPollingInterval: settings.jobs.polling_interval,
      jobRetentionMinutes: settings.jobs.terminal_retention_seconds / SECONDS_PER_MINUTE,
    }, { emitEvent: false });
    this.form.markAsPristine();
    this.form.markAsUntouched();
    this.form.updateValueAndValidity({ emitEvent: false });
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
