import { DestroyRef, Injectable, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { SettingsApiService } from '../api/settings-api.service';
import { errorMessage } from '../api/error-utils';
import type {
  RuntimeSettingKey,
  RuntimeSettingsPatchRequest,
  RuntimeSettingsResetRequest,
  RuntimeSettingsResponse,
  RuntimeSettingsValues,
} from '../api/api.models';

type RuntimeSettingsPatch = Omit<RuntimeSettingsPatchRequest, 'expected_revision'>;

@Injectable({ providedIn: 'root' })
export class SettingsStore {
  private readonly api = inject(SettingsApiService);
  private readonly destroyRef = inject(DestroyRef);
  private loaded = false;

  readonly settings = signal<RuntimeSettingsValues | null>(null);
  readonly defaults = signal<RuntimeSettingsValues | null>(null);
  readonly revision = signal(0);
  readonly overriddenKeys = signal<readonly RuntimeSettingKey[]>([]);
  readonly warning = signal<string | null>(null);
  readonly loading = signal(false);
  readonly saving = signal(false);
  readonly error = signal<string | null>(null);
  readonly conflict = signal(false);

  load(): void {
    if (this.loaded || this.loading()) return;
    this.fetch();
  }

  reload(): void {
    if (this.loading() || this.saving()) return;
    this.loaded = false;
    this.fetch();
  }

  save(patch: RuntimeSettingsPatch): void {
    if (this.saving() || this.settings() === null) return;
    this.saving.set(true);
    this.error.set(null);
    this.conflict.set(false);
    const request: RuntimeSettingsPatchRequest = {
      expected_revision: this.revision(),
      ...patch,
    };
    this.api.patch(request).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (response) => {
        this.applyResponse(response);
        this.saving.set(false);
      },
      error: (error: unknown) => {
        this.saving.set(false);
        this.handleError(error, 'Failed to save settings.');
      },
    });
  }

  reset(keys?: readonly RuntimeSettingKey[]): void {
    if (this.saving() || this.settings() === null) return;
    this.saving.set(true);
    this.error.set(null);
    this.conflict.set(false);
    const request: RuntimeSettingsResetRequest = {
      expected_revision: this.revision(),
      ...(keys === undefined ? {} : { keys: [...keys] }),
    };
    this.api.reset(request).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (response) => {
        this.applyResponse(response);
        this.saving.set(false);
      },
      error: (error: unknown) => {
        this.saving.set(false);
        this.handleError(error, 'Failed to reset settings.');
      },
    });
  }

  isOverridden(key: RuntimeSettingKey): boolean {
    return this.overriddenKeys().includes(key);
  }

  private fetch(): void {
    this.loading.set(true);
    this.error.set(null);
    this.conflict.set(false);
    this.api.get().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (response) => {
        this.applyResponse(response);
        this.loading.set(false);
      },
      error: (error: unknown) => {
        this.loading.set(false);
        this.handleError(error, 'Failed to load settings.');
      },
    });
  }

  private applyResponse(response: RuntimeSettingsResponse): void {
    this.settings.set(response.settings);
    this.defaults.set(response.defaults);
    this.revision.set(response.revision);
    this.overriddenKeys.set(response.overridden_keys ?? []);
    this.warning.set(response.warning ?? null);
    this.loaded = true;
    this.error.set(null);
    this.conflict.set(false);
  }

  private handleError(error: unknown, fallback: string): void {
    const isConflict = this.isConflictError(error);
    this.conflict.set(isConflict);
    this.error.set(isConflict
      ? 'Settings changed elsewhere. Reload before trying again.'
      : errorMessage(error, fallback));
  }

  private isConflictError(error: unknown): boolean {
    return typeof error === 'object'
      && error !== null
      && 'status' in error
      && (error as { status?: unknown }).status === 409;
  }
}
