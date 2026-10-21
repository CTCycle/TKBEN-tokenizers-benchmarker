// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { Component, DestroyRef, OnInit, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { SettingsApiService } from '../core/api/settings-api.service';
import type {
  DatabaseBackend,
  DatabaseConfiguration,
  DatabaseConfigurationRequest,
} from '../core/api/api.models';

@Component({
  selector: 'app-database-settings',
  imports: [ReactiveFormsModule],
  templateUrl: './database-settings.component.html',
})
export class DatabaseSettingsComponent implements OnInit {
  private readonly api = inject(SettingsApiService);
  private readonly destroyRef = inject(DestroyRef);

  protected readonly configuration = signal<DatabaseConfiguration | null>(null);
  protected readonly loading = signal(true);
  protected readonly testing = signal(false);
  protected readonly switching = signal(false);
  protected readonly error = signal<string | null>(null);
  protected readonly notice = signal<string | null>(null);
  protected readonly testResult = signal<{ success: boolean; message: string } | null>(null);
  private validatedCandidate: string | null = null;

  protected readonly form = new FormGroup({
    backend: new FormControl<DatabaseBackend>('sqlite', { nonNullable: true }),
    sqlitePath: new FormControl('database.db', { nonNullable: true, validators: [Validators.required] }),
    host: new FormControl('', { nonNullable: true }),
    port: new FormControl(5432, { nonNullable: true }),
    databaseName: new FormControl('', { nonNullable: true }),
    username: new FormControl('', { nonNullable: true }),
    password: new FormControl('', { nonNullable: true }),
    clearPassword: new FormControl(false, { nonNullable: true }),
    ssl: new FormControl(false, { nonNullable: true }),
    sslCa: new FormControl('', { nonNullable: true }),
    connectTimeout: new FormControl(30, { nonNullable: true, validators: [Validators.required, Validators.min(1)] }),
    insertBatchSize: new FormControl(1000, { nonNullable: true, validators: [Validators.required, Validators.min(1)] }),
  });

  ngOnInit(): void {
    this.form.valueChanges.pipe(takeUntilDestroyed(this.destroyRef)).subscribe(() => {
      this.validatedCandidate = null;
      this.testResult.set(null);
      this.error.set(null);
      this.notice.set(null);
    });
    this.api.getDatabase().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (configuration) => {
        this.configuration.set(configuration);
        this.patchForm(configuration);
        this.loading.set(false);
        if (configuration.connection_status === 'unavailable') {
          this.error.set(configuration.message ?? 'The active database is unavailable.');
        }
      },
      error: (response: { error?: { detail?: string } }) => {
        this.error.set(response.error?.detail ?? 'Unable to load database settings.');
        this.loading.set(false);
      },
    });
  }

  protected backendChanged(): void {
    const postgres = this.form.controls.backend.value === 'postgresql';
    const pgControls = [
      this.form.controls.host,
      this.form.controls.databaseName,
      this.form.controls.username,
    ];
    for (const control of pgControls) {
      control.setValidators(postgres ? [Validators.required] : null);
      control.updateValueAndValidity();
    }
    this.form.controls.sqlitePath.setValidators(postgres ? null : [Validators.required]);
    this.form.controls.sqlitePath.updateValueAndValidity();
    this.testResult.set(null);
    this.error.set(null);
    this.notice.set(null);
  }

  protected testConnection(): void {
    if (this.form.invalid || this.testing() || this.switching()) {
      this.form.markAllAsTouched();
      return;
    }
    this.testing.set(true);
    this.testResult.set(null);
    this.validatedCandidate = null;
    this.error.set(null);
    const request = this.toRequest();
    this.api.testDatabase(request).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (result) => {
        this.testResult.set({ success: result.success, message: result.message });
        this.validatedCandidate = result.success ? JSON.stringify(request) : null;
        this.testing.set(false);
      },
      error: (response: { error?: { detail?: string } }) => {
        this.error.set(response.error?.detail ?? 'Unable to test this database connection.');
        this.testing.set(false);
      },
    });
  }

  protected switchDatabase(): void {
    if (this.form.invalid || this.testing() || this.switching() || !this.hasSuccessfulTest()) {
      this.form.markAllAsTouched();
      return;
    }
    this.switching.set(true);
    this.error.set(null);
    this.notice.set(null);
    this.api.switchDatabase(this.toRequest()).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (configuration) => {
        this.configuration.set(configuration);
        this.notice.set('Database switched. Reloading application data…');
        window.setTimeout(() => window.location.reload(), 500);
      },
      error: (response: { error?: { detail?: string } }) => {
        this.error.set(response.error?.detail ?? 'The database could not be activated.');
        this.switching.set(false);
      },
    });
  }

  protected hasSuccessfulTest(): boolean {
    return Boolean(
      this.testResult()?.success
      && this.validatedCandidate
      && !this.form.invalid
      && this.validatedCandidate === JSON.stringify(this.toRequest()),
    );
  }

  protected currentLabel(): string {
    const current = this.configuration();
    if (!current) return 'Checking';
    if (current.connection_status === 'unavailable') return 'Unavailable';
    return current.backend === 'sqlite' ? 'SQLite · Connected' : 'PostgreSQL · Connected';
  }

  private patchForm(configuration: DatabaseConfiguration): void {
    this.form.patchValue({
      backend: configuration.backend,
      sqlitePath: configuration.sqlite_path ?? 'database.db',
      host: configuration.host ?? '',
      port: configuration.port ?? 5432,
      databaseName: configuration.database_name ?? '',
      username: configuration.username ?? '',
      password: '',
      clearPassword: false,
      ssl: configuration.ssl,
      sslCa: configuration.ssl_ca ?? '',
      connectTimeout: configuration.connect_timeout,
      insertBatchSize: configuration.insert_batch_size,
    });
    this.form.markAsPristine();
    this.backendChanged();
  }

  private toRequest(): DatabaseConfigurationRequest {
    const value = this.form.getRawValue();
    const common = {
      backend: value.backend,
      connect_timeout: value.connectTimeout,
      insert_batch_size: value.insertBatchSize,
    };
    if (value.backend === 'sqlite') {
      return { ...common, sqlite_path: value.sqlitePath.trim() };
    }
    const request: DatabaseConfigurationRequest = {
      ...common,
      host: value.host.trim(),
      port: value.port,
      database_name: value.databaseName.trim(),
      username: value.username.trim(),
      ssl: value.ssl,
      ssl_ca: value.sslCa.trim(),
      clear_password: value.clearPassword,
    };
    if (value.password) request.password = value.password;
    return request;
  }
}
