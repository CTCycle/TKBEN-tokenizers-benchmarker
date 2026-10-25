// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { Component, DestroyRef, OnInit, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { SettingsApiService } from '../core/api/settings-api.service';
import { errorMessage } from '../core/api/error-utils';
import type { PathSettings } from '../core/api/api.models';

@Component({
  selector: 'app-directory-settings',
  imports: [ReactiveFormsModule],
  templateUrl: './directory-settings.component.html',
})
export class DirectorySettingsComponent implements OnInit {
  private readonly api = inject(SettingsApiService);
  private readonly destroyRef = inject(DestroyRef);

  protected readonly paths = signal<PathSettings | null>(null);
  protected readonly loading = signal(true);
  protected readonly saving = signal(false);
  protected readonly error = signal<string | null>(null);

  protected readonly form = new FormGroup({
    dataDir: new FormControl('', { nonNullable: true, validators: [Validators.required] }),
    logDir: new FormControl('', { nonNullable: true, validators: [Validators.required] }),
  });

  ngOnInit(): void {
    this.api.getPaths().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (paths) => {
        this.apply(paths);
        this.loading.set(false);
      },
      error: (error: unknown) => {
        this.error.set(errorMessage(error, 'Unable to load directory settings.'));
        this.loading.set(false);
      },
    });
  }

  protected save(): void {
    if (this.form.invalid || this.saving()) return;
    const { dataDir, logDir } = this.form.getRawValue();
    this.saving.set(true);
    this.error.set(null);
    this.api.putPaths({ data_dir: dataDir.trim(), log_dir: logDir.trim() }).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (paths) => {
        this.apply(paths);
        this.saving.set(false);
      },
      error: (error: unknown) => {
        this.error.set(errorMessage(error, 'Unable to save directory settings.'));
        this.saving.set(false);
      },
    });
  }

  private apply(paths: PathSettings): void {
    this.paths.set(paths);
    this.form.reset({ dataDir: paths.data_dir, logDir: paths.log_dir });
  }
}
