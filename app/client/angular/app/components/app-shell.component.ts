// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, inject, signal } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { SettingsStore } from '../core/state/settings.store';
import { GuidanceService } from '../core/guidance/guidance.service';
import { GuidedTourComponent } from './guided-tour.component';
import { HelpDialogComponent } from './help-dialog.component';
import { JobsPanelComponent } from './jobs-panel.component';

interface NavItem {
  readonly path: string;
  readonly label: string;
  readonly icon: 'datasets' | 'tokenizers' | 'benchmark';
}

@Component({
  selector: 'app-shell',
  imports: [RouterLink, RouterLinkActive, JobsPanelComponent, HelpDialogComponent, GuidedTourComponent],
  templateUrl: './app-shell.component.html',
})
export class AppShellComponent {
  private readonly settingsStore = inject(SettingsStore);
  protected readonly guidance = inject(GuidanceService);
  protected readonly helpOpen = signal(false);
  protected readonly navItems: readonly NavItem[] = [
    { path: '/dataset', label: 'Datasets', icon: 'datasets' },
    { path: '/tokenizers', label: 'Tokenizers', icon: 'tokenizers' },
    { path: '/cross-benchmark', label: 'Cross Benchmark', icon: 'benchmark' },
  ];

  constructor() {
    this.settingsStore.load();
  }
}
