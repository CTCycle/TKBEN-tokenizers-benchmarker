import { Component, inject } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { SettingsStore } from '../core/state/settings.store';

interface NavItem {
  readonly path: string;
  readonly label: string;
  readonly icon: 'datasets' | 'tokenizers' | 'benchmark';
}

@Component({
  selector: 'app-shell',
  imports: [RouterLink, RouterLinkActive],
  templateUrl: './app-shell.component.html',
})
export class AppShellComponent {
  private readonly settingsStore = inject(SettingsStore);
  protected readonly navItems: readonly NavItem[] = [
    { path: '/dataset', label: 'Datasets', icon: 'datasets' },
    { path: '/tokenizers', label: 'Tokenizers', icon: 'tokenizers' },
    { path: '/cross-benchmark', label: 'Cross Benchmark', icon: 'benchmark' },
  ];

  constructor() {
    this.settingsStore.load();
  }
}
