import { Component, signal } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { HfAccessKeyManagerComponent } from './hf-access-key-manager.component';

interface NavItem {
  readonly path: string;
  readonly label: string;
  readonly icon: 'datasets' | 'tokenizers' | 'benchmark';
}

@Component({
  selector: 'app-shell',
  imports: [RouterLink, RouterLinkActive, HfAccessKeyManagerComponent],
  templateUrl: './app-shell.component.html',
})
export class AppShellComponent {
  protected readonly keyManagerOpen = signal(false);
  protected readonly navItems: readonly NavItem[] = [
    { path: '/dataset', label: 'Datasets', icon: 'datasets' },
    { path: '/tokenizers', label: 'Tokenizers', icon: 'tokenizers' },
    { path: '/cross-benchmark', label: 'Cross Benchmark', icon: 'benchmark' },
  ];

  protected toggleKeyManager(): void {
    this.keyManagerOpen.update((open) => !open);
  }

  protected closeKeyManager(): void {
    this.keyManagerOpen.set(false);
  }
}
