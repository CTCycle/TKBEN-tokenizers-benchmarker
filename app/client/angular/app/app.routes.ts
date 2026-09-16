import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'dataset' },
  {
    path: 'dataset',
    loadComponent: () =>
      import('./pages/dataset-page.component').then((module) => module.DatasetPageComponent),
  },
  {
    path: 'tokenizers',
    loadComponent: () =>
      import('./pages/tokenizers-page.component').then((module) => module.TokenizersPageComponent),
  },
  {
    path: 'cross-benchmark',
    loadComponent: () =>
      import('./pages/cross-benchmark-page.component').then(
        (module) => module.CrossBenchmarkPageComponent,
      ),
  },
  {
    path: 'settings',
    loadComponent: () =>
      import('./pages/settings-page.component').then((module) => module.SettingsPageComponent),
  },
  { path: '**', redirectTo: 'dataset' },
];
