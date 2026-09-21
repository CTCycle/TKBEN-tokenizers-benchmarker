import { Component, input, output } from '@angular/core';
import type { StartupStatus } from '../core/services/startup-readiness.service';

@Component({
  selector: 'app-startup-loading',
  templateUrl: './startup-loading.component.html',
})
export class StartupLoadingComponent {
  readonly status = input<StartupStatus>('checking');
  readonly errorMessage = input<string | null>(null);
  readonly retry = output<void>();
}
