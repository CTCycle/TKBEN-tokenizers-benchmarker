import { Component, inject } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { AppShellComponent } from './components/app-shell.component';
import { StartupLoadingComponent } from './components/startup-loading.component';
import { StartupReadinessService } from './core/services/startup-readiness.service';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, AppShellComponent, StartupLoadingComponent],
  templateUrl: './app.html',
})
export class App {
  protected readonly startup = inject(StartupReadinessService);

  constructor() {
    this.startup.start();
  }
}
