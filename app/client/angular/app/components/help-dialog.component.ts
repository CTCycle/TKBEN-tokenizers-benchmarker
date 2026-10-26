// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, inject, output } from '@angular/core';
import { Router } from '@angular/router';
import { TIPS, TOURS } from '../core/guidance/guidance';
import { GuidanceService } from '../core/guidance/guidance.service';
import { ModalA11yDirective } from '../core/ui/modal-a11y.directive';

@Component({
  selector: 'app-help-dialog',
  imports: [ModalA11yDirective],
  templateUrl: './help-dialog.component.html',
})
export class HelpDialogComponent {
  private readonly guidance = inject(GuidanceService);
  private readonly router = inject(Router);

  readonly closed = output<void>();

  protected readonly tips = TIPS;
  protected readonly tours = [...TOURS].sort(
    (a, b) => Number(this.isCurrent(b.route)) - Number(this.isCurrent(a.route)),
  );

  protected isCurrent(route: string): boolean {
    return this.router.url.startsWith(route);
  }

  protected start(id: string): void {
    this.closed.emit();
    void this.guidance.start(id);
  }
}
