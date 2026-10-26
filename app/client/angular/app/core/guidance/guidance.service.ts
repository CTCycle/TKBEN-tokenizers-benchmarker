// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Injectable, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { TOURS, Tour } from './guidance';

@Injectable({ providedIn: 'root' })
export class GuidanceService {
  private readonly router = inject(Router);

  readonly activeTour = signal<Tour | null>(null);

  async start(id: string): Promise<void> {
    const tour = TOURS.find((candidate) => candidate.id === id);
    if (!tour) return;
    if (!this.router.url.startsWith(tour.route)) await this.router.navigateByUrl(tour.route);
    this.activeTour.set(tour);
  }

  stop(): void {
    this.activeTour.set(null);
  }
}
