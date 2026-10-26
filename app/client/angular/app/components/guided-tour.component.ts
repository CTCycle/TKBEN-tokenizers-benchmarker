// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { DOCUMENT } from '@angular/common';
import { Component, DestroyRef, HostListener, effect, inject, input, output, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { NavigationEnd, Router } from '@angular/router';
import { filter } from 'rxjs';
import { Tour, TourPlacement } from '../core/guidance/guidance';
import { ModalA11yDirective } from '../core/ui/modal-a11y.directive';

interface Box {
  readonly top: number;
  readonly left: number;
  readonly width: number;
  readonly height: number;
}

const PANEL_WIDTH = 340;
const PANEL_HEIGHT = 210;
const GAP = 16;
const EDGE = 16;

@Component({
  selector: 'app-guided-tour',
  imports: [ModalA11yDirective],
  templateUrl: './guided-tour.component.html',
})
export class GuidedTourComponent {
  private readonly document = inject(DOCUMENT);
  private readonly router = inject(Router);
  private readonly timers: ReturnType<typeof setTimeout>[] = [];

  readonly tour = input.required<Tour>();
  readonly closed = output<void>();

  protected readonly index = signal(0);
  protected readonly spot = signal<Box | null>(null);
  protected readonly panel = signal({ top: 80, left: 80 });

  constructor() {
    effect(() => {
      this.index();
      this.tour();
      this.scheduleLayout();
    });
    this.router.events
      .pipe(filter((event): event is NavigationEnd => event instanceof NavigationEnd), takeUntilDestroyed())
      .subscribe((event) => {
        if (!event.urlAfterRedirects.startsWith(this.tour().route)) this.closed.emit();
      });
    inject(DestroyRef).onDestroy(() => this.timers.forEach(clearTimeout));
  }

  protected get step() {
    const steps = this.tour().steps;
    return steps[this.index()] ?? steps[0];
  }

  protected get isLast(): boolean {
    return this.index() >= this.tour().steps.length - 1;
  }

  protected back(): void {
    this.index.update((value) => Math.max(0, value - 1));
  }

  protected next(): void {
    if (this.isLast) this.closed.emit();
    else this.index.update((value) => value + 1);
  }

  @HostListener('window:resize')
  @HostListener('window:scroll')
  protected measure(): void {
    const target = this.document.querySelector<HTMLElement>(this.step.target);
    if (!target) {
      this.spot.set(null);
      this.panel.set({
        top: Math.max(EDGE, (window.innerHeight - PANEL_HEIGHT) / 2),
        left: Math.max(EDGE, (window.innerWidth - PANEL_WIDTH) / 2),
      });
      return;
    }
    const rect = target.getBoundingClientRect();
    const box = { top: rect.top, left: rect.left, width: rect.width, height: rect.height };
    this.spot.set(box);
    this.panel.set(this.place(box, this.step.placement ?? 'bottom'));
  }

  private scheduleLayout(): void {
    this.timers.forEach(clearTimeout);
    this.timers.length = 0;
    const target = this.document.querySelector<HTMLElement>(this.step.target);
    target?.scrollIntoView({ block: 'center', behavior: this.reducedMotion() ? 'auto' : 'smooth' });
    this.measure();
    // Smooth scrolling and lazily loaded pages settle after the first measurement.
    for (const delay of [150, 500, 1200]) this.timers.push(setTimeout(() => this.measure(), delay));
  }

  private place(box: Box, placement: TourPlacement): { top: number; left: number } {
    const width = Math.min(PANEL_WIDTH, window.innerWidth - 2 * EDGE);
    let top = box.top + box.height + GAP;
    let left = box.left;
    if (placement === 'top') top = box.top - PANEL_HEIGHT - GAP;
    if (placement === 'right' || placement === 'left') {
      top = box.top;
      left = placement === 'right' ? box.left + box.width + GAP : box.left - width - GAP;
    }
    if (top + PANEL_HEIGHT > window.innerHeight - EDGE) top = box.top - PANEL_HEIGHT - GAP;
    if (top < EDGE) top = Math.min(window.innerHeight - PANEL_HEIGHT - EDGE, box.top + box.height + GAP);
    left = Math.min(Math.max(EDGE, left), window.innerWidth - width - EDGE);
    return { top: Math.max(EDGE, top), left };
  }

  private reducedMotion(): boolean {
    return typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }
}
