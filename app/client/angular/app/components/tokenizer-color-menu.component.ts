// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { AfterViewInit, Component, ElementRef, HostListener, computed, inject, input, output } from '@angular/core';
import { BENCHMARK_SERIES_COLORS } from '../core/utils/benchmark-dashboard-data';

@Component({
  selector: 'app-tokenizer-color-menu',
  templateUrl: './tokenizer-color-menu.component.html',
})
export class TokenizerColorMenuComponent implements AfterViewInit {
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  readonly tokenizer = input.required<string>();
  readonly color = input.required<string>();
  readonly x = input.required<number>();
  readonly y = input.required<number>();
  readonly colorPicked = output<string>();
  readonly colorReset = output<void>();
  readonly closed = output<void>();
  protected readonly palette = BENCHMARK_SERIES_COLORS;
  protected readonly left = computed(() => Math.max(8, Math.min(this.x(), window.innerWidth - 248)));
  protected readonly top = computed(() => Math.max(8, Math.min(this.y(), window.innerHeight - 190)));

  ngAfterViewInit(): void {
    this.host.nativeElement.querySelector<HTMLElement>('button')?.focus();
  }

  @HostListener('document:pointerdown', ['$event'])
  protected onOutsidePointer(event: PointerEvent): void {
    if (event.button === 0 && !this.host.nativeElement.contains(event.target as Node)) this.closed.emit();
  }

  @HostListener('document:keydown.escape')
  protected onEscape(): void {
    this.closed.emit();
  }
}
