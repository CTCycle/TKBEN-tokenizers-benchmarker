import { AfterViewInit, Component, ElementRef, OnDestroy, ViewChild, effect, input, signal } from '@angular/core';
import type { WordCloudTerm } from '../core/api/api.models';

interface WordCloudLayoutTerm extends WordCloudTerm {
  x: number;
  y: number;
  rotate: number;
  fontSize: number;
}

@Component({
  selector: 'app-word-cloud',
  templateUrl: './word-cloud.component.html',
})
export class WordCloudComponent implements AfterViewInit, OnDestroy {
  readonly terms = input<readonly WordCloudTerm[]>([]);
  @ViewChild('canvas') private canvas?: ElementRef<HTMLDivElement>;
  protected readonly layout = signal<readonly WordCloudLayoutTerm[]>([]);
  protected readonly fallback = signal(false);
  private worker: Worker | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private fitFrame: number | null = null;
  private initialLayoutFrame: number | null = null;
  private measurementRetryTimer: ReturnType<typeof setTimeout> | null = null;
  private measurementRetryCount = 0;
  private initialized = false;

  constructor() {
    effect(() => {
      this.terms();
      if (this.initialized) this.layoutTerms();
    });
  }

  ngAfterViewInit(): void {
    this.initialized = true;
    this.resizeObserver = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => this.layoutTerms());
    if (this.canvas?.nativeElement) this.resizeObserver?.observe(this.canvas.nativeElement);
    this.layoutTerms();
    if (typeof requestAnimationFrame !== 'undefined') {
      this.initialLayoutFrame = requestAnimationFrame(() => {
        this.initialLayoutFrame = null;
        if (this.initialized) this.layoutTerms();
      });
    }
  }

  ngOnDestroy(): void {
    this.resizeObserver?.disconnect();
    this.worker?.terminate();
    if (this.fitFrame !== null && typeof cancelAnimationFrame !== 'undefined') {
      cancelAnimationFrame(this.fitFrame);
    }
    if (this.initialLayoutFrame !== null && typeof cancelAnimationFrame !== 'undefined') {
      cancelAnimationFrame(this.initialLayoutFrame);
    }
    if (this.measurementRetryTimer !== null) clearTimeout(this.measurementRetryTimer);
  }

  private layoutTerms(): void {
    const source = [...this.terms()].filter((item) => item.word && item.count > 0).slice(0, 140);
    if (!source.length) {
      this.layout.set([]);
      this.fallback.set(false);
      return;
    }
    const element = this.canvas?.nativeElement;
    if (!element) {
      this.scheduleMeasurementRetry();
      return;
    }
    const bounds = element?.getBoundingClientRect();
    const parentBounds = element?.parentElement?.getBoundingClientRect();
    const measuredWidth = bounds?.width || element?.clientWidth || parentBounds?.width || element?.parentElement?.clientWidth || 0;
    const measuredHeight = bounds?.height || element?.clientHeight || parentBounds?.height || element?.parentElement?.clientHeight || 0;
    if (measuredWidth <= 0 || measuredHeight <= 0) {
      this.scheduleMeasurementRetry();
      return;
    }
    this.measurementRetryCount = 0;
    const width = Math.max(240, Math.round(measuredWidth));
    const height = Math.max(220, Math.round(measuredHeight));
    if (typeof Worker === 'undefined') {
      this.layout.set(this.fallbackLayout(source, width, height));
      this.fallback.set(true);
      this.scheduleFitToCanvas();
      return;
    }
    try {
      this.worker?.terminate();
      this.worker = new Worker(new URL('../core/workers/word-cloud.worker.ts', import.meta.url), { type: 'module' });
      this.worker.onmessage = (event: MessageEvent<{ terms: WordCloudLayoutTerm[] }>) => {
        this.layout.set(event.data.terms);
        this.fallback.set(false);
        this.scheduleFitToCanvas();
      };
      this.worker.onerror = () => {
        this.layout.set(this.fallbackLayout(source, width, height));
        this.fallback.set(true);
        this.scheduleFitToCanvas();
      };
      this.worker.postMessage({ terms: source, width, height });
    } catch {
      this.layout.set(this.fallbackLayout(source, width, height));
      this.fallback.set(true);
      this.scheduleFitToCanvas();
    }
  }

  private scheduleMeasurementRetry(): void {
    if (this.measurementRetryTimer !== null || this.measurementRetryCount >= 10) return;
    this.measurementRetryCount += 1;
    this.measurementRetryTimer = setTimeout(() => {
      this.measurementRetryTimer = null;
      if (this.initialized) this.layoutTerms();
    }, 50);
  }

  private scheduleFitToCanvas(): void {
    if (typeof requestAnimationFrame === 'undefined') return;
    if (this.fitFrame !== null) cancelAnimationFrame(this.fitFrame);
    this.fitFrame = requestAnimationFrame(() => {
      this.fitFrame = null;
      this.fitRenderedTerms();
    });
  }

  private fitRenderedTerms(): void {
    const canvas = this.canvas?.nativeElement;
    if (!canvas || !this.layout().length) return;
    const renderedTerms = [...canvas.querySelectorAll<HTMLElement>('.dataset-word-cloud-term')];
    if (renderedTerms.length !== this.layout().length || canvas.clientWidth <= 0 || canvas.clientHeight <= 0) return;

    const canvasRect = canvas.getBoundingClientRect();
    const padding = 8;
    const availableWidth = Math.max(24, canvas.clientWidth - padding * 2);
    const availableHeight = Math.max(24, canvas.clientHeight - padding * 2);
    let changed = false;
    const nextLayout = this.layout().map((term, index) => {
      const rendered = renderedTerms[index].getBoundingClientRect();
      const currentCenterX = rendered.left - canvasRect.left + rendered.width / 2;
      const currentCenterY = rendered.top - canvasRect.top + rendered.height / 2;
      const fontScale = Math.min(1, availableWidth / Math.max(rendered.width, 1), availableHeight / Math.max(rendered.height, 1));
      const nextFontSize = fontScale < 1 ? Math.max(10, Math.floor(term.fontSize * fontScale * 0.98)) : term.fontSize;
      const requiresResize = nextFontSize < term.fontSize || rendered.width > availableWidth || rendered.height > availableHeight;
      const clampedCenterX = Math.max(padding + rendered.width / 2, Math.min(canvas.clientWidth - padding - rendered.width / 2, currentCenterX));
      const clampedCenterY = Math.max(padding + rendered.height / 2, Math.min(canvas.clientHeight - padding - rendered.height / 2, currentCenterY));
      const nextX = requiresResize
        ? Math.round(canvas.clientWidth / 2)
        : Math.round(term.x + clampedCenterX - currentCenterX);
      const nextY = requiresResize
        ? Math.round(canvas.clientHeight / 2)
        : Math.round(term.y + clampedCenterY - currentCenterY);
      if (nextX !== term.x || nextY !== term.y || nextFontSize !== term.fontSize) changed = true;
      return { ...term, x: nextX, y: nextY, fontSize: nextFontSize };
    });

    if (changed) {
      this.layout.set(nextLayout);
      this.scheduleFitToCanvas();
    }
  }

  private fallbackLayout(source: readonly WordCloudTerm[], width: number, height: number): WordCloudLayoutTerm[] {
    const max = Math.max(...source.map((item) => item.weight || item.count), 1);
    return source.slice(0, 48).map((item, index) => {
      const angle = index * 2.39996;
      const radius = 18 + Math.sqrt(index) * Math.min(width, height) * 0.08;
      return {
        ...item,
        x: Math.max(32, Math.min(width - 32, width / 2 + Math.cos(angle) * radius)),
        y: Math.max(24, Math.min(height - 24, height / 2 + Math.sin(angle) * radius)),
        rotate: index % 7 === 0 ? -8 : index % 9 === 0 ? 8 : 0,
        fontSize: Math.max(12, Math.min(36, 12 + Math.round(((item.weight || item.count) / max) * 24))),
      };
    });
  }
}
