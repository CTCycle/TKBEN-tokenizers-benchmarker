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
  }

  ngOnDestroy(): void {
    this.resizeObserver?.disconnect();
    this.worker?.terminate();
  }

  private layoutTerms(): void {
    const source = [...this.terms()].filter((item) => item.word && item.count > 0).slice(0, 140);
    if (!source.length) {
      this.layout.set([]);
      this.fallback.set(false);
      return;
    }
    const element = this.canvas?.nativeElement;
    const width = Math.max(240, Math.round(element?.clientWidth || 520));
    const height = Math.max(220, Math.round(element?.clientHeight || 300));
    if (typeof Worker === 'undefined') {
      this.layout.set(this.fallbackLayout(source, width, height));
      this.fallback.set(true);
      return;
    }
    try {
      this.worker?.terminate();
      this.worker = new Worker(new URL('../core/workers/word-cloud.worker.ts', import.meta.url), { type: 'module' });
      this.worker.onmessage = (event: MessageEvent<{ terms: WordCloudLayoutTerm[] }>) => {
        this.layout.set(event.data.terms);
        this.fallback.set(false);
      };
      this.worker.onerror = () => {
        this.layout.set(this.fallbackLayout(source, width, height));
        this.fallback.set(true);
      };
      this.worker.postMessage({ terms: source, width, height });
    } catch {
      this.layout.set(this.fallbackLayout(source, width, height));
      this.fallback.set(true);
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
