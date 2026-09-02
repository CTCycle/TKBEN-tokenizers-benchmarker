import { Component, computed, input, signal } from '@angular/core';
import type { HistogramData } from '../core/api/api.models';
import { formatBenchmarkAxisValue } from '../core/utils/benchmark-dashboard-data';

type HistogramView = 'histogram' | 'cumulative';
type ExtendedHistogramData = HistogramData & {
  token_length_std?: number;
  token_length_p90?: number;
  token_length_cv?: number;
  single_character_token_percentage?: number;
};

@Component({
  selector: 'app-histogram-chart',
  templateUrl: './histogram-chart.component.html',
  styles: [`
    .histogram-view-switcher { display: flex; justify-content: flex-end; gap: 4px; margin-bottom: 8px; }
    .histogram-view-button { min-height: 28px; padding: 4px 9px; border: 1px solid var(--color-border); border-radius: 6px; background: transparent; color: var(--color-muted); font: inherit; font-size: 0.75rem; cursor: pointer; }
    .histogram-view-button[aria-pressed='true'] { border-color: var(--color-accent); color: var(--color-text); background: color-mix(in srgb, var(--color-accent) 12%, transparent); }
    .histogram-view-button:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; }
    .histogram-cdf-line { fill: none; stroke: var(--color-accent); stroke-width: 2.5; vector-effect: non-scaling-stroke; }
    .histogram-cdf-point { fill: var(--color-accent); }
    .histogram-shape-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 6px 12px; margin-top: 8px; font-size: 0.75rem; color: var(--color-muted); }
    .histogram-shape-metrics span { display: flex; justify-content: space-between; gap: 8px; }
    .histogram-shape-metrics strong { color: var(--color-text); font-weight: 600; }
  `],
})
export class HistogramChartComponent {
  readonly histogram = input.required<HistogramData>();
  readonly label = input('Histogram');
  protected readonly view = signal<HistogramView>('histogram');
  protected readonly math = Math;
  protected readonly axisFractions = [0, 0.5, 1] as const;
  protected readonly maxCount = computed(() => Math.max(1, ...this.histogram().counts));
  protected readonly totalCount = computed(() => this.histogram().counts.reduce((sum, count) => sum + Math.max(0, count), 0));
  protected readonly cumulativeValues = computed(() => {
    const total = this.totalCount();
    if (total <= 0) return this.histogram().counts.map(() => 0);
    let running = 0;
    return this.histogram().counts.map((count) => {
      running += Math.max(0, count);
      return running / total;
    });
  });
  protected readonly cumulativePoints = computed(() => this.cumulativeValues().map((value, index) => `${this.pointX(index)},${this.pointY(value)}`).join(' '));
  protected readonly shapeMetrics = computed(() => {
    const histogram = this.histogram() as ExtendedHistogramData;
    if (
      histogram.token_length_std === undefined
      && histogram.token_length_p90 === undefined
      && histogram.token_length_cv === undefined
      && histogram.single_character_token_percentage === undefined
    ) return [];
    return [
      { label: 'Std', value: this.formatMetric(histogram.token_length_std) },
      { label: 'p90', value: this.formatMetric(histogram.token_length_p90) },
      { label: 'CV', value: this.formatMetric(histogram.token_length_cv, 3) },
      { label: '1-char', value: histogram.single_character_token_percentage === undefined ? 'N/A' : `${histogram.single_character_token_percentage.toFixed(1)}%` },
    ];
  });

  protected setView(view: HistogramView): void { this.view.set(view); }
  protected tickY(fraction: number): number { return 166 - fraction * 130; }
  protected tickValue(fraction: number): string {
    return this.view() === 'cumulative'
      ? `${Math.round(fraction * 100)}%`
      : formatBenchmarkAxisValue(this.maxCount() * fraction, 'number');
  }
  protected barX(index: number): number {
    const band = 590 / Math.max(this.histogram().counts.length, 1);
    const width = this.barWidth();
    return 34 + index * band + (band - width) / 2;
  }
  protected barWidth(): number { return Math.max(3, (560 / Math.max(this.histogram().counts.length, 1)) * 0.78); }
  protected pointX(index: number): number {
    const count = Math.max(this.histogram().counts.length, 1);
    return count === 1 ? 330 : 34 + (index / (count - 1)) * 590;
  }
  protected pointY(value: number): number { return 166 - Math.max(0, Math.min(1, value)) * 130; }
  private formatMetric(value: number | undefined, digits = 2): string { return value === undefined ? 'N/A' : value.toFixed(digits); }
}
