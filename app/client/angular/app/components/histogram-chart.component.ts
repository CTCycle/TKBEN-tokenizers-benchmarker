import { Component, computed, input } from '@angular/core';
import type { HistogramData } from '../core/api/api.models';
import { formatBenchmarkAxisValue } from '../core/utils/benchmark-dashboard-data';

@Component({ selector: 'app-histogram-chart', templateUrl: './histogram-chart.component.html' })
export class HistogramChartComponent {
  readonly histogram = input.required<HistogramData>();
  readonly label = input('Histogram');
  protected readonly math = Math;
  protected readonly axisFractions = [0, 0.5, 1] as const;
  protected readonly maxCount = computed(() => Math.max(1, ...this.histogram().counts));

  protected tickY(fraction: number): number { return 166 - fraction * 130; }
  protected tickValue(fraction: number): string { return formatBenchmarkAxisValue(this.maxCount() * fraction, 'number'); }
  protected barX(index: number): number {
    const band = 590 / Math.max(this.histogram().counts.length, 1);
    const width = this.barWidth();
    return 34 + index * band + (band - width) / 2;
  }
  protected barWidth(): number { return Math.max(3, (560 / Math.max(this.histogram().counts.length, 1)) * 0.78); }
}
