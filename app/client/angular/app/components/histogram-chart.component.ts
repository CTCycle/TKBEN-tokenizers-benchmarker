import { Component, computed, input } from '@angular/core';
import type { HistogramData } from '../core/api/api.models';

@Component({ selector: 'app-histogram-chart', templateUrl: './histogram-chart.component.html' })
export class HistogramChartComponent {
  readonly histogram = input.required<HistogramData>();
  readonly label = input('Histogram');
  protected readonly math = Math;
  protected readonly maxCount = computed(() => Math.max(1, ...this.histogram().counts));
}
