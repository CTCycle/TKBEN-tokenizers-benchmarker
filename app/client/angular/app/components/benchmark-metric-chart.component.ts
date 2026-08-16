import { Component, computed, input } from '@angular/core';
import type { BenchmarkDashboardWidgetData } from '../core/api/api.models';

@Component({
  selector: 'app-benchmark-metric-chart',
  templateUrl: './benchmark-metric-chart.component.html',
})
export class BenchmarkMetricChartComponent {
  readonly widget = input.required<BenchmarkDashboardWidgetData>();
  protected readonly math = Math;
  protected readonly maxValue = computed(() => Math.max(1, ...this.widget().points.map((point) => Math.abs(point.value))));
}
