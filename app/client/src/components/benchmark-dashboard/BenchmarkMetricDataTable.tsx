import type { BenchmarkDashboardWidgetData } from '../../types/api';
import { formatBenchmarkValue } from '../../features/benchmark-dashboard/benchmarkDashboardFormatters';

type BenchmarkMetricDataTableProps = {
  widget: BenchmarkDashboardWidgetData;
};

export const BenchmarkMetricDataTable = ({ widget }: BenchmarkMetricDataTableProps) => {
  if (widget.histogram_bins.length) {
    return (
      <details className="benchmark-dashboard-data-table">
        <summary>View data table</summary>
        <div>
          <table>
            <caption>{widget.label} histogram bins</caption>
            <thead><tr><th scope="col">Tokenizer</th><th scope="col">Bin low</th><th scope="col">Bin high</th><th scope="col">Count</th><th scope="col">Proportion</th></tr></thead>
            <tbody>
              {widget.histogram_bins.map((item) => (
                <tr key={`${item.tokenizer}-${item.bin_low}-${item.bin_high}`}>
                  <th scope="row">{item.tokenizer}</th>
                  <td>{formatBenchmarkValue(item.bin_low, widget.display_format)}</td>
                  <td>{formatBenchmarkValue(item.bin_high, widget.display_format)}</td>
                  <td>{item.count.toLocaleString()}</td>
                  <td>{(item.proportion * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    );
  }

  if (widget.distributions.length) {
    return (
      <details className="benchmark-dashboard-data-table">
        <summary>View data table</summary>
        <div>
          <table>
            <caption>{widget.label} distribution values</caption>
            <thead><tr><th scope="col">Tokenizer</th><th scope="col">Min</th><th scope="col">Q1</th><th scope="col">Median</th><th scope="col">Q3</th><th scope="col">Max</th><th scope="col">Samples</th></tr></thead>
            <tbody>
              {widget.distributions.map((item) => (
                <tr key={item.tokenizer}>
                  <th scope="row">{item.tokenizer}</th>
                  <td>{formatBenchmarkValue(item.min, widget.display_format)}</td>
                  <td>{formatBenchmarkValue(item.q1, widget.display_format)}</td>
                  <td>{formatBenchmarkValue(item.median, widget.display_format)}</td>
                  <td>{formatBenchmarkValue(item.q3, widget.display_format)}</td>
                  <td>{formatBenchmarkValue(item.max, widget.display_format)}</td>
                  <td>{item.sample_count.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    );
  }

  if (widget.buckets.length) {
    return (
      <details className="benchmark-dashboard-data-table">
        <summary>View data table</summary>
        <div>
          <table>
            <caption>{widget.label} values by bucket</caption>
            <thead><tr><th scope="col">Tokenizer</th><th scope="col">Bucket</th><th scope="col">Value ({widget.unit})</th></tr></thead>
            <tbody>
              {widget.buckets.map((item) => (
                <tr key={`${item.tokenizer}-${item.bucket}`}>
                  <th scope="row">{item.tokenizer}</th>
                  <td>{item.bucket}</td>
                  <td>{formatBenchmarkValue(item.value, widget.display_format)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    );
  }

  const hasIntervals = widget.points.some((item) => item.interval_low !== null && item.interval_high !== null);
  return (
    <details className="benchmark-dashboard-data-table">
      <summary>View data table</summary>
      <div>
        <table>
          <caption>{widget.label} values by tokenizer</caption>
          <thead><tr><th scope="col">Tokenizer</th><th scope="col">Value ({widget.unit})</th>{hasIntervals && <><th scope="col">Interval low</th><th scope="col">Interval high</th></>}</tr></thead>
          <tbody>
            {widget.points.map((item) => (
              <tr key={item.tokenizer}>
                <th scope="row">{item.tokenizer}</th>
                <td>{formatBenchmarkValue(item.value, widget.display_format)}</td>
                {hasIntervals && <>
                  <td>{item.interval_low === null ? 'N/A' : formatBenchmarkValue(item.interval_low, widget.display_format)}</td>
                  <td>{item.interval_high === null ? 'N/A' : formatBenchmarkValue(item.interval_high, widget.display_format)}</td>
                </>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
};
