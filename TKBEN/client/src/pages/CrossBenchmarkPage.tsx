import { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import BenchmarkRunWizard from '../components/BenchmarkRunWizard';
import ChartPlaceholder from '../components/ChartPlaceholder';
import DashboardExportButton from '../components/DashboardExportButton';
import DismissibleBanner from '../components/DismissibleBanner';
import { useBenchmarkWorkspace } from '../hooks/useBenchmarkWorkspace';
import type { BenchmarkRunPayload } from '../hooks/useBenchmarkWorkspace';
import type {
  BenchmarkTokenizerResult,
} from '../types/api';

const BAR_COLORS = ['#4fc3f7', '#81c784', '#ffb74d', '#f06292', '#ba68c8', '#4db6ac'];

const toNumber = (value: unknown): number => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return 0;
};

const normalizeRate = (value: number): number => {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return value > 1 ? value : value * 100;
};

const formatRate = (value: number): string => `${normalizeRate(value).toFixed(2)}%`;
const truncateText = (value: string, maxLength: number): string => {
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, Math.max(0, maxLength - 3))}...`;
};
const formatTokenizerLabel = (tokenizer: string): string => {
  const trimmed = tokenizer.trim();
  if (!trimmed) {
    return 'N/A';
  }
  const segments = trimmed.split('/');
  return segments[segments.length - 1] || trimmed;
};
const formatChartTokenizerLabel = (tokenizer: string): string =>
  truncateText(formatTokenizerLabel(tokenizer), 22);
const chartLegendProps = {
  layout: 'vertical' as const,
  align: 'center' as const,
  verticalAlign: 'bottom' as const,
  wrapperStyle: { fontSize: 12, paddingTop: 2, width: '100%' },
  height: 56,
};
const PRIMARY_CHART_HEIGHT = 300;
const SECONDARY_CHART_HEIGHT = 300;

type AdditionalMetricKey =
  | 'tokens_per_character'
  | 'characters_per_token'
  | 'tokens_per_byte'
  | 'bytes_per_token'
  | 'pieces_per_word_mean'
  | 'encode_latency_p50_ms'
  | 'encode_latency_p95_ms'
  | 'encode_latency_p99_ms'
  | 'memory_delta_mb';

const metricLabelOverrides: Partial<Record<AdditionalMetricKey, string>> = {
  tokens_per_character: 'Tokens / Character',
  characters_per_token: 'Characters / Token',
  tokens_per_byte: 'Tokens / Byte',
  bytes_per_token: 'Bytes / Token',
  pieces_per_word_mean: 'Pieces / Word',
  encode_latency_p50_ms: 'Latency p50 (ms)',
  encode_latency_p95_ms: 'Latency p95 (ms)',
  encode_latency_p99_ms: 'Latency p99 (ms)',
  memory_delta_mb: 'Memory Delta (MB)',
};
const toMetricLabel = (key: AdditionalMetricKey): string =>
  metricLabelOverrides[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, (ch) => ch.toUpperCase());
const isLikelyRatioMetric = (key: AdditionalMetricKey): boolean =>
  key.includes('rate') || key.includes('coverage') || key.endsWith('_percentage');
const formatMetricValue = (key: AdditionalMetricKey, rawValue: unknown): string => {
  let numeric: number | null = null;
  if (typeof rawValue === 'number' && Number.isFinite(rawValue)) {
    numeric = rawValue;
  } else if (typeof rawValue === 'string') {
    const parsed = Number(rawValue);
    if (Number.isFinite(parsed)) {
      numeric = parsed;
    }
  }
  if (numeric === null) {
    return 'N/A';
  }
  if (!Number.isFinite(numeric)) {
    return 'N/A';
  }
  if (isLikelyRatioMetric(key)) {
    return formatRate(numeric);
  }
  if (Number.isInteger(numeric)) {
    return numeric.toLocaleString();
  }
  if (Math.abs(numeric) >= 1000) {
    return numeric.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return numeric.toFixed(4);
};
const preferredAdditionalMetricOrder: AdditionalMetricKey[] = [
  'tokens_per_character',
  'characters_per_token',
  'tokens_per_byte',
  'bytes_per_token',
  'pieces_per_word_mean',
  'encode_latency_p50_ms',
  'encode_latency_p95_ms',
  'encode_latency_p99_ms',
  'memory_delta_mb',
];
const CrossBenchmarkPage = () => {
  const {
    tokenizers,
    datasets,
    metricCategories,
    reports,
    selectedReportId,
    activeReport,
    loadingPage,
    loadingReport,
    runningBenchmark,
    error,
    clearError,
    loadReportById,
    runFromWizard,
  } = useBenchmarkWorkspace();
  const [wizardOpen, setWizardOpen] = useState(false);
  const [selectedDistributionTokenizers, setSelectedDistributionTokenizers] = useState<Record<number, string>>({});
  const distributionSelectionKey = activeReport?.report_id ?? -1;
  const selectedDistributionTokenizer = selectedDistributionTokenizers[distributionSelectionKey]
    ?? activeReport?.chart_data.fragmentation?.[0]?.tokenizer
    ?? '';

  const handleRunFromWizard = async (payload: BenchmarkRunPayload) => {
    const completed = await runFromWizard(payload);
    if (completed) {
      setWizardOpen(false);
    }
  };

  const speedChartData = useMemo(() => {
    if (!activeReport?.chart_data?.efficiency) {
      return [];
    }
    return activeReport.chart_data.efficiency.map((item) => ({
      tokenizer: item.tokenizer,
      tokens_per_second: Math.round(toNumber(item.value)),
      chars_per_second: Math.round(toNumber(activeReport.tokenizer_results.find((r) => r.tokenizer === item.tokenizer)?.efficiency.encode_chars_per_second_mean)),
      ci95_low: toNumber(item.ci95_low),
      ci95_high: toNumber(item.ci95_high),
    }));
  }, [activeReport]);

  const qualityChartData = useMemo(() => {
    if (!activeReport?.chart_data?.fidelity) {
      return [];
    }
    return activeReport.chart_data.fidelity.map((item) => ({
      tokenizer: item.tokenizer,
      unknown_token_rate: Number(normalizeRate(toNumber(activeReport.tokenizer_results.find((r) => r.tokenizer === item.tokenizer)?.fidelity.unknown_token_rate)).toFixed(4)),
      byte_fallback_rate: Number(normalizeRate(toNumber(activeReport.tokenizer_results.find((r) => r.tokenizer === item.tokenizer)?.fidelity.byte_fallback_rate)).toFixed(4)),
      exact_round_trip_rate: Number(normalizeRate(toNumber(activeReport.tokenizer_results.find((r) => r.tokenizer === item.tokenizer)?.fidelity.exact_round_trip_rate)).toFixed(4)),
      normalized_round_trip_rate: Number(normalizeRate(toNumber(activeReport.tokenizer_results.find((r) => r.tokenizer === item.tokenizer)?.fidelity.normalized_round_trip_rate)).toFixed(4)),
    }));
  }, [activeReport]);

  const vocabularyChartData = useMemo(() => {
    if (!activeReport?.chart_data?.vocabulary) {
      return [];
    }
    return activeReport.chart_data.vocabulary.map((item) => ({
      tokenizer: item.tokenizer,
      vocabulary_size: toNumber(item.value),
      added_tokens: toNumber(activeReport.tokenizer_results.find((r) => r.tokenizer === item.tokenizer)?.added_tokens),
      special_token_share: Number(normalizeRate(toNumber(activeReport.tokenizer_results.find((r) => r.tokenizer === item.tokenizer)?.special_token_share)).toFixed(4)),
    }));
  }, [activeReport]);

  const distributionSeries = useMemo(() => {
    if (!activeReport?.chart_data?.fragmentation) {
      return [];
    }
    const selectedTokenizerName = selectedDistributionTokenizer
      || activeReport.chart_data.fragmentation[0]?.tokenizer
      || '';
    const distribution = activeReport.chart_data.fragmentation.find(
      (item) => item.tokenizer === selectedTokenizerName,
    );
    if (!distribution) {
      return [];
    }
    const tokenizerResult = activeReport.tokenizer_results.find((row) => row.tokenizer === distribution.tokenizer);
    return (tokenizerResult?.fragmentation.fragmentation_by_word_length_bucket ?? []).map((bucket) => ({
      label: bucket.bucket,
      count: toNumber(bucket.pieces_per_word_mean),
    }));
  }, [activeReport, selectedDistributionTokenizer]);

  const bytesPerTokenBoxData = useMemo(() => {
    return activeReport?.chart_data?.latency_or_memory_distribution ?? [];
  }, [activeReport]);

  const bytesPerTokenGlobalRange = useMemo(() => {
    if (bytesPerTokenBoxData.length === 0) {
      return { min: 0, max: 1 };
    }
    const min = Math.min(...bytesPerTokenBoxData.map((row) => row.min));
    const max = Math.max(...bytesPerTokenBoxData.map((row) => row.max));
    if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
      return { min: 0, max: 1 };
    }
    return { min, max };
  }, [bytesPerTokenBoxData]);
  const boxplotTicks = useMemo(() => {
    const tickCount = 4;
    const range = Math.max(1e-9, bytesPerTokenGlobalRange.max - bytesPerTokenGlobalRange.min);
    return Array.from({ length: tickCount + 1 }, (_, index) => {
      const ratio = index / tickCount;
      return {
        position: ratio * 100,
        label: (bytesPerTokenGlobalRange.min + range * ratio).toFixed(3),
      };
    });
  }, [bytesPerTokenGlobalRange]);

  const overviewMetrics = useMemo(() => {
    if (!activeReport) {
      return [];
    }
    const tokenizerResults = activeReport.tokenizer_results ?? [];
    const bestSpeed = tokenizerResults.reduce<BenchmarkTokenizerResult | null>((best, item) => {
      if (!best) return item;
      return toNumber(item.efficiency.encode_tokens_per_second_mean) > toNumber(best.efficiency.encode_tokens_per_second_mean) ? item : best;
    }, null);
    const bestUnknown = tokenizerResults.reduce<BenchmarkTokenizerResult | null>((best, item) => {
      if (!best) return item;
      return toNumber(item.fidelity.unknown_token_rate) < toNumber(best.fidelity.unknown_token_rate) ? item : best;
    }, null);
    const bestRoundTrip = tokenizerResults.reduce<BenchmarkTokenizerResult | null>((best, item) => {
      if (!best) return item;
      return toNumber(item.fidelity.exact_round_trip_rate) > toNumber(best.fidelity.exact_round_trip_rate) ? item : best;
    }, null);
    const lowestMemory = tokenizerResults.reduce<BenchmarkTokenizerResult | null>((best, item) => {
      if (!best) return item;
      return toNumber(item.resources.peak_rss_mb) < toNumber(best.resources.peak_rss_mb) ? item : best;
    }, null);
    return [
      { label: 'Dataset', value: activeReport.dataset_name || 'N/A' },
      { label: 'Run Name', value: activeReport.run_name || 'N/A' },
      { label: 'Documents', value: activeReport.documents_processed.toLocaleString() },
      { label: 'Tokenizers', value: activeReport.tokenizers_count.toLocaleString() },
      {
        label: 'Best Encode Throughput',
        value: bestSpeed ? `${Math.round(toNumber(bestSpeed.efficiency.encode_tokens_per_second_mean)).toLocaleString()} tok/s` : 'N/A',
        detail: bestSpeed ? formatTokenizerLabel(bestSpeed.tokenizer) : 'N/A',
      },
      {
        label: 'Lowest Unknown-Token Rate',
        value: bestUnknown ? formatRate(toNumber(bestUnknown.fidelity.unknown_token_rate)) : 'N/A',
        detail: bestUnknown ? formatTokenizerLabel(bestUnknown.tokenizer) : 'N/A',
      },
      {
        label: 'Best Exact Round-Trip Rate',
        value: bestRoundTrip ? formatRate(toNumber(bestRoundTrip.fidelity.exact_round_trip_rate)) : 'N/A',
        detail: bestRoundTrip ? formatTokenizerLabel(bestRoundTrip.tokenizer) : 'N/A',
      },
      {
        label: 'Lowest Peak Memory',
        value: lowestMemory ? `${toNumber(lowestMemory.resources.peak_rss_mb).toFixed(2)} MB` : 'N/A',
        detail: lowestMemory ? formatTokenizerLabel(lowestMemory.tokenizer) : 'N/A',
      },
    ];
  }, [activeReport]);

  const additionalMetricColumns = useMemo(() => {
    if (!activeReport?.tokenizer_results?.length) {
      return [] as AdditionalMetricKey[];
    }
    const discovered = new Set<AdditionalMetricKey>(preferredAdditionalMetricOrder);
    const ordered: AdditionalMetricKey[] = [];
    preferredAdditionalMetricOrder.forEach((key) => {
      if (discovered.has(key)) {
        ordered.push(key);
        discovered.delete(key);
      }
    });
    return [...ordered, ...Array.from(discovered).sort((a, b) => a.localeCompare(b))];
  }, [activeReport]);
  const benchmarkExportReportName = useMemo(() => {
    if (!activeReport) {
      return 'benchmark-dashboard-report';
    }
    const runName = activeReport.run_name?.trim() || `${activeReport.dataset_name}-benchmark`;
    return `${runName}-report-${activeReport.report_id ?? 'latest'}`;
  }, [activeReport]);

  const renderUnavailable = (label: string) => (
    <ChartPlaceholder
      message={label}
      detail="Not available in this report."
    />
  );

  return (
    <div className="page-scroll cross-benchmark-scroll">
      <div className="page-grid cross-benchmark-page">
        <section className="panel dashboard-panel dashboard-plain cross-benchmark-dashboard">
          <header className="panel-header cross-benchmark-dashboard-header">
            <div className="cross-benchmark-dashboard-intro">
              <p className="panel-label">Tokenizer Benchmark Dashboard</p>
              <p className="panel-description">
                Run, reopen, and analyze persisted tokenizer benchmark reports through a metric-level dashboard.
              </p>
              {activeReport && (
                <p className="panel-description cross-benchmark-dashboard-meta">
                  {`Report #${activeReport.report_id ?? 'N/A'}${activeReport.created_at ? ` • ${new Date(activeReport.created_at).toLocaleString()}` : ''}`}
                </p>
              )}
            </div>
            <div className="cross-benchmark-dashboard-controls">
              <button
                type="button"
                className="primary-button cross-benchmark-start-button"
                onClick={() => setWizardOpen(true)}
                disabled={runningBenchmark || loadingPage}
              >
                {runningBenchmark ? 'Running...' : 'Start benchmark'}
              </button>
              <div className="cross-benchmark-report-picker">
                <label className="field-label" htmlFor="benchmark-report-selector">Report</label>
                <select
                  id="benchmark-report-selector"
                  className="text-input cross-benchmark-report-select"
                  value={selectedReportId ?? ''}
                  onChange={(event) => {
                    const nextReportId = Number(event.target.value);
                    if (Number.isFinite(nextReportId) && nextReportId > 0) {
                      void loadReportById(nextReportId);
                    }
                  }}
                  disabled={reports.length === 0 || loadingReport}
                >
                  {reports.length === 0 ? (
                    <option value="">No reports available</option>
                  ) : (
                    reports.map((report) => (
                      <option key={report.report_id} value={report.report_id}>
                        #{report.report_id} - {report.run_name || 'unnamed run'} - {report.dataset_name}
                      </option>
                    ))
                  )}
                </select>
              </div>
              <div className="dashboard-export-header-actions">
                <DashboardExportButton
                  dashboardType="benchmark"
                  reportName={benchmarkExportReportName}
                  dashboardPayload={activeReport
                    ? {
                      report: activeReport,
                      selected_distribution_tokenizer: selectedDistributionTokenizer,
                    }
                    : null}
                />
              </div>
            </div>
          </header>

          {error && (
            <DismissibleBanner
              message={error}
              onDismiss={clearError}
              dismissLabel="Dismiss error"
            />
          )}

          {(loadingPage || loadingReport) && (
            <div className="loading-container">
              <div className="spinner" />
              <p>{loadingPage ? 'Loading benchmark workspace...' : 'Loading report...'}</p>
            </div>
          )}

          {!loadingPage && !activeReport && (
            <ChartPlaceholder
              message="No benchmark report loaded."
              detail="Use Start benchmark or select an existing report."
            />
          )}

          {!loadingPage && activeReport && (
            <>
              <div className="cross-benchmark-overview-grid">
                {overviewMetrics.map((item) => (
                  <article key={item.label} className="cross-benchmark-kpi-card">
                    <p className="cross-benchmark-kpi-label">{item.label}</p>
                    <p className="cross-benchmark-kpi-value">{item.value}</p>
                    {'detail' in item && typeof item.detail === 'string' && (
                      <p className="cross-benchmark-kpi-detail">{item.detail}</p>
                    )}
                  </article>
                ))}
              </div>

              <div className="cross-benchmark-chart-grid">
                <article className="cross-benchmark-chart-card">
                  <div className="cross-benchmark-chart-header">
                    <p className="panel-label">Tokenization Speed</p>
                  </div>
                  {speedChartData.length === 0 ? (
                    renderUnavailable('Tokenization speed unavailable')
                  ) : (
                    <ResponsiveContainer width="100%" height={PRIMARY_CHART_HEIGHT}>
                      <BarChart
                        data={speedChartData}
                        margin={{ top: 6, right: 12, left: 2, bottom: 16 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3440" />
                        <XAxis
                          dataKey="tokenizer"
                          stroke="#9ea7b3"
                          interval="preserveStartEnd"
                          tick={{ fontSize: 11 }}
                          tickFormatter={formatChartTokenizerLabel}
                          tickMargin={8}
                          minTickGap={16}
                          height={54}
                        />
                        <YAxis stroke="#9ea7b3" width={78} tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }} />
                        <Legend {...chartLegendProps} />
                        <Bar dataKey="tokens_per_second" fill="#4fc3f7" name="Tokenization Speed (tokens/sec)" />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </article>

                <article className="cross-benchmark-chart-card">
                  <div className="cross-benchmark-chart-header">
                    <p className="panel-label">Character Throughput</p>
                  </div>
                  {speedChartData.length === 0 ? (
                    renderUnavailable('Character throughput unavailable')
                  ) : (
                    <ResponsiveContainer width="100%" height={PRIMARY_CHART_HEIGHT}>
                      <BarChart
                        data={speedChartData}
                        margin={{ top: 6, right: 12, left: 2, bottom: 16 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3440" />
                        <XAxis
                          dataKey="tokenizer"
                          stroke="#9ea7b3"
                          interval="preserveStartEnd"
                          tick={{ fontSize: 11 }}
                          tickFormatter={formatChartTokenizerLabel}
                          tickMargin={8}
                          minTickGap={16}
                          height={54}
                        />
                        <YAxis stroke="#9ea7b3" width={78} tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }} />
                        <Legend {...chartLegendProps} />
                        <Bar dataKey="chars_per_second" fill="#81c784" name="Character Throughput (chars/sec)" />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </article>

                <article className="cross-benchmark-chart-card">
                  <div className="cross-benchmark-chart-header">
                    <p className="panel-label">Fidelity and Fallback Rates</p>
                  </div>
                  {qualityChartData.length === 0 ? (
                    renderUnavailable('Global rate metrics unavailable')
                  ) : (
                    <ResponsiveContainer width="100%" height={PRIMARY_CHART_HEIGHT}>
                      <BarChart
                        data={qualityChartData}
                        margin={{ top: 6, right: 12, left: 2, bottom: 16 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3440" />
                        <XAxis
                          dataKey="tokenizer"
                          stroke="#9ea7b3"
                          interval="preserveStartEnd"
                          tick={{ fontSize: 11 }}
                          tickFormatter={formatChartTokenizerLabel}
                          tickMargin={8}
                          minTickGap={16}
                          height={54}
                        />
                        <YAxis stroke="#9ea7b3" width={78} tick={{ fontSize: 11 }} />
                        <Tooltip
                          formatter={(value: unknown) => `${toNumber(value).toFixed(2)}%`}
                          contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
                        />
                        <Legend {...chartLegendProps} />
                        <Bar dataKey="unknown_token_rate" fill="#f87171" name="Unknown Token Rate (%)" />
                        <Bar dataKey="byte_fallback_rate" fill="#22c55e" name="Byte Fallback Rate (%)" />
                        <Bar dataKey="exact_round_trip_rate" fill="#38bdf8" name="Exact Round-Trip Rate (%)" />
                        <Bar dataKey="normalized_round_trip_rate" fill="#facc15" name="Normalized Round-Trip Rate (%)" />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </article>

                <article className="cross-benchmark-chart-card">
                  <div className="cross-benchmark-chart-header">
                    <p className="panel-label">Vocabulary Comparison</p>
                  </div>
                  {vocabularyChartData.length === 0 ? (
                    renderUnavailable('Vocabulary metrics unavailable')
                  ) : (
                    <ResponsiveContainer width="100%" height={PRIMARY_CHART_HEIGHT}>
                      <ComposedChart
                        data={vocabularyChartData}
                        margin={{ top: 6, right: 12, left: 2, bottom: 16 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3440" />
                        <XAxis
                          dataKey="tokenizer"
                          stroke="#9ea7b3"
                          interval="preserveStartEnd"
                          tick={{ fontSize: 11 }}
                          tickFormatter={formatChartTokenizerLabel}
                          tickMargin={8}
                          minTickGap={16}
                          height={54}
                        />
                        <YAxis stroke="#9ea7b3" width={78} tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }} />
                        <Legend {...chartLegendProps} />
                        <Bar dataKey="vocabulary_size" fill="#4fc3f7" name="Vocabulary Size (tokens)" />
                        <Bar dataKey="added_tokens" fill="#ffb74d" name="Added Tokens" />
                        <Bar dataKey="special_token_share" fill="#81c784" name="Special Token Share (%)" />
                      </ComposedChart>
                    </ResponsiveContainer>
                  )}
                </article>
              </div>

              <div className="cross-benchmark-chart-grid cross-benchmark-chart-grid-secondary">
                <article className="cross-benchmark-chart-card">
                  <div className="cross-benchmark-chart-header">
                    <div className="cross-benchmark-chart-title">
                      <p className="panel-label">Fragmentation by Word-Length Bucket</p>
                      <p className="cross-benchmark-chart-note">
                        Empty regions are expected when few token lengths are present.
                      </p>
                    </div>
                    <select
                      className="text-input cross-benchmark-inline-select"
                      value={selectedDistributionTokenizer}
                      onChange={(event) => {
                        const value = event.target.value;
                        setSelectedDistributionTokenizers((current) => ({
                          ...current,
                          [distributionSelectionKey]: value,
                        }));
                      }}
                    >
                      {(activeReport.chart_data.fragmentation ?? []).map((entry) => (
                        <option key={entry.tokenizer} value={entry.tokenizer}>{entry.tokenizer}</option>
                      ))}
                    </select>
                  </div>
                  {distributionSeries.length === 0 ? (
                    renderUnavailable('Fragmentation distribution unavailable')
                  ) : (
                    <ResponsiveContainer width="100%" height={SECONDARY_CHART_HEIGHT}>
                      <BarChart
                        data={distributionSeries}
                        margin={{ top: 10, right: 16, left: 4, bottom: 22 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3440" />
                        <XAxis dataKey="label" stroke="#9ea7b3" hide />
                        <YAxis stroke="#9ea7b3" width={78} tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }} />
                        <Bar dataKey="count" fill="#ba68c8">
                          {distributionSeries.map((entry, index) => (
                            <Cell key={`${entry.label}-${entry.count}`} fill={BAR_COLORS[index % BAR_COLORS.length]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </article>

                <article className="cross-benchmark-chart-card cross-benchmark-chart-card--boxplot">
                  <div className="cross-benchmark-chart-header">
                    <p className="panel-label">Latency/Memory Distribution (Box Plot)</p>
                  </div>
                  {bytesPerTokenBoxData.length === 0 ? (
                    renderUnavailable('Per-document bytes per token unavailable')
                  ) : (
                    <>
                      <div className="cross-benchmark-boxplot-scale" aria-hidden="true">
                        <div className="cross-benchmark-boxplot-scale-track" />
                        {boxplotTicks.map((tick) => (
                          <span key={`boxplot-tick-${tick.position}`} style={{ left: `${tick.position}%` }}>
                            {tick.label}
                          </span>
                        ))}
                      </div>
                      <div className="cross-benchmark-boxplot-list">
                        {bytesPerTokenBoxData.map((entry) => {
                          const globalRange = Math.max(1e-9, bytesPerTokenGlobalRange.max - bytesPerTokenGlobalRange.min);
                          const toPct = (value: number) =>
                            ((value - bytesPerTokenGlobalRange.min) / globalRange) * 100;
                          const minPct = Math.max(0, Math.min(100, toPct(entry.min)));
                          const maxPct = Math.max(0, Math.min(100, toPct(entry.max)));
                          const q1Pct = Math.max(0, Math.min(100, toPct(entry.q1)));
                          const q3Pct = Math.max(0, Math.min(100, toPct(entry.q3)));
                          const medianPct = Math.max(0, Math.min(100, toPct(entry.median)));
                          const whiskerLeft = `${Math.min(minPct, maxPct)}%`;
                          const whiskerRight = `${100 - Math.max(minPct, maxPct)}%`;
                          const boxLeft = `${Math.min(q1Pct, q3Pct)}%`;
                          const boxWidth = `${Math.max(2, Math.abs(q3Pct - q1Pct))}%`;
                          const medianLeft = `${medianPct}%`;
                          const minLeft = `${minPct}%`;
                          const maxLeft = `${maxPct}%`;
                          return (
                            <div key={entry.tokenizer} className="cross-benchmark-boxplot-row">
                              <span className="cross-benchmark-boxplot-label">{entry.tokenizer}</span>
                              <div className="cross-benchmark-boxplot-track">
                                <div
                                  className="cross-benchmark-boxplot-whisker"
                                  style={{ left: whiskerLeft, right: whiskerRight }}
                                />
                                <div className="cross-benchmark-boxplot-cap" style={{ left: minLeft }} />
                                <div className="cross-benchmark-boxplot-cap" style={{ left: maxLeft }} />
                                <div
                                  className="cross-benchmark-boxplot-box"
                                  style={{ left: boxLeft, width: boxWidth }}
                                />
                                <div className="cross-benchmark-boxplot-median" style={{ left: medianLeft }} />
                              </div>
                              <span className="cross-benchmark-boxplot-values">
                                  {entry.min.toFixed(3)} / {entry.q1.toFixed(3)} / {entry.median.toFixed(3)} / {entry.q3.toFixed(3)} / {entry.max.toFixed(3)}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </>
                  )}
                </article>
              </div>

              <div className="cross-benchmark-drilldown-grid">
                <article className="cross-benchmark-drilldown-card">
                  <div className="cross-benchmark-chart-header">
                    <p className="panel-label">Per Tokenizer Additional Metrics</p>
                  </div>
                  {activeReport.tokenizer_results.length === 0 || additionalMetricColumns.length === 0 ? (
                    renderUnavailable('No non-plotted additional metrics available')
                  ) : (
                    <div className="cross-benchmark-table-wrap">
                      <table className="tokenizer-meta-table tokenizer-meta-table-compact">
                        <thead>
                          <tr>
                            <th>Tokenizer</th>
                            {additionalMetricColumns.map((metricKey) => (
                              <th key={metricKey} title={toMetricLabel(metricKey)}>{toMetricLabel(metricKey)}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {activeReport.tokenizer_results.map((metric) => (
                            <tr key={`additional-${metric.tokenizer}`}>
                              <td className="cross-benchmark-title-cell" title={metric.tokenizer}>
                                {truncateText(metric.tokenizer, 34)}
                              </td>
                              {additionalMetricColumns.map((metricKey) => (
                                <td key={`${metric.tokenizer}-${metricKey}`}>
                                  {formatMetricValue(metricKey, {
                                    tokens_per_character: metric.fragmentation.tokens_per_character,
                                    characters_per_token: metric.fragmentation.characters_per_token,
                                    tokens_per_byte: metric.fragmentation.tokens_per_byte,
                                    bytes_per_token: metric.fragmentation.bytes_per_token,
                                    pieces_per_word_mean: metric.fragmentation.pieces_per_word_mean,
                                    encode_latency_p50_ms: metric.latency.encode_latency_p50_ms,
                                    encode_latency_p95_ms: metric.latency.encode_latency_p95_ms,
                                    encode_latency_p99_ms: metric.latency.encode_latency_p99_ms,
                                    memory_delta_mb: metric.resources.memory_delta_mb,
                                  }[metricKey])}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </article>
              </div>
            </>
          )}
        </section>
      </div>

      <BenchmarkRunWizard
        isOpen={wizardOpen}
        categories={metricCategories}
        availableTokenizers={tokenizers}
        availableDatasets={datasets}
        defaultDatasetName={activeReport?.dataset_name ?? datasets[0] ?? null}
        defaultMaxDocuments={activeReport?.config?.max_documents ?? activeReport?.documents_processed ?? 1000}
        running={runningBenchmark}
        onClose={() => setWizardOpen(false)}
        onRun={handleRunFromWizard}
      />
    </div>
  );
};

export default CrossBenchmarkPage;

