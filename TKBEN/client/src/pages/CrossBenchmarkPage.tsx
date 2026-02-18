import { useEffect, useMemo, useState } from 'react';
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
import { fetchAvailableDatasets } from '../services/datasetsApi';
import { fetchDownloadedTokenizers } from '../services/tokenizersApi';
import {
  fetchBenchmarkMetricsCatalog,
  fetchBenchmarkReportById,
  fetchBenchmarkReports,
  runBenchmarks,
} from '../services/benchmarksApi';
import type {
  BenchmarkMetricCatalogCategory,
  BenchmarkReportSummary,
  BenchmarkRunResponse,
  BenchmarkPerDocumentTokenizerStats,
  GlobalMetrics,
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
const formatTokenizerLabel = (tokenizer: string): string => {
  const trimmed = tokenizer.trim();
  if (!trimmed) {
    return 'N/A';
  }
  const segments = trimmed.split('/');
  return segments[segments.length - 1] || trimmed;
};
const metricLabelOverrides: Record<string, string> = {
  tokenization_speed_tps: 'Tokenization Speed (tokens per second)',
  throughput_chars_per_sec: 'Character Throughput (characters per second)',
  processing_time_seconds: 'Processing Time (seconds)',
  avg_sequence_length: 'Average Sequence Length',
  median_sequence_length: 'Median Sequence Length',
  subword_fertility: 'Subword fertility',
  word_recovery_rate: 'Word Recovery Rate',
  character_coverage: 'Character Coverage Rate',
  model_size_mb: 'Model size (MB)',
  segmentation_consistency: 'Segmentation Consistency',
  token_distribution_entropy: 'Token Distribution Entropy',
  rare_token_tail_1: 'Rare Token Tail 1 Count',
  rare_token_tail_2: 'Rare Token Tail 2 Count',
  compression_chars_per_token: 'Compression (characters per token)',
  compression_bytes_per_character: 'Compression (bytes per character)',
  round_trip_text_fidelity_rate: 'Round-Trip Text Fidelity Rate',
  token_id_ordering_monotonicity: 'Token ID Ordering Monotonicity',
  token_unigram_coverage: 'Token Unigram Coverage Rate',
};
const toMetricLabel = (key: string): string =>
  metricLabelOverrides[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, (ch) => ch.toUpperCase());
const isLikelyRatioMetric = (key: string): boolean =>
  key.includes('rate') || key.includes('coverage') || key.endsWith('_percentage');
const formatMetricValue = (key: string, rawValue: unknown): string => {
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

const quantile = (values: number[], q: number): number => {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  const next = sorted[base + 1];
  if (next === undefined) {
    return sorted[base] ?? 0;
  }
  return (sorted[base] ?? 0) + rest * (next - (sorted[base] ?? 0));
};

const calcDistribution = (values: number[]) => {
  if (values.length === 0) {
    return {
      min: 0,
      q1: 0,
      median: 0,
      q3: 0,
      max: 0,
    };
  }
  const sorted = [...values].sort((a, b) => a - b);
  return {
    min: sorted[0] ?? 0,
    q1: quantile(sorted, 0.25),
    median: quantile(sorted, 0.5),
    q3: quantile(sorted, 0.75),
    max: sorted[sorted.length - 1] ?? 0,
  };
};

const CrossBenchmarkPage = () => {
  const [tokenizers, setTokenizers] = useState<string[]>([]);
  const [datasets, setDatasets] = useState<string[]>([]);
  const [metricCategories, setMetricCategories] = useState<BenchmarkMetricCatalogCategory[]>([]);
  const [reports, setReports] = useState<BenchmarkReportSummary[]>([]);
  const [selectedReportId, setSelectedReportId] = useState<number | null>(null);
  const [activeReport, setActiveReport] = useState<BenchmarkRunResponse | null>(null);
  const [loadingPage, setLoadingPage] = useState(true);
  const [loadingReport, setLoadingReport] = useState(false);
  const [runningBenchmark, setRunningBenchmark] = useState(false);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedDistributionTokenizer, setSelectedDistributionTokenizer] = useState<string>('');

  const loadReportById = async (reportId: number) => {
    setLoadingReport(true);
    try {
      const report = await fetchBenchmarkReportById(reportId);
      setError(null);
      setActiveReport(report);
      setSelectedReportId(reportId);
      const firstDistributionTokenizer = report.chart_data.token_length_distributions?.[0]?.tokenizer ?? '';
      setSelectedDistributionTokenizer(firstDistributionTokenizer);
    } catch (loadError) {
      const message = loadError instanceof Error ? loadError.message : 'Failed to load report';
      setError(message);
    } finally {
      setLoadingReport(false);
    }
  };

  const refreshReportSummaries = async (preferredReportId?: number | null) => {
    const listResponse = await fetchBenchmarkReports(200);
    const list = listResponse.reports ?? [];
    setReports(list);
    const targetReportId = preferredReportId ?? selectedReportId ?? list[0]?.report_id ?? null;
    if (targetReportId) {
      await loadReportById(targetReportId);
    } else {
      setActiveReport(null);
      setSelectedReportId(null);
    }
  };

  useEffect(() => {
    const loadInitial = async () => {
      setLoadingPage(true);
      setError(null);
      try {
        const [tokenizerResponse, datasetResponse, categoryResponse] = await Promise.all([
          fetchDownloadedTokenizers(),
          fetchAvailableDatasets(),
          fetchBenchmarkMetricsCatalog(),
        ]);
        setTokenizers(tokenizerResponse.tokenizers.map((item) => item.tokenizer_name));
        setDatasets(datasetResponse.datasets.map((item) => item.dataset_name));
        setMetricCategories(categoryResponse.categories ?? []);
        await refreshReportSummaries();
      } catch (loadError) {
        const message = loadError instanceof Error ? loadError.message : 'Failed to load benchmark workspace';
        setError(message);
      } finally {
        setLoadingPage(false);
      }
    };
    void loadInitial();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRunFromWizard = async (payload: {
    tokenizers: string[];
    dataset_name: string;
    max_documents: number;
    run_name: string;
    selected_metric_keys: string[];
  }) => {
    setRunningBenchmark(true);
    setError(null);
    try {
      const report = await runBenchmarks(payload);
      setActiveReport(report);
      setWizardOpen(false);
      await refreshReportSummaries(report.report_id);
    } catch (runError) {
      const message = runError instanceof Error ? runError.message : 'Failed to run benchmark';
      setError(message);
    } finally {
      setRunningBenchmark(false);
    }
  };

  const speedChartData = useMemo(() => {
    if (!activeReport?.chart_data?.speed_metrics) {
      return [];
    }
    return activeReport.chart_data.speed_metrics.map((item) => ({
      tokenizer: item.tokenizer,
      tokens_per_second: Math.round(toNumber(item.tokens_per_second)),
      chars_per_second: Math.round(toNumber(item.chars_per_second)),
      processing_time_seconds: Number(toNumber(item.processing_time_seconds).toFixed(4)),
    }));
  }, [activeReport]);

  const qualityChartData = useMemo(() => {
    if (!activeReport?.global_metrics) {
      return [];
    }
    return activeReport.global_metrics.map((item) => ({
      tokenizer: item.tokenizer,
      oov_rate: Number(normalizeRate(toNumber(item.oov_rate)).toFixed(4)),
      determinism_rate: Number(normalizeRate(toNumber(item.determinism_rate)).toFixed(4)),
      boundary_preservation_rate: Number(normalizeRate(toNumber(item.boundary_preservation_rate)).toFixed(4)),
      round_trip_fidelity_rate: Number(normalizeRate(toNumber(item.round_trip_fidelity_rate)).toFixed(4)),
    }));
  }, [activeReport]);

  const vocabularyChartData = useMemo(() => {
    if (!activeReport?.chart_data?.vocabulary_stats) {
      return [];
    }
    return activeReport.chart_data.vocabulary_stats.map((item) => ({
      tokenizer: item.tokenizer,
      vocabulary_size: toNumber(item.vocabulary_size),
      subwords_count: toNumber(item.subwords_count),
      true_words_count: toNumber(item.true_words_count),
      subwords_percentage: toNumber(item.subwords_percentage),
    }));
  }, [activeReport]);

  const distributionSeries = useMemo(() => {
    if (!activeReport?.chart_data?.token_length_distributions) {
      return [];
    }
    const selectedTokenizerName = selectedDistributionTokenizer
      || activeReport.chart_data.token_length_distributions[0]?.tokenizer
      || '';
    const distribution = activeReport.chart_data.token_length_distributions.find(
      (item) => item.tokenizer === selectedTokenizerName,
    );
    if (!distribution) {
      return [];
    }
    return distribution.bins.map((bin) => ({
      label: `${bin.bin_start}-${bin.bin_end}`,
      count: toNumber(bin.count),
    }));
  }, [activeReport, selectedDistributionTokenizer]);

  const bytesPerTokenBoxData = useMemo(() => {
    if (!activeReport?.per_document_stats?.length) {
      return [];
    }
    return activeReport.per_document_stats.map((item: BenchmarkPerDocumentTokenizerStats) => {
      const distribution = calcDistribution(
        item.bytes_per_token
          .map((value) => toNumber(value))
          .filter((value) => Number.isFinite(value) && value >= 0),
      );
      return {
        tokenizer: item.tokenizer,
        ...distribution,
      };
    });
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

  const overviewMetrics = useMemo(() => {
    if (!activeReport) {
      return [];
    }
    const global = activeReport.global_metrics ?? [];
    const bestSpeed = global.reduce<GlobalMetrics | null>((best, item) => {
      if (!best) return item;
      return toNumber(item.tokenization_speed_tps) > toNumber(best.tokenization_speed_tps) ? item : best;
    }, null);
    const bestOov = global.reduce<GlobalMetrics | null>((best, item) => {
      if (!best) return item;
      return toNumber(item.oov_rate) < toNumber(best.oov_rate) ? item : best;
    }, null);
    const bestRoundTrip = global.reduce<GlobalMetrics | null>((best, item) => {
      if (!best) return item;
      return toNumber(item.round_trip_fidelity_rate) > toNumber(best.round_trip_fidelity_rate) ? item : best;
    }, null);
    return [
      { label: 'Dataset', value: activeReport.dataset_name || 'N/A' },
      { label: 'Run Name', value: activeReport.run_name || 'N/A' },
      { label: 'Documents', value: activeReport.documents_processed.toLocaleString() },
      { label: 'Tokenizers', value: activeReport.tokenizers_count.toLocaleString() },
      {
        label: 'Best Speed',
        value: bestSpeed ? `${Math.round(toNumber(bestSpeed.tokenization_speed_tps)).toLocaleString()} tok/s` : 'N/A',
        detail: bestSpeed ? formatTokenizerLabel(bestSpeed.tokenizer) : 'N/A',
      },
      {
        label: 'Best OOV',
        value: bestOov ? formatRate(toNumber(bestOov.oov_rate)) : 'N/A',
        detail: bestOov ? formatTokenizerLabel(bestOov.tokenizer) : 'N/A',
      },
      {
        label: 'Best Round Trip',
        value: bestRoundTrip ? formatRate(toNumber(bestRoundTrip.round_trip_fidelity_rate)) : 'N/A',
        detail: bestRoundTrip ? formatTokenizerLabel(bestRoundTrip.tokenizer) : 'N/A',
      },
    ];
  }, [activeReport]);

  const additionalMetricColumns = useMemo(() => {
    if (!activeReport?.global_metrics?.length) {
      return [] as string[];
    }
    const excluded = new Set([
      'tokenizer',
      'dataset_name',
      'tokenization_speed_tps',
      'throughput_chars_per_sec',
      'processing_time_seconds',
      'oov_rate',
      'determinism_rate',
      'boundary_preservation_rate',
      'round_trip_fidelity_rate',
      'vocabulary_size',
    ]);
    const preferredOrder = [
      'word_recovery_rate',
      'character_coverage',
      'subword_fertility',
      'avg_sequence_length',
      'median_sequence_length',
      'model_size_mb',
      'segmentation_consistency',
      'token_distribution_entropy',
      'rare_token_tail_1',
      'rare_token_tail_2',
      'compression_chars_per_token',
      'compression_bytes_per_character',
      'round_trip_text_fidelity_rate',
      'token_id_ordering_monotonicity',
      'token_unigram_coverage',
    ];
    const discovered = new Set<string>();
    activeReport.global_metrics.forEach((metric) => {
      Object.entries(metric).forEach(([key, value]) => {
        if (excluded.has(key)) {
          return;
        }
        if (typeof value === 'number' && Number.isFinite(value)) {
          discovered.add(key);
        }
      });
    });
    const ordered: string[] = [];
    preferredOrder.forEach((key) => {
      if (discovered.has(key)) {
        ordered.push(key);
        discovered.delete(key);
      }
    });
    return [...ordered, ...Array.from(discovered).sort()];
  }, [activeReport]);

  const renderUnavailable = (label: string) => (
    <div className="chart-placeholder">
      <p>{label}</p>
      <span>Not available in this report.</span>
    </div>
  );

  return (
    <div className="page-scroll cross-benchmark-scroll">
      <div className="page-grid cross-benchmark-page">
        <section className="cross-benchmark-control-panel">
          <div className="cross-benchmark-control-header">
            <div>
              <p className="panel-label">Tokenizer Benchmark</p>
              <p className="panel-description">
                Run and reopen persisted tokenizer benchmark reports with metric-level dashboards.
              </p>
            </div>
          </div>
          <div className="cross-benchmark-control-actions">
            <button
              type="button"
              className="primary-button"
              onClick={() => setWizardOpen(true)}
              disabled={runningBenchmark || loadingPage}
            >
              {runningBenchmark ? 'Running...' : 'Start benchmark'}
            </button>
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
          {error && (
            <div className="error-banner" role="alert">
              <span>{error}</span>
              <button type="button" onClick={() => setError(null)} aria-label="Dismiss error">×</button>
            </div>
          )}
        </section>

        <section className="panel dashboard-panel dashboard-plain cross-benchmark-dashboard">
          <header className="panel-header">
            <div>
              <p className="panel-label">Benchmark Dashboard</p>
              <p className="panel-description">
                {activeReport
                  ? `Report #${activeReport.report_id ?? 'N/A'}${activeReport.created_at ? ` • ${new Date(activeReport.created_at).toLocaleString()}` : ''}`
                  : 'Select a report or run a benchmark to populate this dashboard.'}
              </p>
            </div>
          </header>

          {(loadingPage || loadingReport) && (
            <div className="loading-container">
              <div className="spinner" />
              <p>{loadingPage ? 'Loading benchmark workspace...' : 'Loading report...'}</p>
            </div>
          )}

          {!loadingPage && !activeReport && (
            <div className="chart-placeholder">
              <p>No benchmark report loaded.</p>
              <span>Use Start benchmark or select an existing report.</span>
            </div>
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
                    <p className="panel-label">Speed Comparison</p>
                  </div>
                  {speedChartData.length === 0 ? (
                    renderUnavailable('Speed metrics unavailable')
                  ) : (
                    <ResponsiveContainer width="100%" height={340}>
                      <BarChart
                        data={speedChartData}
                        margin={{ top: 10, right: 16, left: 4, bottom: 56 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3440" />
                        <XAxis
                          dataKey="tokenizer"
                          stroke="#9ea7b3"
                          interval={0}
                          tick={{ fontSize: 11 }}
                          tickFormatter={formatTokenizerLabel}
                          angle={-20}
                          textAnchor="end"
                          height={72}
                        />
                        <YAxis stroke="#9ea7b3" width={78} tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }} />
                        <Legend wrapperStyle={{ fontSize: 12 }} />
                        <Bar dataKey="tokens_per_second" fill="#4fc3f7" name="Tokenization Speed (tokens/sec)" />
                        <Bar dataKey="chars_per_second" fill="#81c784" name="Character Throughput (chars/sec)" />
                        <Bar dataKey="processing_time_seconds" fill="#ffb74d" name="Processing Time (seconds)" />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </article>

                <article className="cross-benchmark-chart-card">
                  <div className="cross-benchmark-chart-header">
                    <p className="panel-label">Global Rates</p>
                  </div>
                  {qualityChartData.length === 0 ? (
                    renderUnavailable('Global rate metrics unavailable')
                  ) : (
                    <ResponsiveContainer width="100%" height={340}>
                      <BarChart
                        data={qualityChartData}
                        margin={{ top: 10, right: 16, left: 4, bottom: 56 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3440" />
                        <XAxis
                          dataKey="tokenizer"
                          stroke="#9ea7b3"
                          interval={0}
                          tick={{ fontSize: 11 }}
                          tickFormatter={formatTokenizerLabel}
                          angle={-20}
                          textAnchor="end"
                          height={72}
                        />
                        <YAxis stroke="#9ea7b3" width={78} tick={{ fontSize: 11 }} />
                        <Tooltip
                          formatter={(value: number | string | undefined) => `${toNumber(value).toFixed(2)}%`}
                          contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
                        />
                        <Legend wrapperStyle={{ fontSize: 12 }} />
                        <Bar dataKey="oov_rate" fill="#f87171" name="Out-of-Vocabulary Rate (%)" />
                        <Bar dataKey="determinism_rate" fill="#22c55e" name="Tokenization Determinism Rate (%)" />
                        <Bar dataKey="boundary_preservation_rate" fill="#38bdf8" name="Word Boundary Preservation Rate (%)" />
                        <Bar dataKey="round_trip_fidelity_rate" fill="#facc15" name="Round-Trip Fidelity Rate (%)" />
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
                    <ResponsiveContainer width="100%" height={340}>
                      <ComposedChart
                        data={vocabularyChartData}
                        margin={{ top: 10, right: 16, left: 4, bottom: 56 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3440" />
                        <XAxis
                          dataKey="tokenizer"
                          stroke="#9ea7b3"
                          interval={0}
                          tick={{ fontSize: 11 }}
                          tickFormatter={formatTokenizerLabel}
                          angle={-20}
                          textAnchor="end"
                          height={72}
                        />
                        <YAxis stroke="#9ea7b3" width={78} tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }} />
                        <Legend wrapperStyle={{ fontSize: 12 }} />
                        <Bar dataKey="vocabulary_size" fill="#4fc3f7" name="Vocabulary Size (tokens)" />
                        <Bar dataKey="subwords_count" fill="#ffb74d" name="Subword Token Count" />
                        <Bar dataKey="true_words_count" fill="#81c784" name="Whole Word Token Count" />
                      </ComposedChart>
                    </ResponsiveContainer>
                  )}
                </article>
              </div>

              <div className="cross-benchmark-chart-grid cross-benchmark-chart-grid-secondary">
                <article className="cross-benchmark-chart-card">
                  <div className="cross-benchmark-chart-header">
                    <p className="panel-label">Token Length Distribution</p>
                    <select
                      className="text-input cross-benchmark-inline-select"
                      value={selectedDistributionTokenizer}
                      onChange={(event) => setSelectedDistributionTokenizer(event.target.value)}
                    >
                      {(activeReport.chart_data.token_length_distributions ?? []).map((entry) => (
                        <option key={entry.tokenizer} value={entry.tokenizer}>{entry.tokenizer}</option>
                      ))}
                    </select>
                  </div>
                  {distributionSeries.length === 0 ? (
                    renderUnavailable('Token length distribution unavailable')
                  ) : (
                    <ResponsiveContainer width="100%" height={320}>
                      <BarChart
                        data={distributionSeries}
                        margin={{ top: 10, right: 16, left: 4, bottom: 30 }}
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

                <article className="cross-benchmark-chart-card">
                  <div className="cross-benchmark-chart-header">
                    <p className="panel-label">Bytes per Token Distribution (Box Plot)</p>
                  </div>
                  {bytesPerTokenBoxData.length === 0 ? (
                    renderUnavailable('Per-document bytes per token unavailable')
                  ) : (
                    <>
                      <div className="cross-benchmark-boxplot-scale">
                        <span>{bytesPerTokenGlobalRange.min.toFixed(3)}</span>
                        <span>{bytesPerTokenGlobalRange.max.toFixed(3)}</span>
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
                          return (
                            <div key={entry.tokenizer} className="cross-benchmark-boxplot-row">
                              <span className="cross-benchmark-boxplot-label">{entry.tokenizer}</span>
                              <div className="cross-benchmark-boxplot-track">
                                <div
                                  className="cross-benchmark-boxplot-whisker"
                                  style={{ left: whiskerLeft, right: whiskerRight }}
                                />
                                <div
                                  className="cross-benchmark-boxplot-box"
                                  style={{ left: boxLeft, width: boxWidth }}
                                />
                                <div className="cross-benchmark-boxplot-median" style={{ left: medianLeft }} />
                              </div>
                              <span className="cross-benchmark-boxplot-values">
                                {entry.min.toFixed(3)} / {entry.median.toFixed(3)} / {entry.max.toFixed(3)}
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
                  {activeReport.global_metrics.length === 0 || additionalMetricColumns.length === 0 ? (
                    renderUnavailable('No non-plotted additional metrics available')
                  ) : (
                    <div className="cross-benchmark-table-wrap">
                      <table className="tokenizer-meta-table tokenizer-meta-table-compact">
                        <thead>
                          <tr>
                            <th>Tokenizer</th>
                            {additionalMetricColumns.map((metricKey) => (
                              <th key={metricKey}>{toMetricLabel(metricKey)}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {activeReport.global_metrics.map((metric) => (
                            <tr key={`additional-${metric.tokenizer}`}>
                              <td>{metric.tokenizer}</td>
                              {additionalMetricColumns.map((metricKey) => (
                                <td key={`${metric.tokenizer}-${metricKey}`}>
                                  {formatMetricValue(metricKey, (metric as unknown as Record<string, unknown>)[metricKey])}
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
        defaultMaxDocuments={activeReport?.documents_processed ?? 1000}
        running={runningBenchmark}
        onClose={() => setWizardOpen(false)}
        onRun={handleRunFromWizard}
      />
    </div>
  );
};

export default CrossBenchmarkPage;
