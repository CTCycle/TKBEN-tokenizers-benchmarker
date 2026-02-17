import { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
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

  const selectedMetricKeySet = useMemo(() => {
    if (!activeReport?.selected_metric_keys?.length) {
      return null;
    }
    return new Set(activeReport.selected_metric_keys);
  }, [activeReport?.selected_metric_keys]);

  const isMetricEnabled = (keys: string[]) =>
    selectedMetricKeySet === null || keys.some((key) => selectedMetricKeySet.has(key));

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

  const violinEquivalentData = useMemo(() => {
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

  const overviewMetrics = useMemo(() => {
    if (!activeReport) {
      return [];
    }
    const global = activeReport.global_metrics ?? [];
    const avgTokensPerSecond = global.length > 0
      ? global.reduce((acc, item) => acc + toNumber(item.tokenization_speed_tps), 0) / global.length
      : 0;
    const avgOov = global.length > 0
      ? global.reduce((acc, item) => acc + normalizeRate(toNumber(item.oov_rate)), 0) / global.length
      : 0;
    return [
      { label: 'Dataset', value: activeReport.dataset_name || 'N/A' },
      { label: 'Documents', value: activeReport.documents_processed.toLocaleString() },
      { label: 'Tokenizers', value: activeReport.tokenizers_count.toLocaleString() },
      { label: 'Avg Speed', value: `${Math.round(avgTokensPerSecond).toLocaleString()} tok/s` },
      { label: 'Avg OOV', value: `${avgOov.toFixed(2)}%` },
      { label: 'Run Name', value: activeReport.run_name || 'N/A' },
    ];
  }, [activeReport]);

  const internalMetricColumns = [
    { key: 'segmentation_consistency', label: 'Segmentation Consistency' },
    { key: 'token_distribution_entropy', label: 'Token Entropy' },
    { key: 'compression_chars_per_token', label: 'Chars/Token' },
    { key: 'compression_bytes_per_character', label: 'Bytes/Char' },
    { key: 'token_unigram_coverage', label: 'Unigram Coverage' },
  ] as const;

  const renderUnavailable = (label: string) => (
    <div className="chart-placeholder">
      <p>{label}</p>
      <span>Not available in this report.</span>
    </div>
  );

  return (
    <div className="page-scroll">
      <div className="page-grid cross-benchmark-page">
        <section className="panel cross-benchmark-control-panel">
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
              className="text-input"
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
                  </article>
                ))}
              </div>

              <div className="cross-benchmark-chart-grid">
                <article className="cross-benchmark-chart-card">
                  <div className="cross-benchmark-chart-header">
                    <p className="panel-label">Speed Comparison</p>
                  </div>
                  {!isMetricEnabled([
                    'speed.tokens_per_second',
                    'speed.chars_per_second',
                    'speed.processing_time_seconds',
                  ]) || speedChartData.length === 0 ? (
                    renderUnavailable('Speed metrics unavailable')
                  ) : (
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={speedChartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3440" />
                        <XAxis dataKey="tokenizer" stroke="#9ea7b3" hide />
                        <YAxis stroke="#9ea7b3" width={60} />
                        <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }} />
                        <Bar dataKey="tokens_per_second" fill="#4fc3f7" name="tokens/sec" />
                        <Bar dataKey="chars_per_second" fill="#81c784" name="chars/sec" />
                        <Bar dataKey="processing_time_seconds" fill="#ffb74d" name="processing time (s)" />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </article>

                <article className="cross-benchmark-chart-card">
                  <div className="cross-benchmark-chart-header">
                    <p className="panel-label">Global Rates</p>
                  </div>
                  {!isMetricEnabled([
                    'global.oov_rate',
                    'global.determinism_rate',
                    'global.boundary_preservation_rate',
                    'global.round_trip_fidelity_rate',
                  ]) || qualityChartData.length === 0 ? (
                    renderUnavailable('Global rate metrics unavailable')
                  ) : (
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={qualityChartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3440" />
                        <XAxis dataKey="tokenizer" stroke="#9ea7b3" hide />
                        <YAxis stroke="#9ea7b3" width={60} />
                        <Tooltip
                          formatter={(value: number | string | undefined) => `${toNumber(value).toFixed(2)}%`}
                          contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
                        />
                        <Bar dataKey="oov_rate" fill="#f87171" name="oov_rate (%)" />
                        <Bar dataKey="determinism_rate" fill="#22c55e" name="determinism_rate (%)" />
                        <Bar dataKey="boundary_preservation_rate" fill="#38bdf8" name="boundary_preservation_rate (%)" />
                        <Bar dataKey="round_trip_fidelity_rate" fill="#facc15" name="round_trip_fidelity_rate (%)" />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </article>

                <article className="cross-benchmark-chart-card">
                  <div className="cross-benchmark-chart-header">
                    <p className="panel-label">Vocabulary Comparison</p>
                  </div>
                  {!isMetricEnabled([
                    'vocabulary.vocabulary_size',
                    'vocabulary.subwords_count',
                    'vocabulary.true_words_count',
                    'vocabulary.subwords_percentage',
                  ]) || vocabularyChartData.length === 0 ? (
                    renderUnavailable('Vocabulary metrics unavailable')
                  ) : (
                    <ResponsiveContainer width="100%" height={300}>
                      <ComposedChart data={vocabularyChartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3440" />
                        <XAxis dataKey="tokenizer" stroke="#9ea7b3" hide />
                        <YAxis stroke="#9ea7b3" width={60} />
                        <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }} />
                        <Bar dataKey="vocabulary_size" fill="#4fc3f7" name="vocabulary_size" />
                        <Bar dataKey="subwords_count" fill="#ffb74d" name="subwords_count" />
                        <Bar dataKey="true_words_count" fill="#81c784" name="true_words_count" />
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
                  {!isMetricEnabled(['vocabulary.token_length_distribution']) || distributionSeries.length === 0 ? (
                    renderUnavailable('Token length distribution unavailable')
                  ) : (
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={distributionSeries}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3440" />
                        <XAxis dataKey="label" stroke="#9ea7b3" hide />
                        <YAxis stroke="#9ea7b3" width={60} />
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
                    <p className="panel-label">Bytes per Token Distribution (Violin Equivalent)</p>
                  </div>
                  {!isMetricEnabled(['document.bytes_per_token']) || violinEquivalentData.length === 0 ? (
                    renderUnavailable('Per-document bytes per token unavailable')
                  ) : (
                    <div className="cross-benchmark-boxplot-list">
                      {violinEquivalentData.map((entry) => {
                        const range = Math.max(1e-9, entry.max - entry.min);
                        const minPct = 0;
                        const maxPct = 100;
                        const q1Pct = ((entry.q1 - entry.min) / range) * 100;
                        const q3Pct = ((entry.q3 - entry.min) / range) * 100;
                        const medianPct = ((entry.median - entry.min) / range) * 100;
                        return (
                          <div key={entry.tokenizer} className="cross-benchmark-boxplot-row">
                            <span className="cross-benchmark-boxplot-label">{entry.tokenizer}</span>
                            <div className="cross-benchmark-boxplot-track">
                              <div className="cross-benchmark-boxplot-whisker" style={{ left: `${minPct}%`, right: `${100 - maxPct}%` }} />
                              <div
                                className="cross-benchmark-boxplot-box"
                                style={{ left: `${q1Pct}%`, width: `${Math.max(2, q3Pct - q1Pct)}%` }}
                              />
                              <div className="cross-benchmark-boxplot-median" style={{ left: `${medianPct}%` }} />
                            </div>
                            <span className="cross-benchmark-boxplot-values">
                              {entry.min.toFixed(3)} / {entry.median.toFixed(3)} / {entry.max.toFixed(3)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </article>
              </div>

              <div className="cross-benchmark-drilldown-grid">
                <article className="cross-benchmark-drilldown-card">
                  <div className="cross-benchmark-chart-header">
                    <p className="panel-label">Per Tokenizer Core Metrics</p>
                  </div>
                  {activeReport.global_metrics.length === 0 ? (
                    renderUnavailable('Core metrics unavailable')
                  ) : (
                    <div className="cross-benchmark-table-wrap">
                      <table className="tokenizer-meta-table">
                        <thead>
                          <tr>
                            <th>Tokenizer</th>
                            <th>Speed (tok/s)</th>
                            <th>OOV</th>
                            <th>Determinism</th>
                            <th>Boundary</th>
                            <th>Round Trip</th>
                          </tr>
                        </thead>
                        <tbody>
                          {activeReport.global_metrics.map((metric: GlobalMetrics) => (
                            <tr key={`core-${metric.tokenizer}`}>
                              <td>{metric.tokenizer}</td>
                              <td>{Math.round(toNumber(metric.tokenization_speed_tps)).toLocaleString()}</td>
                              <td>{formatRate(toNumber(metric.oov_rate))}</td>
                              <td>{formatRate(toNumber(metric.determinism_rate))}</td>
                              <td>{formatRate(toNumber(metric.boundary_preservation_rate))}</td>
                              <td>{formatRate(toNumber(metric.round_trip_fidelity_rate))}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </article>

                <article className="cross-benchmark-drilldown-card">
                  <div className="cross-benchmark-chart-header">
                    <p className="panel-label">Internal Metrics Drill Down</p>
                  </div>
                  {!isMetricEnabled(['internal.segmentation_consistency']) ? (
                    renderUnavailable('Internal metrics unavailable')
                  ) : (
                    <div className="cross-benchmark-table-wrap">
                      <table className="tokenizer-meta-table tokenizer-meta-table-compact">
                        <thead>
                          <tr>
                            <th>Tokenizer</th>
                            {internalMetricColumns.map((col) => (
                              <th key={col.key}>{col.label}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {activeReport.global_metrics.map((metric) => (
                            <tr key={`internal-${metric.tokenizer}`}>
                              <td>{metric.tokenizer}</td>
                              {internalMetricColumns.map((col) => (
                                <td key={`${metric.tokenizer}-${col.key}`}>
                                  {toNumber(metric[col.key]).toFixed(4)}
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
