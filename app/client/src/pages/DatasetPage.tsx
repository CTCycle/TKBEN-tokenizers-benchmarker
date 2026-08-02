import { useEffect, useMemo, useRef, useState } from 'react';
import type { MouseEvent } from 'react';
import {
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import DatasetValidationWizard from '../components/DatasetValidationWizard';
import CatalogFilterToolbar from '../components/CatalogFilterToolbar';
import DashboardExportButton from '../components/DashboardExportButton';
import DismissibleBanner from '../components/DismissibleBanner';
import HistogramChartCard from '../components/HistogramChartCard';
import ModalCloseButton from '../components/ModalCloseButton';
import {
  CHART_AXIS_PROPS,
  CHART_COLORS,
  CHART_GRID_PROPS,
  CHART_TOOLTIP_STYLE,
  CHART_TOOLTIP_TEXT_STYLE,
  DATASET_DONUT_COLORS,
} from '../common/chartStyles';
import { useDataset } from '../contexts/DatasetContext';
import { useWordCloudLayout } from '../hooks/useWordCloudLayout';
import {
  buildWordCloudFromWordFrequencies,
  buildZipfCurveFromWordFrequencies,
  hasMetricValue,
  isRecord,
  metricDisplayValue,
  normalizeCount,
  normalizePercent,
  parseWordCloudTerms,
  parseWordFrequencyItems,
  parseZipfCurve,
  toHistogramSeries,
  toNumber,
  tooltipCountFormatter,
  tooltipPercentFormatter,
} from '../features/dataset/datasetDashboardData';
import type { DatasetAnalysisRequest } from '../types/api';

type DatasetPreset = {
  id: string;
  label: string;
  description: string;
  defaultConfig?: string;
};

type DatasetGroup = {
  group: string;
  datasets: DatasetPreset[];
};

type DatasetPageProps = {
  showDashboard?: boolean;
  embedded?: boolean;
};

const PREDEFINED_DATASETS: DatasetGroup[] = [
  {
    group: 'General Corpora',
    datasets: [
      {
        id: 'wikitext',
        label: 'wikitext',
        description: 'Clean Wikipedia articles, multiple sizes, common baseline.',
        defaultConfig: 'wikitext-2-v1',
      },
      {
        id: 'c4',
        label: 'c4',
        description: 'Colossal Clean Crawled Corpus, large filtered web crawl.',
      },
      {
        id: 'oscar',
        label: 'oscar',
        description: 'Multilingual web corpus filtered by language.',
      },
      {
        id: 'cc_news',
        label: 'cc_news',
        description: 'News articles from Common Crawl.',
      },
      {
        id: 'openwebtext',
        label: 'openwebtext',
        description: 'Reddit-linked web pages, GPT-style corpus.',
      },
      {
        id: 'bookcorpus',
        label: 'bookcorpus',
        description: 'Fiction books, long-form narrative text.',
      },
    ],
  },
  {
    group: 'News and Formal Writing',
    datasets: [
      {
        id: 'ag_news',
        label: 'ag_news',
        description: 'Short news classification dataset.',
      },
      {
        id: 'cnn_dailymail',
        label: 'cnn_dailymail',
        description: 'News articles with summaries, long documents.',
      },
      {
        id: 'gigaword',
        label: 'gigaword',
        description: 'Newswire text, headline-style language.',
      },
      {
        id: 'multi_news',
        label: 'multi_news',
        description: 'Multi-document news summarization.',
      },
    ],
  },
  {
    group: 'Question Answering and Reading Comprehension',
    datasets: [
      {
        id: 'squad',
        label: 'squad',
        description: 'Wikipedia-based QA dataset.',
      },
      {
        id: 'natural_questions',
        label: 'natural_questions',
        description: 'Real Google search questions with long answers.',
      },
      {
        id: 'hotpot_qa',
        label: 'hotpot_qa',
        description: 'Multi-hop reasoning over multiple passages.',
      },
    ],
  },
  {
    group: 'Instruction, Dialogue, and Conversational Data',
    datasets: [
      {
        id: 'daily_dialog',
        label: 'daily_dialog',
        description: 'Clean, human-written conversations.',
      },
      {
        id: 'empathetic_dialogues',
        label: 'empathetic_dialogues',
        description: 'Emotion-focused conversations.',
      },
      {
        id: 'openassistant_oasst1',
        label: 'openassistant_oasst1',
        description: 'Instruction-following and assistant responses.',
      },
    ],
  },
  {
    group: 'Reviews and Informal Text',
    datasets: [
      {
        id: 'yelp_review_full',
        label: 'yelp_review_full',
        description: 'User reviews of varying length.',
      },
      {
        id: 'amazon_reviews_multi',
        label: 'amazon_reviews_multi',
        description: 'Multilingual product reviews.',
      },
      {
        id: 'imdb',
        label: 'imdb',
        description: 'Long-form movie reviews.',
      },
    ],
  },
  {
    group: 'Academic and Long-Form Text',
    datasets: [
      {
        id: 'arxiv',
        label: 'arxiv',
        description: 'Scientific papers.',
      },
      {
        id: 'pubmed',
        label: 'pubmed',
        description: 'Biomedical abstracts and articles.',
      },
    ],
  },
  {
    group: 'Multilingual Benchmarks',
    datasets: [
      {
        id: 'flores',
        label: 'flores',
        description: 'High-quality multilingual parallel text.',
      },
      {
        id: 'wiki40b',
        label: 'wiki40b',
        description: 'Large multilingual Wikipedia corpus.',
      },
      {
        id: 'opus_books',
        label: 'opus_books',
        description: 'Parallel book translations.',
      },
    ],
  },
];

const DONUT_COLORS = DATASET_DONUT_COLORS;

const DatasetPage = ({ showDashboard = true, embedded = false }: DatasetPageProps) => {
  const {
    datasetName,
    selectedCorpus,
    selectedConfig,
    loading,
    error,
    loadProgress,
    validating,
    validationReport,
    validationProgress,
    fileInputRef,
    availableDatasets,
    datasetsLoading,
    activeValidationDataset,
    activeReportLoadDataset,
    removingDataset,
    metricsCatalog,
    metricsCatalogLoading,
    loadMetricsCatalog,
    setError,
    handleCorpusChange,
    handleConfigChange,
    handleLoadDataset,
    handleUploadClick,
    handleFileChange,
    handleSelectDataset,
    handleValidateDataset,
    handleLoadLatestDatasetReport,
    handleDeleteDataset,
    refreshAvailableDatasets,
  } = useDataset();

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [collapsedPresetGroups, setCollapsedPresetGroups] = useState<Record<string, boolean>>({});
  const [datasetSearch, setDatasetSearch] = useState('');
  const [datasetSourceFilter, setDatasetSourceFilter] = useState<'all' | 'public' | 'custom'>('all');
  const [documentCountValue, setDocumentCountValue] = useState('');
  const [documentCountOperator, setDocumentCountOperator] = useState<'at_least' | 'at_most'>('at_least');
  const [isInsertByNameOpen, setIsInsertByNameOpen] = useState(false);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [wizardDatasetName, setWizardDatasetName] = useState<string | null>(null);
  const manualDatasetInputRef = useRef<HTMLInputElement | null>(null);

  const corpusInputId = 'corpus-input';
  const configInputId = 'config-input';
  const manualInsertRegionId = 'dataset-manual-insert-panel';
  useEffect(() => {
    const numericValue = Number(documentCountValue);
    const filters = {
      search: datasetSearch,
      source: datasetSourceFilter,
      document_count_operator: documentCountOperator,
      ...(documentCountValue.trim() !== '' && Number.isFinite(numericValue) && numericValue >= 0
        ? { document_count: numericValue } : {}),
    };
    const timeoutId = window.setTimeout(() => { void refreshAvailableDatasets(filters); }, 250);
    return () => window.clearTimeout(timeoutId);
  }, [datasetSearch, datasetSourceFilter, documentCountOperator, documentCountValue, refreshAvailableDatasets]);
  const aggregate = useMemo<Record<string, unknown>>(() => {
    const aggregateStats = validationReport?.aggregate_statistics;
    return isRecord(aggregateStats) ? aggregateStats : {};
  }, [validationReport]);
  const hasPersistedReport = validationReport !== null;
  const documentHistogram = hasPersistedReport ? validationReport.document_length_histogram : null;
  const wordHistogram = hasPersistedReport ? validationReport.word_length_histogram : null;
  const documentHistogramSeries = toHistogramSeries(documentHistogram);
  const wordHistogramSeries = toHistogramSeries(wordHistogram);
  const documentCount = hasMetricValue(aggregate['corpus.document_count'])
    ? toNumber(aggregate['corpus.document_count'])
    : hasPersistedReport
      ? validationReport.document_count
      : 0;
  const hasDocumentCount = hasMetricValue(aggregate['corpus.document_count']) || hasPersistedReport;
  const emptyRateRaw = aggregate['quality.empty_rate'];
  const emptyCount = hasMetricValue(emptyRateRaw) && hasDocumentCount
    ? Math.round(toNumber(emptyRateRaw) * documentCount)
    : null;
  const mostCommonWords = useMemo(() => {
    if (!hasPersistedReport) {
      return [];
    }
    if (validationReport.most_common_words?.length) {
      return parseWordFrequencyItems(validationReport.most_common_words);
    }
    return parseWordFrequencyItems(aggregate['words.most_common']);
  }, [aggregate, hasPersistedReport, validationReport]);

  const zipfCurve = useMemo(() => {
    const parsed = parseZipfCurve(
      aggregate['lexical.zipf_curve'],
    );
    if (parsed.length > 0) {
      return parsed;
    }
    return buildZipfCurveFromWordFrequencies(mostCommonWords);
  }, [aggregate, mostCommonWords]);
  const entropyGauge = toNumber(aggregate['words.normalized_entropy']);
  const shannonEntropy = toNumber(aggregate['words.shannon_entropy']);
  const hasEntropyGauge = hasMetricValue(aggregate['words.normalized_entropy']);
  const hasShannonEntropy = hasMetricValue(aggregate['words.shannon_entropy']);
  const duplicateRateRaw = aggregate['quality.duplicate_document_rate'];
  const nearDuplicateRateRaw = aggregate['quality.near_duplicate_document_rate'];
  const topKConcentrationRaw = aggregate['lexical.topk_concentration'];
  const rareTailMassRaw = aggregate['lexical.tail_mass'];
  const duplicateRate = toNumber(duplicateRateRaw);
  const nearDuplicateRate = toNumber(nearDuplicateRateRaw);
  const topKConcentration = toNumber(topKConcentrationRaw);
  const rareTailMass = toNumber(rareTailMassRaw);

  const aggregateRows = [
    { label: 'Num documents', value: hasDocumentCount ? normalizeCount(documentCount) : '—' },
    { label: 'Mean length', value: metricDisplayValue(aggregate['doc.length_mean'], (numeric) => numeric.toFixed(2)) },
    { label: 'Min length', value: metricDisplayValue(aggregate['doc.length_min'], normalizeCount) },
    { label: 'Max length', value: metricDisplayValue(aggregate['doc.length_max'], normalizeCount) },
    { label: 'Empty count', value: emptyCount !== null ? normalizeCount(emptyCount) : '—' },
    { label: 'Length CV', value: metricDisplayValue(aggregate['doc.length_cv'], (numeric) => numeric.toFixed(4)) },
    { label: 'p50', value: metricDisplayValue(aggregate['doc.length_p50'], normalizeCount) },
    { label: 'p90', value: metricDisplayValue(aggregate['doc.length_p90'], normalizeCount) },
    { label: 'p99', value: metricDisplayValue(aggregate['doc.length_p99'], normalizeCount) },
  ];

  const wordMetricRows = [
    { label: 'Vocabulary size', value: metricDisplayValue(aggregate['corpus.unique_words'], normalizeCount) },
    { label: 'MATTR', value: metricDisplayValue(aggregate['corpus.mattr'], (numeric) => numeric.toFixed(4)) },
    { label: 'Entropy', value: metricDisplayValue(aggregate['words.shannon_entropy'], (numeric) => numeric.toFixed(4)) },
    { label: 'Hapax ratio', value: metricDisplayValue(aggregate['words.hapax_ratio'], (numeric) => numeric.toFixed(4)) },
    { label: 'Zipf slope', value: metricDisplayValue(aggregate['words.zipf_slope'], (numeric) => numeric.toFixed(4)) },
    { label: 'Gini', value: metricDisplayValue(aggregate['words.frequency_gini'], (numeric) => numeric.toFixed(4)) },
    { label: 'HHI', value: metricDisplayValue(aggregate['words.hhi'], (numeric) => numeric.toFixed(6)) },
  ];

  const characterSlices = useMemo(() => {
    const rows = [
      { key: 'Whitespace', value: toNumber(aggregate['chars.whitespace_ratio']) },
      { key: 'Punctuation', value: toNumber(aggregate['chars.punctuation_ratio']) },
      { key: 'Digits', value: toNumber(aggregate['chars.digit_ratio']) },
      { key: 'Uppercase', value: toNumber(aggregate['chars.uppercase_ratio']) },
      { key: 'Non-ASCII', value: toNumber(aggregate['chars.non_ascii_ratio']) },
      { key: 'Control', value: toNumber(aggregate['chars.control_ratio']) },
      { key: 'Symbols', value: toNumber(aggregate['chars.symbol_ratio']) },
    ];
    return rows.filter((item) => item.value > 0);
  }, [aggregate]);

  const wordCloudTerms = useMemo(() => {
    if (!hasPersistedReport) {
      return [];
    }
    const parsed = validationReport.word_cloud_terms?.length
      ? parseWordCloudTerms(validationReport.word_cloud_terms)
      : parseWordCloudTerms(
        aggregate['words.word_cloud'],
      );
    if (parsed.length > 0) {
      return parsed;
    }
    return buildWordCloudFromWordFrequencies(mostCommonWords);
  }, [aggregate, hasPersistedReport, mostCommonWords, validationReport]);
  const { wordCloudLayout, wordCloudRef } = useWordCloudLayout(wordCloudTerms);
  const datasetExportReportName = validationReport?.dataset_name
    ? `dataset-${validationReport.dataset_name}-report-${validationReport.report_id ?? 'latest'}`
    : 'dataset-dashboard-report';

  const handlePresetSelect = (preset: DatasetPreset) => {
    setSelectedPreset(preset.id);
    handleCorpusChange(preset.id);
    handleConfigChange(preset.defaultConfig ?? '');
  };

  const handlePresetDownload = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    void handleLoadDataset();
  };

  const openValidationWizard = (targetDataset: string) => {
    handleSelectDataset(targetDataset);
    setWizardDatasetName(targetDataset);
    setWizardOpen(true);
  };

  const selectDatasetAndLoadReport = (targetDataset: string) => {
    handleSelectDataset(targetDataset);
    void handleLoadLatestDatasetReport(targetDataset, { suppressNotFoundError: true });
  };

  const runValidationFromWizard = async (requestOverrides: Partial<DatasetAnalysisRequest>) => {
    const targetDataset = wizardDatasetName ?? datasetName;
    if (!targetDataset) {
      return;
    }
    await handleValidateDataset(targetDataset, requestOverrides);
  };

  const renderValidationStatus = () => {
    if (validating) {
      const progressLabel = validationProgress !== null ? ` (${Math.round(validationProgress)}%)` : '';
      return (
        <div className="loading-container">
          <div className="spinner" />
          <p>Running validation pipeline{progressLabel}...</p>
          <span>Streaming documents and persisting metrics.</span>
        </div>
      );
    }

    if (hasPersistedReport) {
      return null;
    }

    return null;
  };

  const modalDownloadProgress = loadProgress !== null
    ? ` (${Math.round(loadProgress)}%)`
    : '';
  const presetsDisabled = loading;

  useEffect(() => {
    if (isModalOpen && isInsertByNameOpen) {
      manualDatasetInputRef.current?.focus();
    }
  }, [isInsertByNameOpen, isModalOpen]);

  const pageContent = (
    <>
      <div className="page-grid dataset-page-layout">
        <section className="dataset-top-section">
          <div className="dataset-top-row">
            <div className="dataset-intro-panel">
              <div className="dataset-usage-copy">
                <p className="panel-label">Dataset Usage</p>
                <p className="panel-description">
                  Download or upload datasets, then run the validation pipeline to persist advanced
                  quality and lexical metrics for dashboard analysis.
                </p>
              </div>
              <CatalogFilterToolbar
                accessibleName="Dataset filters"
                searchLabel="Search datasets"
                searchValue={datasetSearch}
                searchPlaceholder="Name or namespace"
                onSearchChange={setDatasetSearch}
                sourceLabel="Source"
                sourceValue={datasetSourceFilter}
                sourceOptions={[{ value: 'all', label: 'All datasets' }, { value: 'public', label: 'Public' }, { value: 'custom', label: 'Custom' }]}
                onSourceChange={setDatasetSourceFilter}
                numericLabel="Documents"
                numericValue={documentCountValue}
                numericOperator={documentCountOperator}
                numericPlaceholder="Any count"
                onNumericValueChange={setDocumentCountValue}
                onNumericOperatorChange={setDocumentCountOperator}
                addButtonLabel="Add dataset"
                addButtonTitle="Add or import a dataset"
                onAdd={() => setIsModalOpen(true)}
              />
            </div>
            <div className="dataset-preview-panel">
              <header className="panel-header">
                <div>
                  <p className="panel-label">Dataset Preview</p>
                  <p className="panel-description">
                    Select a dataset and run validation sessions with custom metric selections.
                  </p>
                </div>
              </header>
              <div className="dataset-preview-body">
                {datasetsLoading ? (
                  <div className="dataset-preview-empty">Loading datasets...</div>
                ) : availableDatasets.length === 0 ? (
                  <>
                    <div className="dataset-preview-table dataset-preview-table--empty" role="table" aria-label="Available datasets">
                      <div className="dataset-preview-row dataset-preview-row--header" role="row">
                        <span role="columnheader">Dataset</span>
                        <span role="columnheader">Documents</span>
                        <span role="columnheader">Actions</span>
                      </div>
                    </div>
                    <p className="dataset-preview-empty-label">
                      {datasetSearch.trim() || documentCountValue.trim() || datasetSourceFilter !== 'all'
                        ? 'No datasets match the current filters.' : 'No datasets available.'}
                    </p>
                  </>
                ) : (
                  <div className="dataset-preview-table">
                    <div className="dataset-preview-row dataset-preview-row--header" role="row">
                      <span role="columnheader">Dataset</span>
                      <span role="columnheader">Documents</span>
                      <span role="columnheader">Actions</span>
                    </div>
                    {availableDatasets.map((dataset) => {
                      const isValidating = activeValidationDataset === dataset.dataset_name;
                      const isLoadingReport = activeReportLoadDataset === dataset.dataset_name;
                      const isRemoving = removingDataset === dataset.dataset_name;
                      const isSelectedDataset = datasetName === dataset.dataset_name;
                      return (
                        <div
                          key={dataset.dataset_name}
                          className={`dataset-preview-row${isSelectedDataset ? ' selected' : ''}`}
                        >
                          <button
                            type="button"
                            className="dataset-preview-select"
                            aria-pressed={isSelectedDataset}
                            onClick={() => {
                              if (!isValidating && !isLoadingReport && !isRemoving) {
                                selectDatasetAndLoadReport(dataset.dataset_name);
                              }
                            }}
                            disabled={isValidating || isLoadingReport || isRemoving}
                          >
                            <span className="dataset-preview-name">{dataset.dataset_name}</span>
                          </button>
                          <span className="dataset-preview-count">
                            {normalizeCount(dataset.document_count)}
                          </span>
                          <div className="dataset-preview-actions dataset-preview-actions-wide">
                            <button
                              type="button"
                              className="icon-button subtle dataset-run-pipeline-button"
                              aria-label={`Run validation pipeline for ${dataset.dataset_name}`}
                              title="Run validation pipeline"
                              onClick={(event) => {
                                event.stopPropagation();
                                openValidationWizard(dataset.dataset_name);
                              }}
                              disabled={isValidating || isLoadingReport || isRemoving}
                            >
                              <svg viewBox="0 0 24 24" aria-hidden="true">
                                <path d="M8 5.5v13l10-6.5-10-6.5Z" fill="currentColor" />
                              </svg>
                            </button>
                            <button
                              type="button"
                              className="icon-button subtle"
                              aria-label="Load latest saved report"
                              title="Load latest saved report"
                              onClick={(event) => {
                                event.stopPropagation();
                                handleSelectDataset(dataset.dataset_name);
                                void handleLoadLatestDatasetReport(dataset.dataset_name);
                              }}
                              disabled={isValidating || isLoadingReport || isRemoving}
                            >
                              {isLoadingReport ? (
                                <span className="action-spinner" />
                              ) : (
                                <svg viewBox="0 0 24 24" aria-hidden="true">
                                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Z M14 2v6h6 M8 13h8 M8 17h6" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                              )}
                            </button>
                            <button
                              type="button"
                              className="icon-button danger"
                              aria-label="Remove dataset"
                              title="Delete dataset from database"
                              onClick={(event) => {
                                event.stopPropagation();
                                void handleDeleteDataset(dataset.dataset_name);
                              }}
                              disabled={isValidating || isLoadingReport || isRemoving}
                            >
                              {isRemoving ? (
                                <span className="action-spinner" />
                              ) : (
                                <svg viewBox="0 0 24 24" aria-hidden="true">
                                  <path d="M5 7h14" strokeWidth="2" strokeLinecap="round" />
                                  <path d="M9 7V5h6v2" strokeWidth="2" strokeLinecap="round" />
                                  <rect x="7" y="7" width="10" height="12" rx="2" />
                                </svg>
                              )}
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
          {error && (
            <DismissibleBanner
              message={error}
              onDismiss={() => setError(null)}
              dismissLabel="Dismiss dataset error"
            />
          )}
        </section>

        {showDashboard && (
          <aside className="panel dashboard-panel dashboard-plain dataset-dashboard">
            <header className="panel-header">
              <div>
                <p className="panel-label">Dataset Dashboard</p>
                <p className="panel-description">
                  {validationReport?.dataset_name
                    ? `Latest persisted session for ${validationReport.dataset_name}${validationReport.created_at ? ` (${new Date(validationReport.created_at).toLocaleString()})` : ''}`
                    : 'Load a saved report or run validation to populate this dashboard.'}
                </p>
              </div>
              <div className="dashboard-export-header-actions">
                <DashboardExportButton
                  dashboardType="dataset"
                  reportName={datasetExportReportName}
                  dashboardPayload={validationReport ? { report: validationReport } : null}
                />
              </div>
            </header>
            {renderValidationStatus()}

            <div className="dataset-row dataset-row-one">
              <div className="dataset-card">
                <div className="dataset-card-header">
                  <p className="panel-label">Aggregate Stats</p>
                </div>
                <table className="dataset-table">
                  <tbody>
                    {aggregateRows.map((row) => (
                      <tr key={row.label}>
                        <th>{row.label}</th>
                        <td>{row.value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="dataset-card">
                <div className="dataset-card-header">
                  <p className="panel-label">Word Metrics</p>
                </div>
                <table className="dataset-table">
                  <tbody>
                    {wordMetricRows.map((row) => (
                      <tr key={row.label}>
                        <th>{row.label}</th>
                        <td>{row.value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="dataset-card dataset-chart-card">
                <div className="dataset-card-header">
                  <p className="panel-label">Character Composition</p>
                </div>
                {characterSlices.length === 0 ? (
                  <p className="dataset-empty-label">No character ratio metrics available.</p>
                ) : (
                  <div className="dataset-chart-body">
                    <ResponsiveContainer width="100%" height={280}>
                      <PieChart>
                        <Pie
                          data={characterSlices}
                          dataKey="value"
                          nameKey="key"
                          cx="50%"
                          cy="50%"
                          innerRadius={58}
                          outerRadius={96}
                          paddingAngle={2}
                        >
                          {characterSlices.map((entry, index) => (
                            <Cell key={entry.key} fill={DONUT_COLORS[index % DONUT_COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip
                          formatter={tooltipPercentFormatter}
                          contentStyle={{ ...CHART_TOOLTIP_STYLE, ...CHART_TOOLTIP_TEXT_STYLE }}
                          itemStyle={CHART_TOOLTIP_TEXT_STYLE}
                          labelStyle={CHART_TOOLTIP_TEXT_STYLE}
                          wrapperStyle={CHART_TOOLTIP_TEXT_STYLE}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="dataset-legend">
                      {characterSlices.map((entry, index) => (
                        <span key={entry.key}>
                          <i style={{ backgroundColor: DONUT_COLORS[index % DONUT_COLORS.length] }} />
                          {entry.key}: {normalizePercent(entry.value)}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="dataset-row dataset-row-two">
              <HistogramChartCard
                title="Document Length Histogram"
                data={documentHistogramSeries}
                emptyMessage="No persisted document-length histogram found."
                barFill={CHART_COLORS.yellow}
                tooltipFormatter={tooltipCountFormatter}
              />

              <HistogramChartCard
                title="Word Length Histogram"
                data={wordHistogramSeries}
                emptyMessage="No persisted word-length histogram found."
                barFill={CHART_COLORS.cyan}
                tooltipFormatter={tooltipCountFormatter}
              />
            </div>

            <div className="dataset-row dataset-row-three">
              <div className="dataset-card dataset-extras">
                <div className="dataset-card-header">
                  <p className="panel-label">Additional Visuals</p>
                </div>
                <div className="dataset-extras-grid">
                  <div className="dataset-extras-item">
                    <p className="panel-description">Zipf Curve</p>
                    {zipfCurve.length === 0 ? (
                      <div className="chart-placeholder"><p>No Zipf curve data.</p></div>
                    ) : (
                      <ResponsiveContainer width="100%" height={220}>
                        <LineChart data={zipfCurve}>
                          <CartesianGrid {...CHART_GRID_PROPS} />
                          <XAxis dataKey="rank" {...CHART_AXIS_PROPS} />
                          <YAxis {...CHART_AXIS_PROPS} />
                          <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                          <Line type="monotone" dataKey="frequency" stroke={CHART_COLORS.cyan} dot={false} strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                  </div>

                  <div className="dataset-extras-item">
                    <p className="panel-description">Entropy Gauge</p>
                    <div className="dataset-gauge-track">
                      {hasEntropyGauge && (
                        <div
                          className="dataset-gauge-fill"
                          style={{ width: `${Math.max(0, Math.min(100, entropyGauge * 100))}%` }}
                        />
                      )}
                    </div>
                    <p className="dataset-gauge-value">{hasEntropyGauge ? normalizePercent(entropyGauge) : '—'}</p>
                    {hasShannonEntropy ? (
                      <div className="dataset-indicator-row">
                        <span>Shannon entropy</span>
                        <strong>{shannonEntropy.toFixed(4)}</strong>
                      </div>
                    ) : (
                      <p className="panel-description dataset-entropy-help">
                        Entropy summarizes how evenly words are distributed across the corpus.
                      </p>
                    )}
                    {hasEntropyGauge && (
                      <div className="dataset-indicator-row">
                        <span>Interpretation</span>
                        <strong>
                          {entropyGauge >= 0.75
                            ? 'High lexical variety'
                            : entropyGauge >= 0.45
                              ? 'Moderate lexical variety'
                              : 'Concentrated vocabulary'}
                        </strong>
                      </div>
                    )}
                  </div>

                  <div className="dataset-extras-item">
                    <p className="panel-description">Duplicate Indicators</p>
                    <div className="dataset-indicator-row">
                      <span>Exact duplicate rate</span>
                      <strong>{hasMetricValue(duplicateRateRaw) ? normalizePercent(duplicateRate) : '—'}</strong>
                    </div>
                    <div className="dataset-indicator-row">
                      <span>Near-duplicate rate</span>
                      <strong>{hasMetricValue(nearDuplicateRateRaw) ? normalizePercent(nearDuplicateRate) : '—'}</strong>
                    </div>
                  </div>

                  <div className="dataset-extras-item">
                    <p className="panel-description">Concentration</p>
                    <div className="dataset-indicator-row">
                      <span>Top-k concentration</span>
                      <strong>{hasMetricValue(topKConcentrationRaw) ? normalizePercent(topKConcentration) : '—'}</strong>
                    </div>
                    <div className="dataset-indicator-row">
                      <span>Rare tail mass</span>
                      <strong>{hasMetricValue(rareTailMassRaw) ? normalizePercent(rareTailMass) : '—'}</strong>
                    </div>
                  </div>
                </div>
              </div>

              <div className="dataset-card dataset-word-cloud-card">
                <div className="dataset-card-header">
                  <p className="panel-label">Word Cloud</p>
                </div>
                {!wordCloudTerms.length ? (
                  <p className="dataset-empty-label dataset-word-cloud-empty">
                    No word cloud terms in persisted report.
                  </p>
                ) : (
                  <div className="dataset-word-cloud-canvas" ref={wordCloudRef}>
                    {wordCloudLayout.length > 0 ? wordCloudLayout.map((term) => (
                      <span
                        key={`${term.word}-${term.count}`}
                        className="dataset-word-cloud-term"
                        style={{
                          left: `${term.x}px`,
                          top: `${term.y}px`,
                          fontSize: `${term.fontSize}px`,
                          transform: `translate(-50%, -50%) rotate(${term.rotate}deg)`,
                        }}
                        title={`${term.word}: ${normalizeCount(term.count)}`}
                      >
                        {term.word}
                      </span>
                    )) : (
                      <div className="dataset-word-cloud-fallback" role="status">
                        {wordCloudTerms.slice(0, 48).map((term) => (
                          <span
                            key={`${term.word}-${term.count}`}
                            className="dataset-word-cloud-fallback-term"
                            style={{ fontSize: `${Math.max(12, Math.min(32, 12 + Math.round(term.weight * 0.16)))}px` }}
                            title={`${term.word}: ${normalizeCount(term.count)}`}
                          >
                            {term.word}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </aside>
        )}
      </div>

      <DatasetValidationWizard
        isOpen={wizardOpen}
        datasetName={wizardDatasetName ?? datasetName}
        categories={metricsCatalog}
        loadingCategories={metricsCatalogLoading}
        validating={validating}
        onRetryCatalogLoad={() => { void loadMetricsCatalog(); }}
        onClose={() => setWizardOpen(false)}
        onRun={runValidationFromWizard}
      />

      {isModalOpen && (
        <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="dataset-selector-title">
          <div className="modal-card dataset-modal">
            <header className="dataset-modal-header">
              <p id="dataset-selector-title" className="panel-label">Predefined Datasets</p>
              <div className="dataset-modal-actions">
                <button
                  type="button"
                  className={`icon-button subtle${isInsertByNameOpen ? ' accent' : ''}`}
                  aria-label="Insert dataset by name"
                  aria-expanded={isInsertByNameOpen}
                  aria-controls={manualInsertRegionId}
                  title={isInsertByNameOpen ? 'Hide manual dataset input' : 'Show manual dataset input'}
                  onClick={() => setIsInsertByNameOpen((value) => !value)}
                  disabled={loading}
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M4 7h16M4 12h8M4 17h8" strokeWidth="2" strokeLinecap="round" />
                    <path d="M16 12h4M18 10v4" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                </button>
                <button
                  type="button"
                  className="icon-button subtle"
                  aria-label="Upload custom dataset"
                  title="Upload a CSV or Excel dataset file"
                  onClick={handleUploadClick}
                  disabled={loading}
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 15V5" strokeWidth="2" strokeLinecap="round" />
                    <path d="M8 9l4-4 4 4" strokeWidth="2" strokeLinecap="round" />
                    <path d="M4 19h16" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                </button>
                <ModalCloseButton
                  ariaLabel="Close dataset selector"
                  title="Close dataset selector"
                  onClick={() => {
                    setIsInsertByNameOpen(false);
                    setIsModalOpen(false);
                  }}
                />
              </div>
            </header>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".csv,.xlsx,.xls"
              className="hidden-file-input"
              aria-label="Upload custom dataset file"
            />
            {loading && (
              <div className="dataset-modal-progress" role="status" aria-live="polite">
                Downloading dataset{modalDownloadProgress}...
              </div>
            )}
            <div className="dataset-modal-content">
              {isInsertByNameOpen && (
                <div
                  id={manualInsertRegionId}
                  className="dataset-manual-panel"
                  role="region"
                  aria-label="Manual dataset input"
                >
                  <p className="dataset-manual-help">
                    Enter a Hugging Face dataset ID and optional configuration.
                  </p>
                  <div className="dataset-insert-row">
                    <input
                      ref={manualDatasetInputRef}
                      id={corpusInputId}
                      className="text-input"
                      value={selectedCorpus}
                      onChange={(event) => handleCorpusChange(event.target.value)}
                      disabled={loading}
                      aria-label="Dataset name"
                      placeholder="Dataset name"
                    />
                    <input
                      id={configInputId}
                      className="text-input"
                      value={selectedConfig}
                      onChange={(event) => handleConfigChange(event.target.value)}
                      disabled={loading}
                      aria-label="Configuration"
                      placeholder="Configuration"
                    />
                    <button
                      type="button"
                      className="icon-button accent"
                      onClick={() => void handleLoadDataset()}
                      disabled={loading || !selectedCorpus.trim()}
                      aria-label="Download dataset"
                      title="Download dataset from Hugging Face"
                    >
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M12 3v12" strokeWidth="2" strokeLinecap="round" />
                        <path d="M7 10l5 5 5-5" strokeWidth="2" strokeLinecap="round" />
                        <path d="M5 19h14" strokeWidth="2" strokeLinecap="round" />
                      </svg>
                    </button>
                  </div>
                </div>
              )}
              <div
                className="dataset-preset-list-shell"
                aria-disabled={presetsDisabled}
              >
                <div className="dataset-preset-list">
                  {PREDEFINED_DATASETS.map((group, groupIndex) => {
                    const groupId = `dataset-preset-group-${groupIndex}`;
                    const isCollapsed = collapsedPresetGroups[group.group] ?? false;
                    return (
                      <div className="dataset-preset-group" key={group.group}>
                        <button
                          type="button"
                          className="dataset-preset-heading"
                          aria-expanded={!isCollapsed}
                          aria-controls={groupId}
                          onClick={() => setCollapsedPresetGroups((current) => ({
                            ...current,
                            [group.group]: !isCollapsed,
                          }))}
                        >
                          <span>{group.group}</span>
                          <span className="dataset-preset-heading-icon" aria-hidden="true">{isCollapsed ? '+' : '−'}</span>
                        </button>
                        <div id={groupId} hidden={isCollapsed}>
                          {group.datasets.map((preset) => {
                            const isSelected = selectedPreset === preset.id;
                            return (
                              <div
                                key={preset.id}
                                role="button"
                                tabIndex={presetsDisabled ? -1 : 0}
                                aria-disabled={presetsDisabled}
                                className={`dataset-preset-row${isSelected ? ' selected' : ''}`}
                                onClick={() => {
                                  if (!presetsDisabled) {
                                    handlePresetSelect(preset);
                                  }
                                }}
                                onKeyDown={(event) => {
                                  if (!presetsDisabled && (event.key === 'Enter' || event.key === ' ')) {
                                    handlePresetSelect(preset);
                                  }
                                }}
                              >
                                <div className="dataset-preset-info">
                                  <span className="dataset-preset-name">{preset.label}</span>
                                  <span className="dataset-preset-description">{preset.description}</span>
                                </div>
                                {isSelected && (
                                  <button
                                    type="button"
                                    className="icon-button subtle"
                                    aria-label={`Download ${preset.label}`}
                                    title={`Download ${preset.label}`}
                                    onClick={handlePresetDownload}
                                    disabled={loading || presetsDisabled}
                                  >
                                    <svg viewBox="0 0 24 24" aria-hidden="true">
                                      <path d="M12 3v12" strokeWidth="2" strokeLinecap="round" />
                                      <path d="M7 10l5 5 5-5" strokeWidth="2" strokeLinecap="round" />
                                      <path d="M5 19h14" strokeWidth="2" strokeLinecap="round" />
                                    </svg>
                                  </button>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );

  if (embedded) {
    return pageContent;
  }

  return <div className="page-scroll">{pageContent}</div>;
};

export default DatasetPage;
