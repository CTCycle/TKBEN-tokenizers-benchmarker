import { useEffect, useMemo, useRef, useState } from 'react';
import type { MouseEvent } from 'react';
import {
  Bar,
  BarChart,
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
import { useDataset } from '../contexts/DatasetContext';
import type {
  DatasetAnalysisRequest,
  DatasetMetricCatalogCategory,
  HistogramData,
  WordCloudTerm,
  WordFrequency,
} from '../types/api';

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

type WordCloudLayoutTerm = {
  word: string;
  count: number;
  weight: number;
  x: number;
  y: number;
  rotate: number;
  fontSize: number;
};

type WordCloudWorkerOutput = {
  terms: WordCloudLayoutTerm[];
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

const DONUT_COLORS = ['#f59e0b', '#fb7185', '#38bdf8', '#34d399', '#a78bfa', '#f97316', '#64748b'];

const toNumber = (value: unknown, fallback = 0): number => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
};

const normalizePercent = (value: number): string => `${(value * 100).toFixed(2)}%`;

const normalizeCount = (value: number): string => Math.round(value).toLocaleString();

const toHistogramSeries = (histogram: HistogramData | null): Array<{ bin: string; count: number }> => {
  if (!histogram) {
    return [];
  }
  return histogram.counts.map((count, index) => ({
    bin: histogram.bins[index] ?? String(index),
    count,
  }));
};

const parseJsonLike = (value: unknown): unknown => {
  if (typeof value !== 'string') {
    return value;
  }
  const candidate = value.trim();
  if (!candidate) {
    return null;
  }
  try {
    return JSON.parse(candidate);
  } catch {
    return value;
  }
};

const parseWordFrequencyItems = (value: unknown): WordFrequency[] => {
  const parsed = parseJsonLike(value);
  if (Array.isArray(parsed)) {
    return parsed
      .map((item) => {
        if (!item || typeof item !== 'object') {
          return null;
        }
        const payload = item as Record<string, unknown>;
        const word = typeof payload.word === 'string'
          ? payload.word
          : typeof payload.token === 'string'
            ? payload.token
            : typeof payload.text === 'string'
              ? payload.text
              : '';
        if (!word) {
          return null;
        }
        const count = Math.max(
          0,
          Math.round(
            toNumber(payload.count ?? payload.frequency ?? payload.value, 0),
          ),
        );
        return { word, count };
      })
      .filter((item): item is WordFrequency => item !== null && item.count > 0);
  }
  if (parsed && typeof parsed === 'object') {
    const payload = parsed as Record<string, unknown>;
    if (
      typeof payload.word === 'string'
      || typeof payload.token === 'string'
      || typeof payload.text === 'string'
    ) {
      return parseWordFrequencyItems([payload]);
    }
    return Object.entries(parsed as Record<string, unknown>)
      .map(([word, countValue]) => ({
        word,
        count: Math.max(0, Math.round(toNumber(countValue, 0))),
      }))
      .filter((item) => item.word && item.count > 0);
  }
  return [];
};

const parseWordCloudTerms = (value: unknown): WordCloudTerm[] => {
  const parsed = parseJsonLike(value);
  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
    const payload = parsed as Record<string, unknown>;
    if (
      typeof payload.word === 'string'
      || typeof payload.token === 'string'
      || typeof payload.text === 'string'
    ) {
      return parseWordCloudTerms([payload]);
    }
    if (Array.isArray(payload.terms)) {
      return parseWordCloudTerms(payload.terms);
    }
    if (Array.isArray(payload.items)) {
      return parseWordCloudTerms(payload.items);
    }
    return parseWordCloudTerms(
      Object.entries(payload).map(([word, count]) => ({ word, count })),
    );
  }
  if (!Array.isArray(parsed)) {
    return [];
  }

  const terms = parsed
    .map((item) => {
      if (!item || typeof item !== 'object') {
        return null;
      }
      const payload = item as Record<string, unknown>;
      const word = typeof payload.word === 'string'
        ? payload.word
        : typeof payload.token === 'string'
          ? payload.token
          : typeof payload.text === 'string'
            ? payload.text
            : '';
      if (!word) {
        return null;
      }
      return {
        word,
        count: Math.max(0, Math.round(toNumber(payload.count ?? payload.frequency ?? payload.value, 0))),
        weight: toNumber(payload.weight, 0),
      };
    })
    .filter((item): item is WordCloudTerm => item !== null && item.count > 0);

  if (!terms.length) {
    return [];
  }

  const maxCount = Math.max(...terms.map((item) => item.count));
  return terms.map((item) => ({
    ...item,
    weight: item.weight > 0
      ? item.weight
      : Math.max(1, Math.round((item.count / Math.max(1, maxCount)) * 100)),
  }));
};

const parseZipfCurve = (value: unknown): Array<{ rank: number; frequency: number }> => {
  const parsed = parseJsonLike(value);
  if (Array.isArray(parsed)) {
    return parsed
      .map((item, index) => {
        if (Array.isArray(item)) {
          return {
            rank: toNumber(item[0], index + 1),
            frequency: toNumber(item[1], 0),
          };
        }
        if (!item || typeof item !== 'object') {
          return null;
        }
        const payload = item as Record<string, unknown>;
        return {
          rank: toNumber(payload.rank ?? payload.x, index + 1),
          frequency: toNumber(payload.frequency ?? payload.count ?? payload.y ?? payload.value, 0),
        };
      })
      .filter((item): item is { rank: number; frequency: number } => item !== null && item.rank > 0 && item.frequency > 0)
      .sort((a, b) => a.rank - b.rank)
      .slice(0, 200);
  }

  if (parsed && typeof parsed === 'object') {
    const payload = parsed as Record<string, unknown>;
    if (
      typeof payload.rank === 'number'
      || typeof payload.rank === 'string'
      || typeof payload.frequency === 'number'
      || typeof payload.frequency === 'string'
    ) {
      return parseZipfCurve([payload]);
    }
    if (Array.isArray(payload.curve)) {
      return parseZipfCurve(payload.curve);
    }
    if (Array.isArray(payload.ranks) && Array.isArray(payload.frequencies)) {
      const ranks = payload.ranks as unknown[];
      const frequencies = payload.frequencies as unknown[];
      const points = ranks.map((rank, index) => ({
        rank: toNumber(rank, index + 1),
        frequency: toNumber(frequencies[index], 0),
      }));
      return points
        .filter((item) => item.rank > 0 && item.frequency > 0)
        .slice(0, 200);
    }
    const entries = Object.entries(payload)
      .map(([rank, frequency]) => ({
        rank: toNumber(rank, 0),
        frequency: toNumber(frequency, 0),
      }))
      .filter((item) => item.rank > 0 && item.frequency > 0)
      .sort((a, b) => a.rank - b.rank);
    if (entries.length > 0) {
      return entries.slice(0, 200);
    }
  }
  return [];
};

const tooltipPercentFormatter = (value: number | string | undefined): string =>
  normalizePercent(toNumber(value, 0));

const tooltipCountFormatter = (
  value: number | string | undefined,
): [string, 'count'] => [normalizeCount(toNumber(value, 0)), 'count'];

const buildZipfCurveFromWordFrequencies = (
  items: WordFrequency[],
): Array<{ rank: number; frequency: number }> =>
  items
    .filter((item) => item.count > 0)
    .sort((a, b) => b.count - a.count || a.word.localeCompare(b.word))
    .map((item, index) => ({
      rank: index + 1,
      frequency: item.count,
    }))
    .slice(0, 200);

const buildWordCloudFromWordFrequencies = (items: WordFrequency[]): WordCloudTerm[] => {
  const ranked = items
    .filter((item) => item.count > 0)
    .sort((a, b) => b.count - a.count || a.word.localeCompare(b.word))
    .slice(0, 120);
  if (!ranked.length) {
    return [];
  }
  const maxCount = Math.max(...ranked.map((item) => item.count));
  return ranked.map((item) => ({
    word: item.word,
    count: item.count,
    weight: Math.max(1, Math.round((item.count / Math.max(1, maxCount)) * 100)),
  }));
};

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
  } = useDataset();

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [isInsertByNameOpen, setIsInsertByNameOpen] = useState(false);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [wizardDatasetName, setWizardDatasetName] = useState<string | null>(null);
  const [wordCloudLayout, setWordCloudLayout] = useState<WordCloudLayoutTerm[]>([]);
  const [wordCloudSize, setWordCloudSize] = useState({ width: 0, height: 0 });
  const manualDatasetInputRef = useRef<HTMLInputElement | null>(null);
  const wordCloudRef = useRef<HTMLDivElement | null>(null);

  const corpusInputId = 'corpus-input';
  const configInputId = 'config-input';
  const manualInsertRegionId = 'dataset-manual-insert-panel';
  const aggregate = (validationReport?.aggregate_statistics ?? {}) as Record<string, unknown>;
  const hasPersistedReport = validationReport !== null;
  const documentHistogram = hasPersistedReport ? validationReport.document_length_histogram : null;
  const wordHistogram = hasPersistedReport ? validationReport.word_length_histogram : null;
  const documentHistogramSeries = toHistogramSeries(documentHistogram);
  const wordHistogramSeries = toHistogramSeries(wordHistogram);
  const documentCount = hasPersistedReport ? validationReport.document_count : 0;
  const emptyRate = toNumber(aggregate['quality.empty_rate']);
  const emptyCount = Math.round(emptyRate * documentCount);
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
      aggregate['words.zipf_curve']
      ?? aggregate['words.zipf']
      ?? aggregate.zipf_curve
      ?? aggregate.zipfCurve,
    );
    if (parsed.length > 0) {
      return parsed;
    }
    return buildZipfCurveFromWordFrequencies(mostCommonWords);
  }, [aggregate, mostCommonWords]);
  const entropyGauge = toNumber(aggregate['words.normalized_entropy']);
  const duplicateRate = toNumber(aggregate['quality.duplicate_rate']);
  const nearDuplicateRate = toNumber(aggregate['quality.near_duplicate_rate']);
  const topKConcentration = toNumber(aggregate['words.topk_concentration']);
  const rareTailMass = toNumber(aggregate['words.rare_tail_mass']);

  const aggregateRows = [
    { label: 'Num documents', value: normalizeCount(documentCount) },
    { label: 'Mean length', value: toNumber(aggregate['doc.length_mean']).toFixed(2) },
    { label: 'Min length', value: normalizeCount(toNumber(aggregate['doc.length_min'])) },
    { label: 'Max length', value: normalizeCount(toNumber(aggregate['doc.length_max'])) },
    { label: 'Empty count', value: normalizeCount(emptyCount) },
    { label: 'Length CV', value: toNumber(aggregate['doc.length_cv']).toFixed(4) },
    { label: 'p50', value: normalizeCount(toNumber(aggregate['doc.length_p50'])) },
    { label: 'p90', value: normalizeCount(toNumber(aggregate['doc.length_p90'])) },
    { label: 'p99', value: normalizeCount(toNumber(aggregate['doc.length_p99'])) },
  ];

  const wordMetricRows = [
    { label: 'Vocabulary size', value: normalizeCount(toNumber(aggregate['corpus.unique_words'])) },
    { label: 'MATTR', value: toNumber(aggregate['corpus.mattr']).toFixed(4) },
    { label: 'Entropy', value: toNumber(aggregate['words.shannon_entropy']).toFixed(4) },
    { label: 'Hapax ratio', value: toNumber(aggregate['words.hapax_ratio']).toFixed(4) },
    { label: 'Zipf slope', value: toNumber(aggregate['words.zipf_slope']).toFixed(4) },
    { label: 'Gini', value: toNumber(aggregate['words.frequency_gini']).toFixed(4) },
    { label: 'HHI', value: toNumber(aggregate['words.hhi']).toFixed(6) },
  ];

  const characterSlices = useMemo(() => {
    const rows = [
      { key: 'Whitespace', value: toNumber(aggregate['chars.whitespace_ratio']) },
      { key: 'Punctuation', value: toNumber(aggregate['chars.punctuation_ratio']) },
      { key: 'Digits', value: toNumber(aggregate['chars.digit_ratio']) },
      { key: 'Uppercase', value: toNumber(aggregate['chars.uppercase_ratio']) },
      { key: 'Non-ASCII', value: toNumber(aggregate['chars.non_ascii_ratio']) },
      { key: 'Control', value: toNumber(aggregate['chars.control_ratio']) },
      { key: 'Other', value: toNumber(aggregate['chars.other_ratio']) },
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
        aggregate['words.word_cloud']
        ?? aggregate.word_cloud_terms
        ?? aggregate.wordCloudTerms,
      );
    if (parsed.length > 0) {
      return parsed;
    }
    return buildWordCloudFromWordFrequencies(mostCommonWords);
  }, [aggregate, hasPersistedReport, mostCommonWords, validationReport]);

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

    return (
      <div className="chart-placeholder">
        <p>No persisted validation session loaded.</p>
        <span>Run validation from the dataset preview list.</span>
      </div>
    );
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

  useEffect(() => {
    const node = wordCloudRef.current;
    if (!node) {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      const first = entries[0];
      if (!first) {
        return;
      }
      setWordCloudSize({
        width: Math.max(260, Math.round(first.contentRect.width)),
        height: Math.max(240, Math.round(first.contentRect.height)),
      });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!wordCloudTerms.length || wordCloudSize.width <= 0 || wordCloudSize.height <= 0) {
      setWordCloudLayout([]);
      return;
    }

    const worker = new Worker(new URL('../workers/wordCloudWorker.ts', import.meta.url), {
      type: 'module',
    });
    worker.onmessage = (event: MessageEvent<WordCloudWorkerOutput>) => {
      setWordCloudLayout(event.data?.terms ?? []);
      worker.terminate();
    };
    worker.postMessage({
      terms: wordCloudTerms,
      width: wordCloudSize.width,
      height: wordCloudSize.height,
    });
    return () => worker.terminate();
  }, [wordCloudSize.height, wordCloudSize.width, wordCloudTerms]);

  const pageContent = (
    <>
      <div className="page-grid dataset-page-layout">
        <section className="dataset-top-section">
          <div className="dataset-top-row">
            <div className="dataset-intro-panel">
              <p className="panel-label">Dataset Usage</p>
              <p className="panel-description">
                Download or upload datasets, then run the validation pipeline to persist advanced
                quality and lexical metrics for dashboard analysis.
              </p>
            </div>
            <div className="dataset-top-divider" aria-hidden="true" />
            <div className="dataset-preview-panel">
              <header className="panel-header">
                <div>
                  <p className="panel-label">Dataset Preview</p>
                  <p className="panel-description">
                    Select a dataset and run validation sessions with custom metric selections.
                  </p>
                </div>
                <button
                  type="button"
                  className="icon-button"
                  onClick={() => setIsModalOpen(true)}
                  aria-label="Add dataset"
                  title="Add or import a dataset"
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 5v14M5 12h14" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                </button>
              </header>
              <div className="dataset-preview-body">
                {datasetsLoading ? (
                  <div className="dataset-preview-empty">Loading datasets...</div>
                ) : availableDatasets.length === 0 ? (
                  <div className="dataset-preview-empty">
                    No datasets available.
                  </div>
                ) : (
                  <div className="dataset-preview-table">
                    {availableDatasets.map((dataset) => {
                      const isValidating = activeValidationDataset === dataset.dataset_name;
                      const isLoadingReport = activeReportLoadDataset === dataset.dataset_name;
                      const isRemoving = removingDataset === dataset.dataset_name;
                      const isSelectedDataset = datasetName === dataset.dataset_name;
                      return (
                        <div
                          key={dataset.dataset_name}
                          role="button"
                          tabIndex={0}
                          aria-pressed={isSelectedDataset}
                          className={`dataset-preview-row${isSelectedDataset ? ' selected' : ''}`}
                          onClick={() => handleSelectDataset(dataset.dataset_name)}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter' || event.key === ' ') {
                              event.preventDefault();
                              handleSelectDataset(dataset.dataset_name);
                            }
                          }}
                        >
                          <span className="dataset-preview-name">{dataset.dataset_name}</span>
                          <span className="dataset-preview-count">
                            {normalizeCount(dataset.document_count)}
                          </span>
                          <div className="dataset-preview-actions dataset-preview-actions-wide">
                            <button
                              type="button"
                              className="secondary-button dataset-run-pipeline-button"
                              onClick={(event) => {
                                event.stopPropagation();
                                openValidationWizard(dataset.dataset_name);
                              }}
                              disabled={isValidating || isLoadingReport || isRemoving}
                            >
                              Run validation pipeline
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
                                  <path d="M12 4v9m0 0-4-4m4 4 4-4M5 19h14" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
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
            <div className="error-banner">
              <span>{error}</span>
              <button onClick={() => setError(null)}>×</button>
            </div>
          )}
        </section>

        {showDashboard && (
          <aside className="panel dashboard-panel dashboard-plain dataset-v2-dashboard">
            <header className="panel-header">
              <div>
                <p className="panel-label">Dataset Dashboard</p>
                <p className="panel-description">
                  {validationReport?.dataset_name
                    ? `Latest persisted session for ${validationReport.dataset_name}${validationReport.created_at ? ` (${new Date(validationReport.created_at).toLocaleString()})` : ''}`
                    : 'Load a saved report or run validation to populate this dashboard.'}
                </p>
              </div>
            </header>
            {renderValidationStatus()}

            <div className="dataset-v2-row dataset-v2-row-one">
              <div className="dataset-v2-card">
                <div className="dataset-v2-card-header">
                  <p className="panel-label">Aggregate Stats</p>
                </div>
                <table className="dataset-v2-table">
                  <tbody>
                    {aggregateRows.map((row) => (
                      <tr key={row.label}>
                        <th>{row.label}</th>
                        <td>{hasPersistedReport ? row.value : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="dataset-v2-card">
                <div className="dataset-v2-card-header">
                  <p className="panel-label">Word Metrics</p>
                </div>
                <table className="dataset-v2-table">
                  <tbody>
                    {wordMetricRows.map((row) => (
                      <tr key={row.label}>
                        <th>{row.label}</th>
                        <td>{hasPersistedReport ? row.value : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="dataset-v2-card dataset-v2-chart-card">
                <div className="dataset-v2-card-header">
                  <p className="panel-label">Character Composition</p>
                </div>
                {characterSlices.length === 0 ? (
                  <div className="chart-placeholder">
                    <p>No character ratio metrics available.</p>
                  </div>
                ) : (
                  <div className="dataset-v2-chart-body">
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
                          contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', color: '#f8fafc' }}
                          itemStyle={{ color: '#f8fafc' }}
                          labelStyle={{ color: '#f8fafc' }}
                          wrapperStyle={{ color: '#f8fafc' }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="dataset-v2-legend">
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

            <div className="dataset-v2-row dataset-v2-row-two">
              <div className="dataset-v2-card dataset-v2-chart-card">
                <div className="dataset-v2-card-header">
                  <p className="panel-label">Document Length Histogram</p>
                </div>
                {documentHistogramSeries.length === 0 ? (
                  <div className="chart-placeholder">
                    <p>No persisted document-length histogram found.</p>
                  </div>
                ) : (
                  <div className="dataset-v2-chart-body">
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart data={documentHistogramSeries}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3440" />
                        <XAxis dataKey="bin" hide />
                        <YAxis stroke="#9ea7b3" width={48} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
                          formatter={tooltipCountFormatter}
                        />
                        <Bar dataKey="count" fill="#facc15" radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>

              <div className="dataset-v2-card dataset-v2-chart-card">
                <div className="dataset-v2-card-header">
                  <p className="panel-label">Word Length Histogram</p>
                </div>
                {wordHistogramSeries.length === 0 ? (
                  <div className="chart-placeholder">
                    <p>No persisted word-length histogram found.</p>
                  </div>
                ) : (
                  <div className="dataset-v2-chart-body">
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart data={wordHistogramSeries}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3440" />
                        <XAxis dataKey="bin" hide />
                        <YAxis stroke="#9ea7b3" width={48} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
                          formatter={tooltipCountFormatter}
                        />
                        <Bar dataKey="count" fill="#38bdf8" radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            </div>

            <div className="dataset-v2-row dataset-v2-row-three">
              <div className="dataset-v2-card dataset-v2-extras">
                <div className="dataset-v2-card-header">
                  <p className="panel-label">Additional Visuals</p>
                </div>
                <div className="dataset-v2-extras-grid">
                  <div className="dataset-v2-mini-card">
                    <p className="panel-description">Zipf Curve</p>
                    {zipfCurve.length === 0 ? (
                      <div className="chart-placeholder"><p>No Zipf curve data.</p></div>
                    ) : (
                      <ResponsiveContainer width="100%" height={220}>
                        <LineChart data={zipfCurve}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#2d3440" />
                          <XAxis dataKey="rank" stroke="#9ea7b3" />
                          <YAxis stroke="#9ea7b3" />
                          <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }} />
                          <Line type="monotone" dataKey="frequency" stroke="#38bdf8" dot={false} strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                  </div>

                  <div className="dataset-v2-mini-card">
                    <p className="panel-description">Entropy Gauge</p>
                    <div className="dataset-v2-gauge-track">
                      <div
                        className="dataset-v2-gauge-fill"
                        style={{ width: `${Math.max(0, Math.min(100, entropyGauge * 100))}%` }}
                      />
                    </div>
                    <p className="dataset-v2-gauge-value">{normalizePercent(entropyGauge)}</p>
                  </div>

                  <div className="dataset-v2-mini-card">
                    <p className="panel-description">Duplicate Indicators</p>
                    <div className="dataset-v2-indicator-row">
                      <span>Exact duplicate rate</span>
                      <strong>{normalizePercent(duplicateRate)}</strong>
                    </div>
                    <div className="dataset-v2-indicator-row">
                      <span>Near-duplicate rate</span>
                      <strong>{normalizePercent(nearDuplicateRate)}</strong>
                    </div>
                  </div>

                  <div className="dataset-v2-mini-card">
                    <p className="panel-description">Concentration</p>
                    <div className="dataset-v2-indicator-row">
                      <span>Top-k concentration</span>
                      <strong>{normalizePercent(topKConcentration)}</strong>
                    </div>
                    <div className="dataset-v2-indicator-row">
                      <span>Rare tail mass</span>
                      <strong>{normalizePercent(rareTailMass)}</strong>
                    </div>
                  </div>
                </div>
              </div>

              <div className="dataset-v2-card dataset-v2-word-cloud-card">
                <div className="dataset-v2-card-header">
                  <p className="panel-label">Word Cloud</p>
                </div>
                <div className="dataset-v2-word-cloud-canvas" ref={wordCloudRef}>
                  {!wordCloudTerms.length && (
                    <div className="chart-placeholder"><p>No word cloud terms in persisted report.</p></div>
                  )}
                  {wordCloudLayout.map((term) => (
                    <span
                      key={`${term.word}-${term.count}`}
                      className="dataset-v2-word-cloud-term"
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
                  ))}
                </div>
              </div>
            </div>
          </aside>
        )}
      </div>

      <DatasetValidationWizard
        isOpen={wizardOpen}
        datasetName={wizardDatasetName ?? datasetName}
        categories={metricsCatalog as DatasetMetricCatalogCategory[]}
        loadingCategories={metricsCatalogLoading}
        validating={validating}
        onClose={() => setWizardOpen(false)}
        onRun={runValidationFromWizard}
      />

      {isModalOpen && (
        <div className="modal-overlay" role="dialog" aria-modal="true">
          <div className="modal-card dataset-modal">
            <header className="dataset-modal-header">
              <p className="panel-label">Predefined Datasets</p>
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
              </div>
            </header>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".csv,.xlsx,.xls"
              style={{ display: 'none' }}
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
                  {PREDEFINED_DATASETS.map((group) => (
                    <div className="dataset-preset-group" key={group.group}>
                      <p className="dataset-preset-heading">{group.group}</p>
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
                  ))}
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button
                type="button"
                className="secondary-button"
                title="Close dataset selector"
                onClick={() => {
                  setIsInsertByNameOpen(false);
                  setIsModalOpen(false);
                }}
              >
                Close
              </button>
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
