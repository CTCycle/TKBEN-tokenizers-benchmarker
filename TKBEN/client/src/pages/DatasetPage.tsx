import { useState } from 'react';
import type { MouseEvent } from 'react';
import { useDataset } from '../contexts/DatasetContext';

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
        id: 'the_pile',
        label: 'the_pile',
        description: 'Diverse mix of academic, code, web, books, and forums.',
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

const DatasetPage = ({ showDashboard = true, embedded = false }: DatasetPageProps) => {
  const {
    datasetName,
    selectedCorpus,
    selectedConfig,
    loading,
    error,
    datasetLoaded,
    stats,
    histogram,
    loadProgress,
    validating,
    validationReport,
    validationProgress,
    fileInputRef,
    availableDatasets,
    datasetsLoading,
    activeValidationDataset,
    removingDataset,
    setError,
    handleCorpusChange,
    handleConfigChange,
    handleLoadDataset,
    handleUploadClick,
    handleFileChange,
    handleSelectDataset,
    handleValidateDataset,
    handleDeleteDataset,
  } = useDataset();

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [isInsertByNameOpen, setIsInsertByNameOpen] = useState(false);

  const corpusInputId = 'corpus-input';
  const configInputId = 'config-input';
  const formatNumber = (num: number) => num.toLocaleString();

  const documentHistogram = validationReport?.document_length_histogram ?? histogram;
  const wordHistogram = validationReport?.word_length_histogram ?? null;
  const documentCount = validationReport?.document_count ?? stats?.documentCount ?? null;
  const minDocumentLength = validationReport?.min_document_length ?? stats?.minLength ?? null;
  const maxDocumentLength = validationReport?.max_document_length ?? stats?.maxLength ?? null;

  const handlePresetSelect = (preset: DatasetPreset) => {
    setSelectedPreset(preset.id);
    handleCorpusChange(preset.id);
    handleConfigChange(preset.defaultConfig ?? '');
  };

  const handlePresetDownload = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    void handleLoadDataset();
  };

  const renderHistogram = (
    title: string,
    histogramData: typeof histogram | null,
    emptyLabel: string,
    countLabel: string,
  ) => {
    if (!histogramData || histogramData.counts.length === 0) {
      return (
        <div className="chart-placeholder">
          <p>{title}</p>
          <span>{emptyLabel}</span>
        </div>
      );
    }

    const maxCount = Math.max(...histogramData.counts);

    return (
      <div className="histogram-container">
        <p className="histogram-title">{title}</p>
        <div className="histogram-chart">
          {histogramData.counts.map((count, index) => {
            const heightPercent = maxCount > 0 ? (count / maxCount) * 100 : 0;
            return (
              <div
                key={`${histogramData.bins[index]}-${count}`}
                className="histogram-bar-wrapper"
                title={`${histogramData.bins[index]}: ${formatNumber(count)} ${countLabel}`}
              >
                <div
                  className="histogram-bar"
                  style={{ height: `${heightPercent}%` }}
                />
              </div>
            );
          })}
        </div>
        <div className="histogram-labels">
          <span>{formatNumber(histogramData.min_length)} chars</span>
          <span>{formatNumber(histogramData.max_length)} chars</span>
        </div>
      </div>
    );
  };

  const renderValidationStatus = () => {
    if (validating) {
      const progressLabel =
        validationProgress !== null ? ` (${Math.round(validationProgress)}%)` : '';
      return (
        <div className="loading-container">
          <div className="spinner" />
          <p>Validating dataset{progressLabel}...</p>
          <span>Computing document length and word distribution metrics.</span>
        </div>
      );
    }

    if (validationReport) {
      return null;
    }

    return (
      <div className="chart-placeholder">
        <p>Validation results will appear here.</p>
        <span>Run evaluation from the dataset preview list.</span>
      </div>
    );
  };

  let datasetOverviewDescription = 'Run dataset validation to view results';
  if (validationReport?.dataset_name) {
    datasetOverviewDescription = `Validation results for ${validationReport.dataset_name}`;
  } else if (datasetLoaded && datasetName) {
    datasetOverviewDescription = `Stats for ${datasetName}`;
  }

  const modalDownloadProgress = loadProgress !== null
    ? ` (${Math.round(loadProgress)}%)`
    : '';
  const isPresetListInactive = isInsertByNameOpen;
  const presetsDisabled = loading || isInsertByNameOpen;

  const pageContent = (
    <>
      <div className="page-grid dataset-page-layout">
        <section className="dataset-top-section">
          <div className="dataset-top-row">
            <div className="dataset-intro-panel">
              <p className="panel-label">Dataset Usage</p>
              <p className="panel-description">
                This page is dedicated to downloading and analyzing text datasets from Hugging
                Face. Select a predefined corpus or upload your own file, then validate the
                dataset and inspect the resulting metrics directly in the dashboard.
              </p>
            </div>
            <div className="dataset-top-divider" aria-hidden="true" />
            <div className="dataset-preview-panel">
              <header className="panel-header">
                <div>
                  <p className="panel-label">Dataset Preview</p>
                  <p className="panel-description">
                    Review datasets stored in the database and run validation on demand.
                  </p>
                </div>
                <button
                  type="button"
                  className="icon-button"
                  onClick={() => setIsModalOpen(true)}
                  aria-label="Add dataset"
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
                    No datasets available. Please download or upload a dataset.
                  </div>
                ) : (
                  <div className="dataset-preview-table">
                    {availableDatasets.map((dataset) => {
                      const isValidating = activeValidationDataset === dataset.dataset_name;
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
                            {formatNumber(dataset.document_count)}
                          </span>
                          <div className="dataset-preview-actions">
                            <button
                              type="button"
                              className="icon-button subtle"
                              aria-label="Run dataset evaluation"
                              onClick={(event) => {
                                event.stopPropagation();
                                handleSelectDataset(dataset.dataset_name);
                                void handleValidateDataset(dataset.dataset_name);
                              }}
                              disabled={isValidating || isRemoving}
                            >
                              {isValidating ? (
                                <span className="action-spinner" />
                              ) : (
                                <svg viewBox="0 0 24 24" aria-hidden="true">
                                  <path d="M5 4h14v4H5z" />
                                  <path d="M5 12h14v8H5z" />
                                </svg>
                              )}
                            </button>
                            <button
                              type="button"
                              className="icon-button danger"
                              aria-label="Remove dataset"
                              onClick={(event) => {
                                event.stopPropagation();
                                void handleDeleteDataset(dataset.dataset_name);
                              }}
                              disabled={isValidating || isRemoving}
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
          <aside className="panel dashboard-panel">
            <header className="panel-header">
              <div>
                <p className="panel-label">Dataset Overview</p>
                <p className="panel-description">
                  {datasetOverviewDescription}
                </p>
              </div>
            </header>
            {renderValidationStatus()}
            <div className="dashboard-grid">
              <div className="stat-card">
                <p className="stat-label">Documents</p>
                <p className="stat-value">
                  {documentCount !== null ? formatNumber(documentCount) : '—'}
                </p>
              </div>
              <div className="stat-card">
                <p className="stat-label">Min Length</p>
                <p className="stat-value">
                  {minDocumentLength !== null ? formatNumber(minDocumentLength) : '—'}
                </p>
              </div>
              <div className="stat-card">
                <p className="stat-label">Max Length</p>
                <p className="stat-value">
                  {maxDocumentLength !== null ? formatNumber(maxDocumentLength) : '—'}
                </p>
              </div>
            </div>
            <div className="chart-block">
              {renderHistogram(
                'Document Length Distribution',
                documentHistogram,
                'Load or validate a dataset to see document lengths.',
                'docs',
              )}
            </div>
            <div className="chart-block">
              {renderHistogram(
                'Word Length Distribution',
                wordHistogram,
                'Validate a dataset to see word length distribution.',
                'words',
              )}
            </div>
            <div className="word-frequency-grid">
              <div className="word-frequency-panel">
                <p className="panel-label">Most Common Words</p>
                {validationReport?.most_common_words?.length ? (
                  <ul className="word-frequency-list">
                    {validationReport.most_common_words.map((item) => (
                      <li key={`${item.word}-${item.count}`}>
                        <span>{item.word}</span>
                        <span>{formatNumber(item.count)}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="word-frequency-empty">No validation results yet.</div>
                )}
              </div>
              <div className="word-frequency-panel">
                <p className="panel-label">Least Common Words</p>
                {validationReport?.least_common_words?.length ? (
                  <ul className="word-frequency-list">
                    {validationReport.least_common_words.map((item) => (
                      <li key={`${item.word}-${item.count}`}>
                        <span>{item.word}</span>
                        <span>{formatNumber(item.count)}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="word-frequency-empty">No validation results yet.</div>
                )}
              </div>
            </div>
          </aside>
        )}
      </div>

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
              <div
                className={`dataset-insert-row-shell${isInsertByNameOpen ? ' is-open' : ''}`}
                aria-hidden={!isInsertByNameOpen}
              >
                <div className="dataset-insert-row">
                  <input
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
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M12 3v12" strokeWidth="2" strokeLinecap="round" />
                      <path d="M7 10l5 5 5-5" strokeWidth="2" strokeLinecap="round" />
                      <path d="M5 19h14" strokeWidth="2" strokeLinecap="round" />
                    </svg>
                  </button>
                </div>
              </div>
              <div
                className={`dataset-preset-list-shell${isPresetListInactive ? ' is-inactive' : ''}`}
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
