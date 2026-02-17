import { useMemo, useState } from 'react';
import type { KeyboardEvent, MouseEvent } from 'react';
import { useTokenizers } from '../contexts/TokenizersContext';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

type TokenizersPageProps = {
  showDashboard?: boolean;
  embedded?: boolean;
};

const TokenizersPage = ({ showDashboard = true, embedded = false }: TokenizersPageProps) => {
  const {
    scanInProgress,
    scanError,
    downloadInProgress,
    downloadProgress,
    downloadWarning,
    fetchedTokenizers,
    selectedTokenizer,
    tokenizers,
    customTokenizerName,
    customTokenizerUploading,
    maxDocuments,
    availableDatasets,
    selectedDataset,
    datasetsLoading,
    benchmarkInProgress,
    benchmarkError,
    benchmarkResult,
    benchmarkProgress,
    activeOpeningTokenizer,
    tokenizerReport,
    customTokenizerInputRef,
    setSelectedTokenizer,
    setTokenizers,
    setMaxDocuments,
    setSelectedDataset,
    setScanError,
    setDownloadWarning,
    setBenchmarkError,
    addTokenizer,
    downloadTokenizers,
    handleScan,
    handleRunBenchmarks,
    handleOpenTokenizerReport,
    refreshDatasets,
    handleUploadCustomTokenizer,
    handleClearCustomTokenizer,
    triggerCustomTokenizerUpload,
  } = useTokenizers();

  const [isTokenizerModalOpen, setIsTokenizerModalOpen] = useState(false);
  const [manualTokenizerInput, setManualTokenizerInput] = useState('');
  const [selectedScannedTokenizers, setSelectedScannedTokenizers] = useState<string[]>([]);

  const isEmbeddedSelectionLayout = embedded && !showDashboard;

  const chartStats = useMemo(
    () => [
      { label: 'Queued runs', value: tokenizers.length + (customTokenizerName ? 1 : 0) },
      {
        label: 'Avg. throughput',
        value: benchmarkResult?.global_metrics?.[0]?.tokenization_speed_tps
          ? `${Math.round(benchmarkResult.global_metrics[0].tokenization_speed_tps).toLocaleString()} tok/s`
          : '0 tok/s'
      },
      { label: 'Custom tokenizer', value: customTokenizerName ? 'loaded' : 'none' },
    ],
    [tokenizers.length, customTokenizerName, benchmarkResult],
  );

  const vocabularyChartData = useMemo(() => {
    if (!benchmarkResult?.chart_data?.vocabulary_stats) return [];
    return benchmarkResult.chart_data.vocabulary_stats.map((stat) => ({
      name: stat.tokenizer.split('/').pop() || stat.tokenizer,
      'Vocabulary Size': stat.vocabulary_size,
      Subwords: stat.subwords_count,
      'True Words': stat.true_words_count,
    }));
  }, [benchmarkResult]);

  const speedChartData = useMemo(() => {
    if (!benchmarkResult?.chart_data?.speed_metrics) return [];
    return benchmarkResult.chart_data.speed_metrics.map((stat) => ({
      name: stat.tokenizer.split('/').pop() || stat.tokenizer,
      'Tokens/sec': Math.round(stat.tokens_per_second),
      'Chars/sec': Math.round(stat.chars_per_second),
    }));
  }, [benchmarkResult]);

  const sortedFetchedTokenizers = useMemo(
    () => [...fetchedTokenizers].sort((a, b) => a.localeCompare(b)),
    [fetchedTokenizers],
  );

  const manualTokenizerIds = useMemo(
    () => manualTokenizerInput.split('\n').map((tokenizerId) => tokenizerId.trim()).filter(Boolean),
    [manualTokenizerInput],
  );

  const downloadedTokenizerSet = useMemo(() => new Set(tokenizers), [tokenizers]);

  const handleScannedTokenizerSelection = (
    event: MouseEvent<HTMLDivElement>,
    tokenizerId: string,
  ) => {
    setSelectedScannedTokenizers((current) => {
      if (event.ctrlKey || event.metaKey) {
        if (current.includes(tokenizerId)) {
          return current.filter((item) => item !== tokenizerId);
        }
        return [...current, tokenizerId];
      }
      return [tokenizerId];
    });
  };

  const handleScannedTokenizerSelectionByKeyboard = (
    event: KeyboardEvent<HTMLDivElement>,
    tokenizerId: string,
  ) => {
    if (event.key !== 'Enter' && event.key !== ' ') {
      return;
    }
    event.preventDefault();
    setSelectedScannedTokenizers([tokenizerId]);
  };

  const handleRemoveScannedSelection = (
    event: MouseEvent<HTMLButtonElement>,
    tokenizerId: string,
  ) => {
    event.stopPropagation();
    setSelectedScannedTokenizers((current) => current.filter((item) => item !== tokenizerId));
  };

  const runManualDownload = () => {
    void downloadTokenizers(manualTokenizerIds);
  };

  const runScannedDownload = () => {
    void downloadTokenizers(selectedScannedTokenizers);
  };

  const pageContent = (
    <div className={`page-grid tokenizers-page${showDashboard ? '' : ' tokenizers-page--single'}`}>
      <section className="panel large-panel">
        <header className="panel-header">
          <div>
            <p className="panel-label">Select tokenizers</p>
            <p className="panel-description">
              Manage Hugging Face tokenizer IDs in a single text area and fetch additional IDs via scan.
            </p>
          </div>
          <div className="tokenizer-select">
            <select
              value={selectedTokenizer}
              onChange={(event) => setSelectedTokenizer(event.target.value)}
              className="text-input"
            >
              {fetchedTokenizers.length === 0 ? (
                <option value="">Click Scan to fetch tokenizers</option>
              ) : (
                fetchedTokenizers.map((tokenizer) => (
                  <option key={tokenizer} value={tokenizer}>
                    {tokenizer}
                  </option>
                ))
              )}
            </select>
            <button
              type="button"
              className="primary-button ghost"
              onClick={() => addTokenizer(selectedTokenizer)}
              disabled={!selectedTokenizer}
            >
              Add
            </button>
            <button
              type="button"
              className="primary-button"
              onClick={handleScan}
              disabled={scanInProgress}
            >
              {scanInProgress ? 'Scanning...' : 'Scan'}
            </button>
          </div>
        </header>
        <div className="panel-body">
          {scanError && (
            <div className="error-banner" role="alert">
              <span>{scanError}</span>
              <button type="button" aria-label="Dismiss" onClick={() => setScanError(null)}>
                ×
              </button>
            </div>
          )}
          {benchmarkError && (
            <div className="error-banner" role="alert">
              <span>{benchmarkError}</span>
              <button type="button" aria-label="Dismiss" onClick={() => setBenchmarkError(null)}>
                ×
              </button>
            </div>
          )}
          <textarea
            className="text-area"
            placeholder="Paste tokenizer IDs here, one per line..."
            value={tokenizers.join('\n')}
            onChange={(event) =>
              setTokenizers(event.target.value.split('\n').filter((tokenizerId) => tokenizerId.trim()))
            }
          />
          <div className="field-row">
            <label className="field-label" htmlFor="datasetSelect">
              Dataset:
            </label>
            <select
              id="datasetSelect"
              value={selectedDataset}
              onChange={(event) => setSelectedDataset(event.target.value)}
              className="text-input"
              disabled={datasetsLoading}
            >
              {availableDatasets.length === 0 ? (
                <option value="">Click Refresh to load</option>
              ) : (
                availableDatasets.map((datasetName) => (
                  <option key={datasetName} value={datasetName}>
                    {datasetName}
                  </option>
                ))
              )}
            </select>
            <button
              type="button"
              className="primary-button ghost"
              onClick={refreshDatasets}
              disabled={datasetsLoading}
            >
              {datasetsLoading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
          <div className="field-row">
            <label className="field-label" htmlFor="maxDocs">
              Max documents:
            </label>
            <input
              id="maxDocs"
              type="number"
              className="text-input"
              value={maxDocuments}
              onChange={(event) => setMaxDocuments(Number(event.target.value))}
              min={0}
            />
          </div>
          <div className="checkbox-group">
            <div className="custom-tokenizer-row">
              <input
                type="file"
                ref={customTokenizerInputRef}
                onChange={handleUploadCustomTokenizer}
                accept=".json"
                style={{ display: 'none' }}
              />
              <button
                type="button"
                className="primary-button ghost"
                onClick={triggerCustomTokenizerUpload}
                disabled={customTokenizerUploading}
              >
                {customTokenizerUploading ? 'Uploading...' : 'Upload tokenizer.json'}
              </button>
              {customTokenizerName && (
                <span className="custom-tokenizer-badge">
                  {customTokenizerName}
                  <button
                    type="button"
                    className="clear-button"
                    onClick={handleClearCustomTokenizer}
                    aria-label="Clear custom tokenizer"
                  >
                    ×
                  </button>
                </span>
              )}
            </div>
          </div>
        </div>
        <footer className="panel-footer">
          <button
            type="button"
            className="primary-button"
            onClick={handleRunBenchmarks}
            disabled={benchmarkInProgress || (tokenizers.length === 0 && !customTokenizerName) || !selectedDataset}
          >
            {benchmarkInProgress ? 'Running benchmarks...' : 'Run Benchmarks'}
          </button>
        </footer>
      </section>
      {showDashboard && (
        <aside className="panel dashboard-panel">
          <header className="panel-header">
            <div>
              <p className="panel-label">Benchmark Dashboard</p>
              <p className="panel-description">
                {benchmarkResult
                  ? `Processed ${benchmarkResult.documents_processed} documents with ${benchmarkResult.tokenizers_count} tokenizers`
                  : 'Charts will appear after running benchmarks.'}
              </p>
            </div>
          </header>

          {vocabularyChartData.length > 0 && (
            <div className="chart-container">
              <h3 className="chart-title">Vocabulary Size Comparison</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={vocabularyChartData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis type="number" stroke="#888" />
                  <YAxis dataKey="name" type="category" stroke="#888" width={100} tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #333' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Bar dataKey="Vocabulary Size" fill="#4fc3f7" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {speedChartData.length > 0 && (
            <div className="chart-container">
              <h3 className="chart-title">Tokenization Speed</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={speedChartData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis type="number" stroke="#888" />
                  <YAxis dataKey="name" type="category" stroke="#888" width={100} tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #333' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Bar dataKey="Tokens/sec" fill="#81c784" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="Chars/sec" fill="#ffb74d" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="dashboard-grid">
            {chartStats.map((stat) => (
              <div key={stat.label} className="stat-card">
                <p className="stat-label">{stat.label}</p>
                <p className="stat-value">{stat.value}</p>
              </div>
            ))}
          </div>

          {!benchmarkResult && (
            <div className="chart-empty-state">
              {benchmarkInProgress ? (
                <>
                  <div className="spinner" />
                  <p>
                    Processing benchmarks
                    {benchmarkProgress !== null ? ` (${Math.round(benchmarkProgress)}%)` : ''}...
                  </p>
                  <span>Analyzing tokenizers and computing metrics. This may take a few minutes.</span>
                </>
              ) : (
                <p>Run benchmarks to see charts.</p>
              )}
            </div>
          )}

          <ul className="insights-list">
            <li>Tokenizer list stays synchronized with the text area for quick edits and bulk paste.</li>
            {benchmarkResult && (
              <li>Benchmarks complete! Charts are now interactive - hover for details.</li>
            )}
          </ul>
        </aside>
      )}
    </div>
  );

  if (!isEmbeddedSelectionLayout) {
    if (embedded) {
      return pageContent;
    }

    return <div className="page-scroll">{pageContent}</div>;
  }

  return (
    <>
      <div className="page-grid tokenizers-page tokenizers-page--single tokenizers-embedded-layout">
        <section className="tokenizer-top-section">
          <div className="tokenizer-top-row">
            <div className="tokenizer-intro-panel">
              <p className="panel-label">Tokenizer Selection</p>
              <p className="panel-description">
                Manage tokenizer identifiers and custom JSON tokenizers from a single popup window.
              </p>
            </div>
            <div className="dataset-top-divider" aria-hidden="true" />
            <div className="tokenizer-preview-panel">
              <header className="panel-header">
                <div>
                  <p className="panel-label">Tokenizer Preview</p>
                  <p className="panel-description">
                    Review selected tokenizers and open persisted reports (auto-generates if missing).
                  </p>
                </div>
                <button
                  type="button"
                  className="icon-button"
                  onClick={() => setIsTokenizerModalOpen(true)}
                  aria-label="Add tokenizer"
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 5v14M5 12h14" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                </button>
              </header>
              <div className="tokenizer-preview-body">
                {tokenizers.length === 0 ? (
                  <div className="dataset-preview-empty">
                    No tokenizers selected. Click + to add tokenizers.
                  </div>
                ) : (
                  <div className="tokenizer-preview-list">
                    {tokenizers.map((tokenizerId) => (
                      <div key={tokenizerId} className="tokenizer-preview-row">
                        <span className="tokenizer-preview-name">{tokenizerId}</span>
                        <div className="tokenizer-preview-actions">
                          <button
                            type="button"
                            className="icon-button subtle"
                            aria-label={`Open tokenizer report for ${tokenizerId}`}
                            title="Open tokenizer report (loads latest or generates if missing)"
                            onClick={() => void handleOpenTokenizerReport(tokenizerId)}
                            disabled={
                              activeOpeningTokenizer === tokenizerId
                            }
                          >
                            {activeOpeningTokenizer === tokenizerId ? (
                              <span className="action-spinner" />
                            ) : (
                              <svg viewBox="0 0 24 24" aria-hidden="true">
                                <path d="M5 4h14v4H5z" />
                                <path d="M5 12h14v8H5z" />
                              </svg>
                            )}
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {scanError && (
            <div className="error-banner" role="alert">
              <span>{scanError}</span>
              <button type="button" aria-label="Dismiss" onClick={() => setScanError(null)}>
                ×
              </button>
            </div>
          )}
          {benchmarkError && (
            <div className="error-banner" role="alert">
              <span>{benchmarkError}</span>
              <button type="button" aria-label="Dismiss" onClick={() => setBenchmarkError(null)}>
                ×
              </button>
            </div>
          )}
          {tokenizerReport && (
            <div className="tokenizer-report-hint">
              Latest loaded report: {tokenizerReport.tokenizer_name}
            </div>
          )}

        </section>
      </div>

      {isTokenizerModalOpen && (
        <div className="modal-overlay" role="dialog" aria-modal="true">
          <div className="tokenizer-modal-window">
            <header className="tokenizer-modal-header">
              <div>
                <p className="panel-label">Tokenizer Manager</p>
                <p className="panel-description">Download tokenizer IDs from text input or Hugging Face scan results.</p>
              </div>
              <button
                type="button"
                className="secondary-button"
                onClick={() => setIsTokenizerModalOpen(false)}
              >
                Close
              </button>
            </header>

            {(downloadWarning || scanError || benchmarkError) && (
              <div className="tokenizer-modal-messages">
                {downloadWarning && (
                  <div className="error-banner tokenizer-warning-banner" role="status">
                    <span>{downloadWarning}</span>
                    <button type="button" aria-label="Dismiss" onClick={() => setDownloadWarning(null)}>
                      ×
                    </button>
                  </div>
                )}
                {scanError && (
                  <div className="error-banner" role="alert">
                    <span>{scanError}</span>
                    <button type="button" aria-label="Dismiss" onClick={() => setScanError(null)}>
                      ×
                    </button>
                  </div>
                )}
                {benchmarkError && (
                  <div className="error-banner" role="alert">
                    <span>{benchmarkError}</span>
                    <button type="button" aria-label="Dismiss" onClick={() => setBenchmarkError(null)}>
                      ×
                    </button>
                  </div>
                )}
              </div>
            )}

            <div className="tokenizer-modal-columns">
              <div className="tokenizer-modal-column tokenizer-modal-column--manual">
                <div className="tokenizer-modal-column-header">
                  <p className="panel-label">Manual Input</p>
                  <div className="tokenizer-column-actions">
                    <button
                      type="button"
                      className="icon-button subtle tokenizer-column-action"
                      onClick={runManualDownload}
                      disabled={manualTokenizerIds.length === 0 || downloadInProgress}
                      aria-label="Download tokenizers from manual input"
                      title="Download manual tokenizer names"
                    >
                      {downloadInProgress ? (
                        <span className="action-spinner" aria-hidden="true" />
                      ) : (
                        <svg viewBox="0 0 24 24" aria-hidden="true">
                          <path d="M12 4v11m0 0-4-4m4 4 4-4M5 19h14" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      )}
                    </button>
                    <button
                      type="button"
                      className="icon-button subtle tokenizer-column-action"
                      onClick={triggerCustomTokenizerUpload}
                      disabled={customTokenizerUploading}
                      aria-label="Upload custom tokenizer JSON"
                      title="Upload custom tokenizer JSON"
                    >
                      {customTokenizerUploading ? (
                        <span className="action-spinner" aria-hidden="true" />
                      ) : (
                        <svg viewBox="0 0 24 24" aria-hidden="true">
                          <path d="M12 20V9m0 0-4 4m4-4 4 4M5 5h14" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      )}
                    </button>
                  </div>
                </div>
                <p className="panel-description tokenizer-modal-column-description">
                  Add one tokenizer name per line, then download all entries in a single batch or upload a custom tokenizer JSON.
                </p>

                <div className="tokenizer-modal-column-content">
                  <textarea
                    className="tokenizer-manual-input"
                    value={manualTokenizerInput}
                    onChange={(event) => setManualTokenizerInput(event.target.value)}
                    placeholder="Tokenizer names, one per line"
                  />
                  <input
                    type="file"
                    ref={customTokenizerInputRef}
                    onChange={handleUploadCustomTokenizer}
                    accept=".json"
                    style={{ display: 'none' }}
                  />
                  {customTokenizerName && (
                    <span className="custom-tokenizer-badge">
                      {customTokenizerName}
                      <button
                        type="button"
                        className="clear-button"
                        onClick={handleClearCustomTokenizer}
                        aria-label="Clear custom tokenizer"
                      >
                        ×
                      </button>
                    </span>
                  )}
                </div>
              </div>

              <div className="tokenizer-modal-column tokenizer-modal-column--scan">
                <div className="tokenizer-modal-column-header">
                  <p className="panel-label">Hugging Face Selection</p>
                  <div className="tokenizer-column-actions">
                    <button
                      type="button"
                      className="icon-button subtle tokenizer-column-action"
                      onClick={handleScan}
                      disabled={scanInProgress}
                      aria-label="Scan Hugging Face tokenizers"
                      title="Scan Hugging Face tokenizer IDs"
                    >
                      {scanInProgress ? (
                        <span className="action-spinner" aria-hidden="true" />
                      ) : (
                        <svg viewBox="0 0 24 24" aria-hidden="true">
                          <circle cx="11" cy="11" r="6" fill="none" strokeWidth="2" />
                          <path d="M16 16l4 4" strokeWidth="2" strokeLinecap="round" />
                        </svg>
                      )}
                    </button>
                    <button
                      type="button"
                      className="icon-button subtle tokenizer-column-action"
                      onClick={runScannedDownload}
                      disabled={selectedScannedTokenizers.length === 0 || downloadInProgress}
                      aria-label="Download selected scanned tokenizers"
                      title="Download selected scanned tokenizers"
                    >
                      {downloadInProgress ? (
                        <span className="action-spinner" aria-hidden="true" />
                      ) : (
                        <svg viewBox="0 0 24 24" aria-hidden="true">
                          <path d="M12 4v11m0 0-4-4m4 4 4-4M5 19h14" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      )}
                    </button>
                  </div>
                </div>
                <p className="panel-description tokenizer-modal-column-description">
                  Fetch tokenizer IDs from the Hugging Face index and select one or more rows to download.
                </p>

                <div className="tokenizer-modal-column-content">
                  <div className="tokenizer-scan-list" role="listbox" aria-multiselectable="true">
                    {sortedFetchedTokenizers.length === 0 ? (
                      <div className="tokenizer-scan-empty">Run scan to load Hugging Face tokenizer IDs.</div>
                    ) : (
                      sortedFetchedTokenizers.map((tokenizerId) => {
                        const isSelected = selectedScannedTokenizers.includes(tokenizerId);
                        const isDownloaded = downloadedTokenizerSet.has(tokenizerId);

                        return (
                          <div
                            key={tokenizerId}
                            role="option"
                            aria-selected={isSelected}
                            tabIndex={0}
                            className={`tokenizer-scan-row${isSelected ? ' selected' : ''}${isDownloaded ? ' downloaded' : ''}`}
                            onClick={(event) => handleScannedTokenizerSelection(event, tokenizerId)}
                            onKeyDown={(event) => handleScannedTokenizerSelectionByKeyboard(event, tokenizerId)}
                          >
                            <span className="tokenizer-scan-id">{tokenizerId}</span>
                            {isSelected && (
                              <button
                                type="button"
                                className="icon-button subtle tokenizer-selection-remove"
                                aria-label={`Remove ${tokenizerId} from selection`}
                                onClick={(event) => handleRemoveScannedSelection(event, tokenizerId)}
                              >
                                <svg viewBox="0 0 24 24" aria-hidden="true">
                                  <path d="M5 12h14" strokeWidth="2" strokeLinecap="round" />
                                </svg>
                              </button>
                            )}
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              </div>
            </div>

            <div className="tokenizer-modal-progress-row" role="status" aria-live="polite">
              <span className="tokenizer-modal-progress-label">
                Download progress: {downloadProgress !== null ? `${Math.round(downloadProgress)}%` : '0%'}
              </span>
              <div className="tokenizer-modal-progress-track" aria-hidden="true">
                <div
                  className="tokenizer-modal-progress-fill"
                  style={{ width: `${downloadProgress ?? 0}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default TokenizersPage;
